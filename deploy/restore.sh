#!/bin/bash
# MLOps Project DNA - Disaster Recovery Restore Script
# Скрипт полного восстановления системы "с нуля"
# Использование: sudo ./restore.sh /opt/backups/dna_20260329_120000.tar.gz
# Дата: 2026-03-29

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Ошибка: Укажите полный путь к `.tar.gz` архиву бэкапа."
    echo "Пример: sudo ./restore.sh /opt/backups/dna_20260329.tar.gz"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Ошибка: Файл архива $BACKUP_FILE не найден."
    exit 1
fi

RESTORE_TMP="/opt/backups/restore_tmp"
sudo mkdir -p "$RESTORE_TMP"

echo "=== 1. Unpacking Archive ==="
sudo tar -xzf "$BACKUP_FILE" -C "$RESTORE_TMP"

echo "=== 2. Destructive Cleanup (Wiping Existing Environment) ==="
# Полностью гасим контейнеры и УДАЛЯЕМ старые вольюмы
# -v удалит все данные баз (Qdrant, MinIO, N8n).
docker compose down -v

echo "=== 3. Restoring Configs ==="
# На всякий случай накатываем docker-compose, который лежал в бэкапе
if [ -f "$RESTORE_TMP/docker-compose.yml" ]; then
    sudo cp "$RESTORE_TMP/docker-compose.yml" .
fi
if [ -f "$RESTORE_TMP/.env" ]; then
    sudo cp "$RESTORE_TMP/.env" .
fi

echo "=== 4. Re-creating Empty Docker Volumes ==="
# Поднимаем контейнеры без запуска, чтобы Docker просто создал пустые вольюмы нужного типа
docker compose create qdrant minio n8n open-webui

echo "=== 5. Injecting Data into Reborn Volumes ==="
# Префикс папки
PREFIX="deploy"

restore_volume() {
    VOL_SUFFIX=$1
    VOLUME_NAME="${PREFIX}_${VOL_SUFFIX}"
    BACKUP_TAR="$RESTORE_TMP/${VOL_SUFFIX}.tar.gz"
    
    if [ -f "$BACKUP_TAR" ]; then
        echo "Restoring $VOLUME_NAME..."
        # Монтируем временно пустой Volume в Alpine
        # Затем заходим в директорию /data, чистим ее и распаковываем туда бэкап
        docker run --rm \
            -v "$VOLUME_NAME":/data \
            -v "$RESTORE_TMP":/backup \
            alpine sh -c "cd /data && rm -rf ./* && tar -xzf /backup/${VOL_SUFFIX}.tar.gz"
    else
        echo "Внимание [Желтый]: Бэкап для $VOLUME_NAME не найден (пропуск)."
    fi
}

restore_volume "pg_data"
restore_volume "minio_data"
restore_volume "qdrant_data" 
restore_volume "n8n_data"
restore_volume "webui_data"

echo "=== 6. PostgreSQL Logical Restore ==="
# Поднимаем только пустой PostgreСУБД
docker compose up -d postgres
echo "Ожидаем 10 секунд пока новая, девственно чистая PostgreSQL инициализируется (Entrypoint scripts)..."
sleep 15

# Попытаться пересоздать базу на случай, если она не создалась
PG_USER=$(docker exec ai-postgres printenv POSTGRES_USER)
if [ -z "$PG_USER" ]; then PG_USER="postgres"; fi

docker exec -i ai-postgres psql -U "$PG_USER" -c "CREATE DATABASE ai_metadata;" 2>/dev/null

# Восстанавливаем логический дамп
echo "Заливаем таблицы, данные и сиквенсы с помощью psql..."
sudo cat "$RESTORE_TMP/ai_metadata_dump.sql" | docker exec -i ai-postgres psql -U "$PG_USER" -d ai_metadata

echo "=== 7. Starting the Full AI Stack ==="
sudo rm -rf "$RESTORE_TMP"
docker compose up -d

echo "✅ Disaster Recovery Successful! System is back online."
echo "Проверьте дашборд и Open WebUI. Если что-то пошло не так, логи тут: docker-compose logs."

