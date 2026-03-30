#!/bin/bash
# MLOps Project DNA - Disaster Recovery Backup Script
# Вызывать из корневой папки, где лежит docker-compose.yml (например, ~/ai-design-workspace/deploy)
# Дата: 2026-03-29

# Папка назначения (если /opt недоступна без sudo, можно изменить на ~/backups)
BACKUP_DIR="/opt/backups/dna_$(date +%Y%m%d_%H%M%S)"
sudo mkdir -p "$BACKUP_DIR"

echo "=== 1. Starting Database Logical Backup (Zero Downtime) ==="
# Узнаем под каким юзером крутится база (берем из ENV контейнера)
PG_USER=$(docker exec ai-postgres printenv POSTGRES_USER)
if [ -z "$PG_USER" ]; then PG_USER="postgres"; fi
# Выполняем дамп работающей базы Postgres.
docker exec ai-postgres pg_dump -U "$PG_USER" ai_metadata | sudo tee "$BACKUP_DIR/ai_metadata_dump.sql" > /dev/null

echo "=== 2. Stopping Containers for Consistent Volume Snapshots ==="
# Чтобы база векторов и медиафайлы не сломались (in-flight state), делаем паузу
# Используем ИМЕНА СЕРВИСОВ (llm-router, open-webui), а не имена контейнеров (ai-router)
docker compose stop llm-router open-webui qdrant minio n8n

echo "=== 3. Backing up Docker Volumes ==="
# В Docker-compose префикс вольюмов по умолчанию равен имени папки (без спецсимволов).
# Папка `deploy` -> префикс `deploy_`
PREFIX=$(basename "$PWD" | tr -d '-' | tr -d '_')
# Но часто compose оставляет тире, зависит от версии. Safest way is to define an exact prefix:
PREFIX="deploy"

# Функция для "чистого" архивирования Docker Named Volume
backup_volume() {
    VOL_SUFFIX=$1
    VOLUME_NAME="${PREFIX}_${VOL_SUFFIX}"
    echo "Backing up volume $VOLUME_NAME..."
    
    # Запускаем крошечный контейнер Alpine, монтируем в него Volume и папку бакапа, пакуем в tar
    docker run --rm \
        -v "$VOLUME_NAME":/data:ro \
        -v "$BACKUP_DIR":/backup \
        alpine tar -czf "/backup/${VOL_SUFFIX}.tar.gz" -C /data .
}

backup_volume "pg_data"      # Физический бэкап Postgres (страховка от кривых имен баз)
backup_volume "minio_data"
backup_volume "qdrant_data" 
backup_volume "n8n_data"
backup_volume "webui_data"

echo "=== 4. Backing up Configs ==="
sudo cp docker-compose.yml "$BACKUP_DIR/"
sudo cp .env "$BACKUP_DIR/" 2>/dev/null

echo "=== 5. Restarting Containers ==="
docker compose start

echo "=== 6. Packaging Target Archive ==="
cd /opt/backups
sudo tar -czf "${BACKUP_DIR}.tar.gz" -C "$BACKUP_DIR" .
sudo rm -rf "$BACKUP_DIR"

echo "✅ Backup Complete! Archive located at: ${BACKUP_DIR}.tar.gz"
echo "--------------------------------------------------------"
echo "Для отправки на XPenology NAS можно раскомментировать строку в скрипте:"
# scp "${BACKUP_DIR}.tar.gz" admin@xpenology_ip:/volume1/backups/mlops/

