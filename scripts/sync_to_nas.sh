#!/bin/bash
# G12-next: Project DNA Data Sync → NAS + ntfy.sh alerts
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../deploy/.env"
LOG_DIR="$SCRIPT_DIR/../logs"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d).log"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
TMP_DIR="/tmp/dna_sync_$TIMESTAMP"
get_env() { grep "^${1}=" "$ENV_FILE" | head -1 | cut -d= -f2-; }
NAS_HOST=$(get_env NAS_HOST)
NAS_USER=$(get_env NAS_USER)
NAS_SSH_PASS=$(get_env NAS_SSH_PASS)
PG_USER=$(get_env PG_USER)
PG_PASSWORD=$(get_env PG_PASSWORD)
NTFY_TOPIC=$(get_env NTFY_TOPIC)
QDRANT_COLLECTIONS=("project_prompts" "test_collection")
PG_CONTAINER="ai-postgres"
PG_DB="project_dna"
NAS_BACKUP="/volume1/docker/project_dna_sync"
QDRANT_URL="http://localhost:6333"
KEEP_BACKUPS=7
mkdir -p "$LOG_DIR" "$TMP_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
log()     { echo "[$(date +%H:%M:%S)] $1"; }
log_ok()  { echo "[$(date +%H:%M:%S)] ✅ $1"; }
log_warn(){ echo "[$(date +%H:%M:%S)] ⚠️  $1"; }
log_err() { echo "[$(date +%H:%M:%S)] ❌ $1"; }
nas_rsync() { sshpass -p "$NAS_SSH_PASS" rsync -avz "$1" "$NAS_USER@$NAS_HOST:$2"; }
nas_ssh()   { sshpass -p "$NAS_SSH_PASS" ssh "$NAS_USER@$NAS_HOST" "$@"; }
# ntfy.sh notification
notify() {
    local title="$1" msg="$2" priority="${3:-default}" tags="${4:-floppy_disk}"
    curl -s --max-time 8 \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Tags: $tags" \
        -d "$msg" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
}
sync_postgres() {
    log "📊 PostgreSQL backup..."
    local dump="$TMP_DIR/${PG_DB}_${TIMESTAMP}.sql.gz"
    docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
        pg_dump -U "$PG_USER" "$PG_DB" | gzip > "$dump"
    nas_rsync "$dump" "$NAS_BACKUP/postgres/"
    nas_ssh "ls -t '$NAS_BACKUP/postgres/'*.sql.gz 2>/dev/null | \
        tail -n +$((KEEP_BACKUPS+1)) | xargs rm -f 2>/dev/null || true"
    log_ok "PostgreSQL: $(du -sh "$dump" | cut -f1) → NAS"
}
sync_qdrant() {
    log "🔍 Qdrant snapshots..."
    for COLLECTION in "${QDRANT_COLLECTIONS[@]}"; do
        log "  📸 $COLLECTION..."
        SNAP_NAME=$(curl -sf -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots" | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('result',{}).get('name',''))" 2>/dev/null || echo "")
        if [ -z "$SNAP_NAME" ]; then
            log_warn "$COLLECTION: snapshot failed, skip"
            continue
        fi
        local snap_file="$TMP_DIR/${COLLECTION}.snapshot"
        curl -sf "$QDRANT_URL/collections/$COLLECTION/snapshots/$SNAP_NAME" -o "$snap_file"
        nas_rsync "$snap_file" "$NAS_BACKUP/qdrant/"
        curl -sf -X DELETE "$QDRANT_URL/collections/$COLLECTION/snapshots/$SNAP_NAME" > /dev/null || true
        log_ok "$COLLECTION: $(du -sh "$snap_file" | cut -f1) → NAS"
    done
}
sync_minio() {
    log "📦 MinIO mirror..."
    for BUCKET in $(mc ls local/ 2>/dev/null | awk '{print $NF}' | tr -d '/'); do
        mc mb --ignore-existing "nas/$BUCKET" 2>/dev/null || true
        mc mirror --overwrite "local/$BUCKET" "nas/$BUCKET" 2>&1 | grep -vE "^$|Calculating" || true
        log_ok "$BUCKET mirrored"
    done
}
log "══════════════════════════════════════"
log "🚀 Sync Start: $TIMESTAMP"
log "══════════════════════════════════════"
ERRORS=0
FAILED_COMPONENTS=""
sync_postgres || {
    log_err "PostgreSQL FAILED"
    ERRORS=$((ERRORS+1))
    FAILED_COMPONENTS="${FAILED_COMPONENTS} PostgreSQL"
}
sync_qdrant   || {
    log_err "Qdrant FAILED"
    ERRORS=$((ERRORS+1))
    FAILED_COMPONENTS="${FAILED_COMPONENTS} Qdrant"
}
sync_minio    || {
    log_err "MinIO FAILED"
    ERRORS=$((ERRORS+1))
    FAILED_COMPONENTS="${FAILED_COMPONENTS} MinIO"
}
rm -rf "$TMP_DIR"
log "══════════════════════════════════════"
if [ "$ERRORS" -eq 0 ]; then
    log_ok "All synced!"
    # Тихий успех — только раз в сутки (в 02:00) шлём OK
    if [ "$(date +%H)" = "02" ]; then
        notify "✅ DNA Sync OK" "Ежедневный отчёт: PostgreSQL + Qdrant + MinIO → NAS. $(date '+%d.%m %H:%M')" "low" "white_check_mark"
    fi
else
    log_err "$ERRORS component(s) failed:$FAILED_COMPONENTS"
    notify "❌ DNA Sync FAILED" "Сбой синхронизации:${FAILED_COMPONENTS}. $(date '+%d.%m %H:%M') Проверь: $LOG_FILE" "urgent" "rotating_light"
    exit 1
fi
