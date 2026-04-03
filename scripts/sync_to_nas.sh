#!/bin/bash
# G12-next: Project DNA Data Sync → NAS Mirror
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../deploy/.env"
LOG_DIR="$SCRIPT_DIR/../logs"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d).log"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
TMP_DIR="/tmp/dna_sync_$TIMESTAMP"
# Безопасный парсинг .env (каждую переменную отдельно, спецсимволы не пугают)
get_env() { grep "^${1}=" "$ENV_FILE" | head -1 | cut -d= -f2-; }
NAS_HOST=$(get_env NAS_HOST)
NAS_USER=$(get_env NAS_USER)
NAS_SSH_PASS=$(get_env NAS_SSH_PASS)
PG_USER=$(get_env PG_USER)
PG_PASSWORD=$(get_env PG_PASSWORD)
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
        log "  🪣  $BUCKET..."
        mc mb --ignore-existing "nas/$BUCKET" 2>/dev/null || true
        mc mirror --overwrite "local/$BUCKET" "nas/$BUCKET" 2>&1 | \
            grep -vE "^$|Calculating" || true
        log_ok "$BUCKET done"
    done
}
log "══════════════════════════════════════"
log "🚀 Sync Start: $TIMESTAMP"
log "   Target: $NAS_USER@$NAS_HOST"
log "══════════════════════════════════════"
ERRORS=0
sync_postgres || { log_err "PostgreSQL FAILED"; ERRORS=$((ERRORS+1)); }
sync_qdrant   || { log_err "Qdrant FAILED";     ERRORS=$((ERRORS+1)); }
sync_minio    || { log_err "MinIO FAILED";      ERRORS=$((ERRORS+1)); }
rm -rf "$TMP_DIR"
log "══════════════════════════════════════"
[ "$ERRORS" -eq 0 ] \
    && log_ok "All synced! Log: $LOG_FILE" \
    || { log_err "$ERRORS component(s) failed"; exit 1; }
