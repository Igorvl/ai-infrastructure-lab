#!/bin/bash
# G12-alerts: NAS Mirror healthcheck → ntfy.sh alert
# Cron: */30 * * * *
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
get_env() { grep "^${1}=" "$SCRIPT_DIR/../deploy/.env" | head -1 | cut -d= -f2-; }
NTFY_TOPIC=$(get_env NTFY_TOPIC)
NAS_HOST=$(get_env NAS_HOST)
ALERT_LOCK="/tmp/nas_down_alerted"
SERVICES=(
    "NAS Router|http://${NAS_HOST}:8000/v1/dna/health"
    "NAS MinIO|http://${NAS_HOST}:9002/minio/health/live"
    "NAS Qdrant|http://${NAS_HOST}:6334/healthz"
)
notify_alert() {
    curl -s --max-time 8 \
        -H "Title: ⚠️ NAS Mirror DOWN" \
        -H "Priority: urgent" \
        -H "Tags: warning,rotating_light" \
        -d "$1" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
}
ALL_OK=true
DOWN_LIST=""
for SERVICE in "${SERVICES[@]}"; do
    NAME="${SERVICE%%|*}"
    URL="${SERVICE##*|}"
    if ! timeout 5 curl -sf "$URL" > /dev/null 2>&1; then
        ALL_OK=false
        DOWN_LIST="$DOWN_LIST $NAME"
    fi
done
if [ "$ALL_OK" = false ]; then
    # Алёрт раз в час, не каждые 30 мин (антиспам)
    if [ ! -f "$ALERT_LOCK" ] || \
       [ $(($(date +%s) - $(stat -c %Y "$ALERT_LOCK" 2>/dev/null || echo 0))) -gt 3600 ]; then
        touch "$ALERT_LOCK"
        notify_alert "DOWN:${DOWN_LIST}. $(date '+%d.%m %H:%M'). Запусти failover.sh если нужно!"
    fi
else
    # Сервисы восстановились — уведомляем и снимаем блокировку
    if [ -f "$ALERT_LOCK" ]; then
        rm -f "$ALERT_LOCK"
        curl -s --max-time 8 \
            -H "Title: ✅ NAS Mirror RECOVERED" \
            -H "Priority: default" \
            -H "Tags: white_check_mark" \
            -d "Все сервисы NAS работают. $(date '+%d.%m %H:%M')" \
            "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
    fi
fi
