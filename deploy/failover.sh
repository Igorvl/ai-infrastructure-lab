#!/bin/bash
# ============================================================
# Project DNA — One-Click Failover Script
# Run on NAS when Ubuntu primary is down.
# WARM STANDBY: Start mirror stack manually, NO auto-failover.
# ============================================================

set -e

COMPOSE_FILE="/volume1/docker/dna-mirror/docker-compose.mirror.yml"
PRIMARY_URL="http://172.25.9.33:8000/v1/dna/health"
MIRROR_URL="http://localhost:8000/v1/dna/health"

echo "======================================"
echo " Project DNA — FAILOVER TO NAS MIRROR"
echo "======================================"
echo ""

# 1. Check if primary is actually down
echo "[1/4] Checking primary (Ubuntu) status..."
if curl -s --connect-timeout 5 "$PRIMARY_URL" > /dev/null 2>&1; then
    echo "  ⚠️  WARNING: Primary is still RESPONDING at $PRIMARY_URL"
    echo "  Are you sure you want to activate the mirror?"
    read -p "  Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "  Aborted."
        exit 0
    fi
else
    echo "  ✅ Primary is DOWN. Proceeding with failover."
fi

echo ""
echo "[2/4] Pulling latest images from registry..."
docker compose -f "$COMPOSE_FILE" pull 2>/dev/null || true

echo ""
echo "[3/4] Starting mirror stack..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "[4/4] Waiting for mirror health check..."
for i in $(seq 1 30); do
    if curl -s --connect-timeout 3 "$MIRROR_URL" > /dev/null 2>&1; then
        echo "  ✅ Mirror is UP and healthy!"
        echo ""
        echo "======================================"
        echo " ✅ FAILOVER COMPLETE"
        echo " Mirror URL: http://172.25.9.147:8000"
        echo " Update Open WebUI to point to NAS IP"
        echo "======================================"
        exit 0
    fi
    echo "  Waiting... ($i/30)"
    sleep 3
done

echo "  ❌ Mirror did not respond in 90s. Check logs:"
echo "     docker compose -f $COMPOSE_FILE logs --tail=50"
exit 1

