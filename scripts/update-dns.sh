#!/usr/bin/env bash
# Updates Cloudflare DNS A records for braceml.com with the current public IP.
# Run via cron every 5 minutes: */5 * * * * /path/to/update-dns.sh
#
# Required env vars (set in /etc/default/cloudflare-dns or export before running):
#   CF_API_TOKEN  - Cloudflare API token with Zone.DNS edit permission
#   CF_ZONE_ID    - Zone ID for braceml.com (found on Cloudflare dashboard overview page)

set -e

# Load env from file if present
[ -f /etc/default/cloudflare-dns ] && . /etc/default/cloudflare-dns

if [ -z "$CF_API_TOKEN" ] || [ -z "$CF_ZONE_ID" ]; then
    echo "Error: CF_API_TOKEN and CF_ZONE_ID must be set"
    exit 1
fi

DOMAIN="braceml.com"
SUBDOMAINS=("braceml.com" "ws.braceml.com")

# Get current public IP
CURRENT_IP=$(curl -4 -s ifconfig.me)
if [ -z "$CURRENT_IP" ]; then
    echo "Error: Could not determine public IP"
    exit 1
fi

CF_API="https://api.cloudflare.com/client/v4"
AUTH_HEADER="Authorization: Bearer $CF_API_TOKEN"

for NAME in "${SUBDOMAINS[@]}"; do
    # Get existing record
    RECORD=$(curl -s -X GET "$CF_API/zones/$CF_ZONE_ID/dns_records?type=A&name=$NAME" \
        -H "$AUTH_HEADER" -H "Content-Type: application/json")

    RECORD_IP=$(echo "$RECORD" | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print(r[0]['content'] if r else '')" 2>/dev/null)
    RECORD_ID=$(echo "$RECORD" | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print(r[0]['id'] if r else '')" 2>/dev/null)

    if [ "$RECORD_IP" = "$CURRENT_IP" ]; then
        continue  # Already up to date
    fi

    if [ -n "$RECORD_ID" ]; then
        # Update existing record
        curl -s -X PUT "$CF_API/zones/$CF_ZONE_ID/dns_records/$RECORD_ID" \
            -H "$AUTH_HEADER" -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"$NAME\",\"content\":\"$CURRENT_IP\",\"ttl\":300,\"proxied\":false}" > /dev/null
        echo "Updated $NAME → $CURRENT_IP"
    else
        # Create new record
        curl -s -X POST "$CF_API/zones/$CF_ZONE_ID/dns_records" \
            -H "$AUTH_HEADER" -H "Content-Type: application/json" \
            --data "{\"type\":\"A\",\"name\":\"$NAME\",\"content\":\"$CURRENT_IP\",\"ttl\":300,\"proxied\":false}" > /dev/null
        echo "Created $NAME → $CURRENT_IP"
    fi
done
