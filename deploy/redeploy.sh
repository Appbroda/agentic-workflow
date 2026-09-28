#!/usr/bin/env bash
# Run on the EC2 instance (locally, or via `aws ssm send-command`) to roll out the image that was
# just pushed to ECR. Idempotent: safe to run again if a step fails partway through.
set -euo pipefail

cd /opt/agentic-workflow

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

$COMPOSE pull migrate api
$COMPOSE run --rm migrate
$COMPOSE up -d --remove-orphans

echo "redeploy complete: $($COMPOSE images api --format '{{.Repository}}:{{.Tag}}' 2>/dev/null || echo unknown)"
