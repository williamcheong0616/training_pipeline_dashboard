.PHONY: setup build up down restart ps ip logs logs-api logs-worker logs-flower \
        shell-api shell-worker shell-redis db-backup clean prune

# ── First-time workstation setup ──────────────────────────────────────────────
setup:
	@mkdir -p data/db data/redis models datasets outputs exports logs backups
	@[ -f .env ] || (cp .env.example .env && echo "⚠  Created .env from example — edit it before starting")
	@echo "✓  Directory layout ready"

# ── Image management ──────────────────────────────────────────────────────────
build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

pull:
	docker compose pull redis

# ── Lifecycle ─────────────────────────────────────────────────────────────────
up: setup
	docker compose up -d
	@echo "✓  Stack is up — API: http://localhost:$$(grep API_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)"
	@echo "   Frontend : http://localhost:$$(grep FRONTEND_PORT .env 2>/dev/null | cut -d= -f2 || echo 3000)"
	@echo "   Flower   : http://localhost:$$(grep FLOWER_PORT .env 2>/dev/null | cut -d= -f2 || echo 5555)/flower"

down:
	docker compose down

restart:
	docker compose restart

ps:
	docker compose ps

# Print the LAN URL teammates should use ──────────────────────────────────────
ip:
	@LAN_IP=$$(ip route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($$i=="src") print $$(i+1); exit}' \
	  || ipconfig getifaddr en0 2>/dev/null \
	  || hostname -I 2>/dev/null | awk '{print $$1}'); \
	PORT=$$(grep -s FRONTEND_PORT .env | cut -d= -f2 || echo 3000); \
	echo "Share this URL with teammates:"; \
	echo "  http://$$LAN_IP:$${PORT:-3000}"

# ── Logs (live-follow, last 100 lines) ────────────────────────────────────────
logs:
	docker compose logs -f --tail=100

logs-api:
	docker compose logs -f --tail=100 api

logs-worker:
	docker compose logs -f --tail=100 worker

logs-flower:
	docker compose logs -f --tail=100 flower

logs-redis:
	docker compose logs -f --tail=100 redis

# ── Shells ────────────────────────────────────────────────────────────────────
shell-api:
	docker compose exec api bash

shell-worker:
	docker compose exec worker bash

shell-redis:
	docker compose exec redis redis-cli

# ── Database backup (copies the SQLite file from the bind-mount) ──────────────
db-backup:
	@mkdir -p backups
	@cp data/db/forge.db backups/forge_$$(date +%Y%m%d_%H%M%S).db
	@echo "✓  Backup saved to backups/"

db-restore:
	@[ -n "$(FILE)" ] || (echo "Usage: make db-restore FILE=backups/forge_YYYYMMDD_HHMMSS.db" && exit 1)
	docker compose stop api worker
	cp $(FILE) data/db/forge.db
	docker compose start api worker
	@echo "✓  Restored $(FILE)"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	docker compose down --remove-orphans

prune:
	docker compose down --remove-orphans
	docker system prune -f
	@echo "⚠  Images removed — run 'make build' before next 'make up'"
