.PHONY: setup build up down restart ps ip logs logs-api logs-worker logs-flower \
        shell-api shell-worker shell-redis db-backup install-flash-attn clean prune

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

# Install flash-attn — builds wheel in a one-shot container (auto-detects nvcc on host)
# then copies and installs the pre-built wheel into the running worker.
# Re-run after `make up` if the worker container is recreated.
install-flash-attn:
	@NVCC=$$(command -v nvcc 2>/dev/null \
	  || find /usr/local/cuda*/bin /usr/local/cuda/bin /usr/bin -name nvcc 2>/dev/null \
	     | sort -rV | head -1); \
	[ -n "$$NVCC" ] || { \
	  echo "❌  nvcc not found on host."; \
	  echo "    Install the CUDA toolkit devel package, e.g.:"; \
	  echo "      sudo apt install cuda-toolkit-13-0"; \
	  exit 1; \
	}; \
	CUDA="$$(realpath "$$(dirname "$$(dirname "$$NVCC")")")"; \
	echo "Found nvcc : $$NVCC"; \
	echo "CUDA_HOME  : $$CUDA"; \
	IMG=$$(docker inspect "$$(docker compose ps -q worker)" --format='{{.Config.Image}}' 2>/dev/null); \
	[ -n "$$IMG" ] || { echo "❌  worker container not running — run: make up"; exit 1; }; \
	WHEELDIR=$$(mktemp -d); \
	echo "Building flash-attn wheel (this takes ~10 min)..."; \
	docker run --rm --runtime=nvidia \
	  -u root \
	  -e CUDA_HOME="$$CUDA" \
	  -e PATH="$$CUDA/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
	  -v "$$CUDA:/usr/local/cuda:ro" \
	  -v "$$WHEELDIR:/wheels" \
	  "$$IMG" \
	  sh -c "pip install packaging setuptools wheel && \
	         MAX_JOBS=4 pip wheel flash-attn --no-build-isolation -w /wheels"; \
	WHEEL=$$(ls "$$WHEELDIR"/flash_attn-*.whl 2>/dev/null | head -1); \
	[ -n "$$WHEEL" ] || { echo "❌  wheel build failed"; rm -rf "$$WHEELDIR"; exit 1; }; \
	echo "Installing wheel into running worker..."; \
	WORKER_ID=$$(docker compose ps -q worker); \
	docker cp "$$WHEEL" "$$WORKER_ID:/tmp/flash_attn.whl"; \
	docker compose exec -u root worker pip install /tmp/flash_attn.whl; \
	rm -rf "$$WHEELDIR"; \
	echo "✓  flash-attn installed"

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
