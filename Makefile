SHELL := /bin/bash
PROFILE ?=
copy-env:
	@test -f .env || cp .env.example .env
up: copy-env
	docker compose up -d --build
	docker compose ps
logs:
	docker compose logs -f --tail=200
down:
	docker compose down
clean:
	docker compose down -v
bash-backend:
	docker compose exec backend-cpu bash || true
seed:
	docker compose exec backend-cpu python scripts/seed_demo.py || true
restart:
	docker compose restart
