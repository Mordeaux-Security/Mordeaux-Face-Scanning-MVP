SHELL := /bin/bash

up:
	 docker compose up -d --build

down:
	 docker compose down -v

logs:
	 docker compose logs -f --tail=200

restart:
	 docker compose restart

ensure:
	 docker compose exec face-pipeline python -c "from pipeline.ensure import ensure_all; ensure_all(); print('ensure complete')"

health:
	 curl -s http://localhost/api/v1/health || true; echo
	 curl -s http://localhost/api/healthz || true; echo
	 curl -s http://localhost/pipeline/api/v1/health || true; echo

build-frontend:
	 cd frontend && npm ci && npm run build

seed-qdrant:
	python -c "from qdrant_client import QdrantClient; from qdrant_client.http.models import PointStruct; import os,random; qc=QdrantClient(url=os.getenv('QDRANT_URL','http://localhost:6333')); pts=[PointStruct(id=i, vector=[random.random() for _ in range(512)], payload={'tenant_id':'demo','p_hash_prefix':'0000'}) for i in range(1000)]; qc.upsert(os.getenv('QDRANT_COLLECTION','faces_v1'), points=pts, wait=True); print('seeded 1k points')"

worker:
	docker compose exec -e ENABLE_QUEUE_WORKER=true face-pipeline python worker.py

worker-once:
	docker compose exec -e ENABLE_QUEUE_WORKER=true face-pipeline python worker.py --once

publish-test:
	docker compose exec face-pipeline python publish_test_message.py \
		--tenant demo \
		--bucket raw-images \
		--key samples/person4.jpg \
		--url file:///app/samples/person4.jpg \
		--file /app/samples/person4.jpg