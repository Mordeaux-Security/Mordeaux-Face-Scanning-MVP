# Diabetes Crawler (Isolated Repo)

This repository has been reduced to a single deliverable: the multi-process crawler that powers the diabetes data collection pipeline. All legacy backend/frontend/service code has been removed so you can focus exclusively on the crawler stack located under `diabetes-crawler/`.

## Contents

```
.
├── diabetes-crawler/        # Standalone crawler project
├── README.md                # You are here
└── pyproject.toml           # Shared tooling config (ruff, etc.)
```

See `diabetes-crawler/README.md` for architecture details, environment configuration, and run instructions. In short:

```bash
cd diabetes-crawler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Redis/MinIO/GPU worker settings
export PYTHONPATH=src
python -m diabetes_crawler.main --sites-file sites.txt
```

That folder also contains the test suite (`python -m diabetes_crawler.test_suite`), docs, and all supporting modules (config, Redis manager, GPU interface, storage logic, etc.).
