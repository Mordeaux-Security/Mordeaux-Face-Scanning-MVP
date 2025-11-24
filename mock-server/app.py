"""
Mock Server for Face Search API - Phase 3
==========================================

Simple FastAPI mock server that returns exact Phase 2 API contracts.
Decouples frontend development from backend readiness.

Features:
- Multiple fixture sets (tiny/medium/large)
- Realistic score distributions
- Error scenarios (404, expired URLs)
- Presigned URL simulation
- Configurable response delays

Usage:
    python app.py
    # Server runs on http://localhost:8000
"""

from fastapi import FastAPI, HTTPException, Header, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import uvicorn
from datetime import datetime, timedelta
import random
import time

# Import fixtures
from fixtures import (
    get_fixture_by_name,
    generate_mock_face_id,
    FIXTURE_SETS,
    ERROR_SCENARIOS
)

app = FastAPI(
    title="Face Search API Mock Server",
    description="Phase 3 Mock Server with configurable fixtures and error scenarios",
    version="0.1.0-mock"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "default_fixture": "medium",  # tiny, medium, large
    "simulate_latency": True,
    "min_latency_ms": 50,
    "max_latency_ms": 300,
    "error_rate": 0.0,  # 0-1, probability of random errors
}


def simulate_network_latency():
    """Simulate realistic network latency"""
    if CONFIG["simulate_latency"]:
        delay = random.uniform(
            CONFIG["min_latency_ms"] / 1000,
            CONFIG["max_latency_ms"] / 1000
        )
        time.sleep(delay)


def maybe_inject_error():
    """Randomly inject errors based on error_rate"""
    if random.random() < CONFIG["error_rate"]:
        error_type = random.choice(["500", "429", "timeout"])
        if error_type == "500":
            raise HTTPException(500, detail="Mock server error injected")
        elif error_type == "429":
            raise HTTPException(429, detail="Rate limit exceeded (mock)")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face-pipeline-search-api-mock",
        "version": "0.1.0-mock",
        "api_version": "v0.1",
        "note": "Mock server - returning fixture data",
        "available_fixtures": list(FIXTURE_SETS.keys()),
        "config": CONFIG
    }


@app.post("/api/v1/search")
async def search_faces(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    image: Optional[UploadFile] = File(None),
    tenant_id: Optional[str] = Form(None),
    top_k: int = Form(10),
    threshold: float = Form(0.75),
    fixture: Optional[str] = Query(None, description="Override fixture set: tiny, medium, large"),
    error_scenario: Optional[str] = Query(None, description="Simulate error: no_results, api_error, timeout")
):
    """
    Mock search endpoint - returns fixture data
    
    Query params:
    - fixture: Override default fixture (tiny/medium/large)
    - error_scenario: Simulate error (no_results/api_error/timeout)
    """
    # Simulate latency
    simulate_network_latency()
    
    # Maybe inject random error
    maybe_inject_error()
    
    # Validate tenant
    tenant = x_tenant_id or tenant_id
    if not tenant:
        raise HTTPException(400, detail="X-Tenant-ID header or tenant_id required")
    
    # Handle error scenarios
    if error_scenario:
        if error_scenario == "no_results":
            return {
                "query": {
                    "tenant_id": tenant,
                    "search_mode": "image",
                    "top_k": top_k,
                    "threshold": threshold
                },
                "hits": [],
                "count": 0
            }
        elif error_scenario == "api_error":
            raise HTTPException(500, detail="Internal server error (simulated)")
        elif error_scenario == "timeout":
            time.sleep(5)
            raise HTTPException(504, detail="Gateway timeout (simulated)")
        elif error_scenario in ERROR_SCENARIOS:
            scenario = ERROR_SCENARIOS[error_scenario]
            raise HTTPException(scenario["status_code"], detail=scenario["message"])
    
    # Select fixture set
    fixture_name = fixture or CONFIG["default_fixture"]
    if fixture_name not in FIXTURE_SETS:
        raise HTTPException(400, detail=f"Invalid fixture: {fixture_name}. Available: {list(FIXTURE_SETS.keys())}")
    
    fixture_data = get_fixture_by_name(fixture_name)
    
    # Apply top_k and threshold filters
    filtered_hits = [
        hit for hit in fixture_data["hits"]
        if hit["score"] >= threshold
    ][:top_k]
    
    # Build response
    response = {
        "query": {
            "tenant_id": tenant,
            "search_mode": "image",
            "top_k": top_k,
            "threshold": threshold
        },
        "hits": filtered_hits,
        "count": len(filtered_hits)
    }
    
    return response


@app.get("/api/v1/faces/{face_id}")
async def get_face_by_id(
    face_id: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    fixture: Optional[str] = Query("medium", description="Fixture set to search in")
):
    """
    Get face details by ID
    """
    simulate_network_latency()
    maybe_inject_error()
    
    if not x_tenant_id:
        raise HTTPException(400, detail="X-Tenant-ID header required")
    
    # Search across fixture
    fixture_data = get_fixture_by_name(fixture)
    
    for hit in fixture_data["hits"]:
        if hit["face_id"] == face_id:
            return {
                "face_id": hit["face_id"],
                "payload": hit["payload"],
                "thumb_url": hit["thumb_url"]
            }
    
    raise HTTPException(404, detail=f"Face not found: {face_id}")


@app.get("/api/v1/stats")
async def get_stats(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Mock statistics endpoint
    """
    simulate_network_latency()
    
    if not x_tenant_id:
        raise HTTPException(400, detail="X-Tenant-ID header required")
    
    return {
        "processed": 15234,
        "rejected": 432,
        "dup_skipped": 187
    }


@app.get("/mock/fixtures")
async def list_fixtures():
    """
    List all available fixtures (mock-specific endpoint)
    """
    return {
        "fixtures": {
            name: {
                "count": len(FIXTURE_SETS[name]["hits"]),
                "score_range": f"{min(h['score'] for h in FIXTURE_SETS[name]['hits']):.2f} - {max(h['score'] for h in FIXTURE_SETS[name]['hits']):.2f}",
                "sites": list(set(h["payload"]["site"] for h in FIXTURE_SETS[name]["hits"]))
            }
            for name in FIXTURE_SETS
        },
        "error_scenarios": list(ERROR_SCENARIOS.keys()),
        "config": CONFIG
    }


@app.post("/mock/config")
async def update_config(
    default_fixture: Optional[str] = None,
    simulate_latency: Optional[bool] = None,
    min_latency_ms: Optional[int] = None,
    max_latency_ms: Optional[int] = None,
    error_rate: Optional[float] = None
):
    """
    Update mock server configuration (mock-specific endpoint)
    """
    if default_fixture is not None:
        if default_fixture not in FIXTURE_SETS:
            raise HTTPException(400, detail=f"Invalid fixture: {default_fixture}")
        CONFIG["default_fixture"] = default_fixture
    
    if simulate_latency is not None:
        CONFIG["simulate_latency"] = simulate_latency
    
    if min_latency_ms is not None:
        CONFIG["min_latency_ms"] = min_latency_ms
    
    if max_latency_ms is not None:
        CONFIG["max_latency_ms"] = max_latency_ms
    
    if error_rate is not None:
        if not 0 <= error_rate <= 1:
            raise HTTPException(400, detail="error_rate must be between 0 and 1")
        CONFIG["error_rate"] = error_rate
    
    return {
        "message": "Configuration updated",
        "config": CONFIG
    }


if __name__ == "__main__":
    print("=" * 80)
    print("🎭 Mock Server - Phase 3")
    print("=" * 80)
    print(f"Available fixtures: {list(FIXTURE_SETS.keys())}")
    print(f"Default fixture: {CONFIG['default_fixture']}")
    print(f"Latency simulation: {CONFIG['simulate_latency']}")
    print(f"Error rate: {CONFIG['error_rate']}")
    print("=" * 80)
    print("Starting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("Mock config: http://localhost:8000/mock/fixtures")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

