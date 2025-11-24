"""
Fixture Generator for Mock Server - Phase 3
===========================================

Generates realistic face search result fixtures with:
- Multiple dataset sizes (tiny/medium/large)
- Realistic score distributions
- Variety in bounding boxes
- Multiple sites
- Some broken/expired URLs for error testing

Score Distributions:
- Tiny (10): High scores (0.85-0.95) - testing best matches
- Medium (200): Normal distribution (0.40-0.95) - realistic workload
- Large (2000): Wide distribution (0.30-0.98) - stress testing
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict
import hashlib

# Mock site configurations
MOCK_SITES = [
    "example.com",
    "demo-site.org",
    "test-faces.net",
    "face-archive.io",
    "photo-gallery.com",
    "sample-images.org"
]

# Placeholder image services
PLACEHOLDER_SERVICES = [
    "https://via.placeholder.com/256",
    "https://i.pravatar.cc/256",
    "https://randomuser.me/api/portraits/thumb/men/{}.jpg",
    "https://randomuser.me/api/portraits/thumb/women/{}.jpg",
]

# Error scenarios
ERROR_SCENARIOS = {
    "broken_url": {
        "status_code": 404,
        "message": "Thumbnail URL not found (simulated broken link)"
    },
    "expired_url": {
        "status_code": 403,
        "message": "Presigned URL expired (simulated)"
    },
    "rate_limit": {
        "status_code": 429,
        "message": "Rate limit exceeded"
    },
    "server_error": {
        "status_code": 500,
        "message": "Internal server error"
    }
}


def generate_mock_face_id() -> str:
    """Generate a realistic UUID-like face ID"""
    return f"face-{hashlib.md5(str(random.random()).encode()).hexdigest()[:16]}"


def generate_p_hash() -> str:
    """Generate a realistic perceptual hash"""
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:16]


def generate_bbox() -> List[int]:
    """
    Generate realistic face bounding box [x, y, width, height]
    
    Variety:
    - Different positions (centered, off-center, edge cases)
    - Different sizes (small, medium, large faces)
    - Maintains realistic proportions
    """
    # Image assumed to be 1024x1024 (common size)
    image_w, image_h = 1024, 1024
    
    # Face size varies (80-400px typical for face detection)
    face_size = random.randint(80, 400)
    aspect_ratio = random.uniform(0.8, 1.2)  # Faces are roughly square
    
    width = face_size
    height = int(face_size * aspect_ratio)
    
    # Position with margins
    x = random.randint(0, image_w - width)
    y = random.randint(0, image_h - height)
    
    return [x, y, width, height]


def generate_quality_score() -> float:
    """
    Generate realistic quality score (0.0-1.0)
    Most faces have decent quality (0.7-1.0), some are poor
    """
    if random.random() < 0.8:
        # 80% are good quality
        return round(random.uniform(0.7, 1.0), 3)
    else:
        # 20% are poor quality
        return round(random.uniform(0.3, 0.7), 3)


def generate_thumbnail_url(face_id: str, broken: bool = False) -> str:
    """
    Generate mock presigned URL or placeholder
    
    Args:
        face_id: Face identifier
        broken: If True, generate an intentionally broken URL
    """
    if broken:
        # Intentionally broken URLs for error testing
        broken_types = [
            f"https://broken-cdn.example.com/404/{face_id}.jpg",
            f"https://expired.minio.local/thumbnails/{face_id}_thumb.jpg?expired=true",
            "https://invalid-domain-12345.local/image.jpg"
        ]
        return random.choice(broken_types)
    
    # Mock presigned URL (MinIO style)
    tenant = "demo-tenant"
    timestamp = int(datetime.now().timestamp())
    expires = timestamp + 600  # 10 minutes
    
    # Simulated signature
    signature = hashlib.md5(f"{face_id}{timestamp}".encode()).hexdigest()[:32]
    
    return (
        f"https://minio.example.com/thumbnails/{tenant}/{face_id}_thumb.jpg?"
        f"X-Amz-Algorithm=AWS4-HMAC-SHA256&"
        f"X-Amz-Credential=minioadmin&"
        f"X-Amz-Date={timestamp}&"
        f"X-Amz-Expires={expires}&"
        f"X-Amz-SignedHeaders=host&"
        f"X-Amz-Signature={signature}"
    )


def generate_timestamp(index: int, total: int) -> str:
    """
    Generate realistic timestamps
    Spreads results over past 30 days with some clustering
    """
    now = datetime.now()
    # More recent results have higher scores (realistic pattern)
    days_ago = int((index / total) * 30)  # 0-30 days ago
    hours_offset = random.randint(0, 23)
    minutes_offset = random.randint(0, 59)
    
    timestamp = now - timedelta(days=days_ago, hours=hours_offset, minutes=minutes_offset)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_hit(score: float, index: int, total: int, broken_url: bool = False) -> Dict:
    """
    Generate a single search hit with realistic data
    
    Args:
        score: Similarity score (0.0-1.0)
        index: Hit index in result set
        total: Total hits in set
        broken_url: If True, generate broken thumbnail URL
    """
    face_id = generate_mock_face_id()
    site = random.choice(MOCK_SITES)
    
    # Generate source URL
    source_url = f"https://{site}/images/photo-{random.randint(1000, 9999)}.jpg"
    
    hit = {
        "face_id": face_id,
        "score": round(score, 4),
        "payload": {
            "site": site,
            "url": source_url,
            "ts": generate_timestamp(index, total),
            "bbox": generate_bbox(),
            "p_hash": generate_p_hash(),
            "quality": generate_quality_score()
        },
        "thumb_url": generate_thumbnail_url(face_id, broken=broken_url)
    }
    
    return hit


def generate_fixture_tiny() -> Dict:
    """
    Tiny fixture: 10 results
    High scores only (0.85-0.95)
    Perfect for testing "best matches" scenarios
    """
    hits = []
    for i in range(10):
        score = 0.95 - (i * 0.01)  # Decreasing: 0.95, 0.94, 0.93, ...
        hits.append(generate_hit(score, i, 10))
    
    return {
        "name": "tiny",
        "description": "10 high-quality matches (scores 0.85-0.95)",
        "hits": hits
    }


def generate_fixture_medium() -> Dict:
    """
    Medium fixture: 200 results
    Normal distribution (0.40-0.95)
    Realistic workload for development
    Includes 5 broken URLs for error testing
    """
    hits = []
    
    # Generate score distribution (bell curve centered at 0.70)
    scores = []
    for i in range(200):
        # Use beta distribution for realistic score spread
        # More results in middle range, fewer at extremes
        if i < 20:
            # Top 20: Very high scores (0.85-0.95)
            score = 0.95 - (i * 0.005)
        elif i < 80:
            # Next 60: High scores (0.70-0.85)
            score = random.uniform(0.70, 0.85)
        elif i < 160:
            # Next 80: Medium scores (0.55-0.70)
            score = random.uniform(0.55, 0.70)
        else:
            # Last 40: Lower scores (0.40-0.55)
            score = random.uniform(0.40, 0.55)
        
        scores.append(score)
    
    # Sort descending
    scores.sort(reverse=True)
    
    # Generate hits
    broken_indices = random.sample(range(200), 5)  # 5 broken URLs
    for i, score in enumerate(scores):
        broken = i in broken_indices
        hits.append(generate_hit(score, i, 200, broken_url=broken))
    
    return {
        "name": "medium",
        "description": "200 results with normal distribution (0.40-0.95), includes 5 broken URLs",
        "hits": hits
    }


def generate_fixture_large() -> Dict:
    """
    Large fixture: 2000 results
    Wide distribution (0.30-0.98)
    Stress testing pagination, filtering, performance
    Includes 20 broken URLs
    """
    hits = []
    
    # Generate realistic score distribution
    scores = []
    for i in range(2000):
        if i < 50:
            # Top 50: Excellent matches (0.90-0.98)
            score = 0.98 - (i * 0.001)
        elif i < 300:
            # Next 250: Very good matches (0.75-0.90)
            score = random.uniform(0.75, 0.90)
        elif i < 1000:
            # Next 700: Good matches (0.60-0.75)
            score = random.uniform(0.60, 0.75)
        elif i < 1600:
            # Next 600: Fair matches (0.45-0.60)
            score = random.uniform(0.45, 0.60)
        else:
            # Last 400: Poor matches (0.30-0.45)
            score = random.uniform(0.30, 0.45)
        
        scores.append(score)
    
    # Sort descending
    scores.sort(reverse=True)
    
    # Generate hits
    broken_indices = random.sample(range(2000), 20)  # 20 broken URLs
    for i, score in enumerate(scores):
        broken = i in broken_indices
        hits.append(generate_hit(score, i, 2000, broken_url=broken))
    
    return {
        "name": "large",
        "description": "2000 results with wide distribution (0.30-0.98), includes 20 broken URLs",
        "hits": hits
    }


def generate_fixture_edge_cases() -> Dict:
    """
    Edge cases fixture: Various edge cases for testing
    - Exact match (score 1.0)
    - Very low scores (< 0.3)
    - Missing fields (partial data)
    - Extreme bbox positions
    - Very old timestamps
    """
    hits = []
    
    # Perfect match
    hits.append(generate_hit(1.0, 0, 15))
    
    # Very high scores
    for i in range(3):
        hits.append(generate_hit(0.99 - i * 0.01, i + 1, 15))
    
    # Threshold edge cases (around common threshold of 0.75)
    for score in [0.76, 0.75, 0.74]:
        hits.append(generate_hit(score, len(hits), 15))
    
    # Low scores
    for score in [0.25, 0.15, 0.05]:
        hits.append(generate_hit(score, len(hits), 15))
    
    # Edge case: Minimum score
    hits.append(generate_hit(0.01, len(hits), 15))
    
    # Add more to reach 15
    while len(hits) < 15:
        score = random.uniform(0.3, 0.7)
        hits.append(generate_hit(score, len(hits), 15))
    
    return {
        "name": "edge_cases",
        "description": "15 edge case results including perfect match, threshold boundaries, very low scores",
        "hits": hits
    }


def generate_fixture_errors() -> Dict:
    """
    Error fixture: All results have broken URLs
    For testing error handling
    """
    hits = []
    for i in range(20):
        score = 0.90 - (i * 0.02)
        hits.append(generate_hit(score, i, 20, broken_url=True))
    
    return {
        "name": "errors",
        "description": "20 results with all broken thumbnail URLs (for error testing)",
        "hits": hits
    }


# Generate all fixtures on module load
FIXTURE_SETS = {
    "tiny": generate_fixture_tiny(),
    "medium": generate_fixture_medium(),
    "large": generate_fixture_large(),
    "edge_cases": generate_fixture_edge_cases(),
    "errors": generate_fixture_errors()
}


def get_fixture_by_name(name: str) -> Dict:
    """Get fixture by name, with fallback to medium"""
    if name not in FIXTURE_SETS:
        print(f"Warning: Fixture '{name}' not found, using 'medium'")
        return FIXTURE_SETS["medium"]
    return FIXTURE_SETS[name]


def print_fixture_summary():
    """Print summary of all fixtures"""
    print("\n" + "=" * 80)
    print("FIXTURE SUMMARY")
    print("=" * 80)
    
    for name, fixture in FIXTURE_SETS.items():
        hits = fixture["hits"]
        scores = [h["score"] for h in hits]
        sites = set(h["payload"]["site"] for h in hits)
        broken_count = sum(1 for h in hits if "broken" in h["thumb_url"] or "expired" in h["thumb_url"] or "invalid" in h["thumb_url"])
        
        print(f"\n{name.upper()} ({len(hits)} results):")
        print(f"  Description: {fixture['description']}")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"  Mean score: {sum(scores)/len(scores):.4f}")
        print(f"  Sites: {len(sites)} ({', '.join(sorted(sites))})")
        print(f"  Broken URLs: {broken_count}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test fixture generation
    print_fixture_summary()
    
    # Print example hit
    print("\nEXAMPLE HIT (from medium fixture):")
    print("=" * 80)
    import json
    example_hit = FIXTURE_SETS["medium"]["hits"][0]
    print(json.dumps(example_hit, indent=2))
    print("=" * 80)

