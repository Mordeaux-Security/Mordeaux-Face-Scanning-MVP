"""
Debug script to analyze cosine similarities between specific Qdrant points and sample 
the global similarity distribution in faces_v1.

Usage:
    cd face-pipeline
    source venv/bin/activate  # if you have one
    python -m scripts.debug_similarity

Output Sections:

1. Payloads:
   - Shows tenant_id, identity_id, image_id, and URL for each point
   - Sanity check: Verify Person 1 and Person 4 have different identity_id values
   - Verify URLs/image_ids match expectations

2. Within-person similarities:
   - Pairwise similarities between images of the same person (Person 1 or Person 4)
   - Expected: Very high similarity (0.85-0.99+) for same-person pairs

3. Cross-person similarities:
   - Pairwise similarities between Person 1 and Person 4 images
   - KEY DIAGNOSTIC:
     * If ~0.95-0.99 → embeddings are likely broken (collapse, reused vectors, wrong preprocessing)
     * If ~0.3-0.7 → backend is fine, problem is likely interpretation/thresholds/UI

4. Global similarity stats:
   - Random sampling of pairs across the entire collection
   - Tells you about the embedding space:
     * If median is very high (e.g. 0.9) → whole space is collapsed
     * If median is moderate but max is high → space is healthy
"""

import itertools

from typing import List, Dict, Any



import numpy as np



from pipeline.indexer import get_client

from config.settings import settings





FACES_COLLECTION = settings.QDRANT_COLLECTION





# === IDs you gave me ===



PERSON_4_IDS = [

    "3d3ede4f-a45f-c8f4-2954-a4ce7043c21d",  # person4_a.jpg

    "7e95c0a9-19a9-c634-1e5f-fd91f1d8ca8c",  # person4_b.jpg

    "e544c1a5-bf29-ba86-c0a5-ee88548bb6b6",  # person4_c.jpg

]



PERSON_1_IDS = [

    "df68ada3-5917-cebd-cba8-d8b82e36f819",  # person1_A.jpg

    "761b609c-a41c-06e6-9971-28fad1c1fe88",  # person1_B.jpeg

    "30ae3dd9-651d-1e1c-c7bd-000fc903cd86",  # person1_C.jpg

]





def cos(a: np.ndarray, b: np.ndarray) -> float:

    a = a / (np.linalg.norm(a) + 1e-9)

    b = b / (np.linalg.norm(b) + 1e-9)

    return float(np.dot(a, b))





def fetch_points(ids: List[str]) -> Dict[str, Any]:

    """

    Fetch points by ID from faces_v1.

    Returns mapping id -> {"vector": np.ndarray, "payload": dict}

    """

    qc = get_client()

    pts = qc.retrieve(

        collection_name=FACES_COLLECTION,

        ids=ids,

        with_vectors=True,

        with_payload=True,

    )

    out: Dict[str, Any] = {}

    for p in pts:

        vec = np.asarray(p.vector, dtype=np.float32)

        payload = p.payload or {}

        out[str(p.id)] = {"vector": vec, "payload": payload}

    return out





def print_pairwise(group_name: str, id_list: List[str], points: Dict[str, Any]) -> None:

    print(f"\n=== {group_name} pairwise similarities ===")

    for id1, id2 in itertools.combinations(id_list, 2):

        p1 = points.get(id1)

        p2 = points.get(id2)

        if p1 is None or p2 is None:

            print(f"  Skipping pair ({id1}, {id2}) – missing in Qdrant")

            continue

        v1 = p1["vector"]

        v2 = p2["vector"]

        c = cos(v1, v2)

        l2 = float(np.linalg.norm(v1 - v2))

        print(f"  ({id1[:8]}, {id2[:8]}): cos={c:.4f}, L2={l2:.4f}")





def print_payloads(label: str, ids: List[str], points: Dict[str, Any]) -> None:

    print(f"\n=== {label} payloads ===")

    for pid in ids:

        p = points.get(pid)

        if p is None:

            print(f"  {pid[:8]}: NOT FOUND")

            continue

        payload = p["payload"]

        print(

            f"  {pid[:8]}: tenant={payload.get('tenant_id')}, "

            f"identity_id={payload.get('identity_id')}, "

            f"image_id={payload.get('image_id')}, "

            f"url={payload.get('url') or payload.get('source_url')}"

        )





def sample_random_pairs(n_points: int = 200, n_pairs: int = 100) -> None:

    """

    Sample random points + random pairs to see global similarity distribution.

    """

    qc = get_client()



    print("\n=== Sampling random points for global similarity stats ===")

    # simple scroll to get some points

    pts = []

    offset = None

    while len(pts) < n_points:

        res, offset = qc.scroll(

            collection_name=FACES_COLLECTION,

            limit=min(64, n_points - len(pts)),

            offset=offset,

            with_vectors=True,

            with_payload=False,

        )

        pts.extend(res)

        if offset is None:

            break



    if len(pts) < 2:

        print("  Not enough points in collection to sample.")

        return



    vectors = [np.asarray(p.vector, dtype=np.float32) for p in pts]

    idx_pairs = list(

        set(

            tuple(sorted(pair))

            for pair in (

                (np.random.randint(0, len(vectors)), np.random.randint(0, len(vectors)))

                for _ in range(n_pairs * 3)

            )

        )

    )

    idx_pairs = [p for p in idx_pairs if p[0] != p[1]][:n_pairs]



    sims = []

    for i, j in idx_pairs:

        sims.append(cos(vectors[i], vectors[j]))



    sims = np.array(sims, dtype=np.float32)

    print(f"  sampled_pairs={len(sims)}")

    print(f"  min={sims.min():.4f}, median={np.median(sims):.4f}, "

          f"p90={np.percentile(sims, 90):.4f}, max={sims.max():.4f}")





def main():

    all_ids = PERSON_1_IDS + PERSON_4_IDS

    points = fetch_points(all_ids)



    # Show basic payload info (sanity check: are these really different people?)

    print_payloads("Person 1", PERSON_1_IDS, points)

    print_payloads("Person 4", PERSON_4_IDS, points)



    # Intra-person similarities (should be high)

    print_pairwise("Person 1 (within identity)", PERSON_1_IDS, points)

    print_pairwise("Person 4 (within identity)", PERSON_4_IDS, points)



    # Cross-person similarities (should be clearly lower)

    print("\n=== Cross-person similarities (Person 1 vs Person 4) ===")

    for id1 in PERSON_1_IDS:

        for id2 in PERSON_4_IDS:

            p1 = points.get(id1)

            p2 = points.get(id2)

            if p1 is None or p2 is None:

                continue

            v1 = p1["vector"]

            v2 = p2["vector"]

            c = cos(v1, v2)

            l2 = float(np.linalg.norm(v1 - v2))

            print(f"  ({id1[:8]}, {id2[:8]}): cos={c:.4f}, L2={l2:.4f}")



    # Global sanity check on embedding space

    sample_random_pairs()





if __name__ == "__main__":

    main()
