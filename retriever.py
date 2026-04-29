from data_loaders.pdf_ingestion import query_db
from pathlib import Path


def retrieve(DB_DIR):
    TEST_QUERY = "YOLO"
    # DB_DIR = Path("./chroma_db")
    COLLECTION = "literature_db"

    print(f"\n── Test query: '{TEST_QUERY}' ──")

    hits = query_db(TEST_QUERY, db_dir=DB_DIR, collection=COLLECTION)

    for i, h in enumerate(hits, 1):
        print(
            f"\n[{i}] score={h['score']} | "
            f"{h['metadata'].get('title','?')} "
            f"({h['metadata'].get('year','')})"
        )
        print(h["text"][:300])
