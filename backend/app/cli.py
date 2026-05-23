from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import (
    export_orders_csv,
    import_knowledge_to_postgres,
    import_orders_to_postgres,
    load_olist_orders,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="supportbot-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize-olist")
    summarize.add_argument("--dataset-dir", required=True)
    summarize.add_argument("--limit", type=int, default=10)

    export = subparsers.add_parser("export-orders")
    export.add_argument("--dataset-dir", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--limit", type=int)

    import_orders = subparsers.add_parser("import-orders")
    import_orders.add_argument("--dataset-dir", required=True)
    import_orders.add_argument("--database-url")
    import_orders.add_argument("--limit", type=int)

    compare = subparsers.add_parser(
        "compare-retrieval", help="Show FTS vs hybrid results side-by-side"
    )
    compare.add_argument("--database-url")
    compare.add_argument("--voyage-api-key")

    import_knowledge = subparsers.add_parser("import-knowledge")
    import_knowledge.add_argument("--knowledge-dir", default="./backend/knowledge")
    import_knowledge.add_argument("--database-url")
    import_knowledge.add_argument("--voyage-api-key")
    import_knowledge.add_argument(
        "--chunk-strategy", default="semantic", choices=["fixed", "semantic"]
    )

    args = parser.parse_args()

    if args.command == "summarize-olist":
        rows = load_olist_orders(dataset_dir=args.dataset_dir, limit=args.limit)
        print(f"loaded_orders={len(rows)}")
        if rows:
            print(f"first_order_id={rows[0].order_id}")
            print(f"first_item={rows[0].item}")
            print(f"first_status={rows[0].status}")
        return

    if args.command == "export-orders":
        count = export_orders_csv(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            limit=args.limit,
        )
        print(f"exported_orders={count}")
        return

    if args.command == "import-orders":
        database_url = args.database_url or _load_database_url_from_env()
        count = import_orders_to_postgres(
            database_url=database_url,
            dataset_dir=args.dataset_dir,
            limit=args.limit,
        )
        print(f"imported_orders={count}")
        return

    if args.command == "compare-retrieval":
        database_url = args.database_url or _load_database_url_from_env()
        voyage_api_key = args.voyage_api_key or _load_env_value("VOYAGE_API_KEY")
        _run_retrieval_comparison(database_url=database_url, voyage_api_key=voyage_api_key)
        return

    if args.command == "import-knowledge":
        database_url = args.database_url or _load_database_url_from_env()
        voyage_api_key = args.voyage_api_key or _load_env_value("VOYAGE_API_KEY")
        result = import_knowledge_to_postgres(
            database_url=database_url,
            knowledge_dir=args.knowledge_dir,
            voyage_api_key=voyage_api_key,
            chunk_strategy=args.chunk_strategy,
        )
        print(f"imported_documents={result['documents']}")
        print(f"imported_chunks={result['chunks']}")
        print(f"dropped_duplicates={result.get('dropped_duplicates', 0)}")
        if voyage_api_key:
            print("embeddings=generated")


def _load_env_value(key: str) -> str | None:
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, value = line.split("=", 1)
        if k == key and value:
            return value
    return None


def _load_database_url_from_env() -> str:
    value = _load_env_value("DATABASE_URL")
    if value:
        return value
    raise ValueError(".env not found or DATABASE_URL missing")


def _run_retrieval_comparison(database_url: str, voyage_api_key: str | None) -> None:
    import psycopg

    from .data_loader import embed_queries

    test_queries = [
        ("Can I get my money back?", "Refund policy"),
        ("My purchase arrived damaged, I want compensation", "Refund policy"),
        ("How long until my package shows up?", "Shipping policy"),
        ("Does the portable blender have a safety lock?", "Portable blender guide"),
        ("How do I configure a Kubernetes ingress?", None),
    ]

    fts_sql = """
        select kd.title, ts_rank(kc.search_vector, plainto_tsquery('english', %s)) as score
        from knowledge_chunks kc
        join knowledge_documents kd on kd.id = kc.document_id
        where kc.search_vector @@ plainto_tsquery('english', %s)
        order by score desc limit 1
    """
    hybrid_sql = """
        select kd.title,
            (0.3 * ts_rank(kc.search_vector, plainto_tsquery('english', %s))
             + 0.7 * (1 - (kc.embedding <=> %s::vector))) as score
        from knowledge_chunks kc
        join knowledge_documents kd on kd.id = kc.document_id
        where kc.embedding is not null
        order by score desc limit 1
    """

    THRESHOLD = 0.25
    header = f"{'Query':<48}  {'FTS result':<26}  {'Hybrid result':<26}  Correct?"
    print(header)
    print("-" * len(header))

    # Batch-embed all queries in one API call to respect free-tier rate limits
    query_texts = [q for q, _ in test_queries]
    embeddings_map: dict[str, list[float]] = {}
    if voyage_api_key:
        embeddings = embed_queries(query_texts, api_key=voyage_api_key)
        embeddings_map = dict(zip(query_texts, embeddings))

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for query, expected in test_queries:
                cur.execute(fts_sql, (query, query))
                fts_row = cur.fetchone()
                fts_label = f"{fts_row[0][:20]} ({fts_row[1]:.3f})" if fts_row else "no result"

                if voyage_api_key and query in embeddings_map:
                    emb = embeddings_map[query]
                    emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                    cur.execute(hybrid_sql, (query, emb_str))
                    hybrid_row = cur.fetchone()
                    if hybrid_row and hybrid_row[1] >= THRESHOLD:
                        hybrid_label = f"{hybrid_row[0][:20]} ({hybrid_row[1]:.3f})"
                        hybrid_title = hybrid_row[0]
                    else:
                        score_str = f"{hybrid_row[1]:.3f}" if hybrid_row else "n/a"
                        hybrid_label = f"escalate ({score_str})"
                        hybrid_title = None
                else:
                    hybrid_label = "no API key"
                    hybrid_title = None

                if expected is None:
                    correct = "✓" if hybrid_title is None else "✗ (should escalate)"
                else:
                    correct = (
                        "✓" if (hybrid_title and expected.lower() in hybrid_title.lower()) else "✗"
                    )

                q_display = query[:46] + ".." if len(query) > 46 else query
                print(f"{q_display:<48}  {fts_label:<26}  {hybrid_label:<26}  {correct}")


if __name__ == "__main__":
    main()
