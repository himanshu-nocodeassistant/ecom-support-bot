from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormalizedOrder:
    order_id: str
    customer_name: str
    status: str
    shipping_date: str | None
    delivery_estimate: str | None
    item: str
    delivered: bool


@dataclass
class KnowledgeDocument:
    id: str
    title: str
    category: str
    content: str
    doc_type: str = "guide"  # 'policy' | 'guide' | 'faq'
    date_updated: str | None = None  # ISO date string, e.g. "2024-01-15"


@dataclass
class KnowledgeChunk:
    document_id: str
    chunk_text: str
    metadata: dict[str, str]


REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({(key or "").strip(): value for key, value in row.items()})
        return rows


def validate_olist_dir(dataset_dir: str) -> dict[str, Path]:
    base = Path(dataset_dir)
    files = {key: base / filename for key, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Olist dataset files: " + ", ".join(sorted(missing))
        )
    return files


def load_olist_orders(dataset_dir: str, limit: int | None = None) -> list[NormalizedOrder]:
    files = validate_olist_dir(dataset_dir)
    orders = _read_csv(files["orders"])
    order_items = _read_csv(files["order_items"])
    products = _read_csv(files["products"])
    customers = _read_csv(files["customers"])
    translations = _read_csv(files["category_translation"])

    english_name_by_portuguese_name = {
        row["product_category_name"]: row.get("product_category_name_english", "")
        for row in translations
    }

    product_name_by_id = {
        row["product_id"]: english_name_by_portuguese_name.get(
            row.get("product_category_name", ""),
            row.get("product_category_name", "") or "unknown-product",
        )
        for row in products
    }
    customer_name_by_id = {
        row["customer_id"]: row.get("customer_unique_id", "") or "unknown-customer"
        for row in customers
    }
    item_names_by_order_id: dict[str, list[str]] = defaultdict(list)
    for row in order_items:
        product_id = row.get("product_id", "")
        item_names_by_order_id[row["order_id"]].append(
            product_name_by_id.get(product_id, "unknown-product")
        )

    normalized: list[NormalizedOrder] = []
    for row in orders[:limit] if limit else orders:
        order_id = row["order_id"]
        item_names = item_names_by_order_id.get(order_id, ["unknown-product"])
        normalized.append(
            NormalizedOrder(
                order_id=order_id,
                customer_name=customer_name_by_id.get(
                    row.get("customer_id", ""), "unknown-customer"
                ),
                status=row.get("order_status", "unknown"),
                shipping_date=row.get("order_delivered_carrier_date") or None,
                delivery_estimate=row.get("order_estimated_delivery_date") or None,
                item=", ".join(item_names[:3]),
                delivered=row.get("order_status") == "delivered",
            )
        )
    return normalized


def export_orders_csv(dataset_dir: str, output_path: str, limit: int | None = None) -> int:
    rows = load_olist_orders(dataset_dir=dataset_dir, limit=limit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order_id",
                "customer_name",
                "status",
                "shipping_date",
                "delivery_estimate",
                "item",
                "delivered",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "order_id": row.order_id,
                    "customer_name": row.customer_name,
                    "status": row.status,
                    "shipping_date": row.shipping_date or "",
                    "delivery_estimate": row.delivery_estimate or "",
                    "item": row.item,
                    "delivered": str(row.delivered).lower(),
                }
            )
    return len(rows)


def import_orders_to_postgres(
    database_url: str,
    dataset_dir: str,
    limit: int | None = None,
) -> int:
    import psycopg

    rows = load_olist_orders(dataset_dir=dataset_dir, limit=limit)
    payload: Iterable[tuple[str, str, str, str | None, str | None, str, bool]] = (
        (
            row.order_id,
            row.customer_name,
            row.status,
            row.shipping_date,
            row.delivery_estimate,
            row.item,
            row.delivered,
        )
        for row in rows
    )

    query = """
        insert into support_orders (
            order_id,
            customer_name,
            status,
            shipping_date,
            delivery_estimate,
            item,
            delivered
        )
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (order_id) do update set
            customer_name = excluded.customer_name,
            status = excluded.status,
            shipping_date = excluded.shipping_date,
            delivery_estimate = excluded.delivery_estimate,
            item = excluded.item,
            delivered = excluded.delivered
    """

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, payload)
    return len(rows)


def _derive_doc_type(stem: str) -> str:
    if "policy" in stem:
        return "policy"
    if "faq" in stem:
        return "faq"
    return "guide"


def _extract_date_updated(raw: str) -> str | None:
    """Read optional frontmatter line: `<!-- date_updated: 2024-01-15 -->`"""
    import re

    match = re.search(r"<!--\s*date_updated:\s*(\d{4}-\d{2}-\d{2})\s*-->", raw)
    return match.group(1) if match else None


def load_knowledge_documents(knowledge_dir: str) -> list[KnowledgeDocument]:
    base = Path(knowledge_dir)
    documents: list[KnowledgeDocument] = []
    for path in sorted(base.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title = lines[0].removeprefix("# ").strip() if lines else path.stem.replace("-", " ")
        content = "\n".join(line for line in lines[1:]).strip()
        category = "policy" if "policy" in path.stem else "product"
        doc_type = _derive_doc_type(path.stem)
        date_updated = _extract_date_updated(raw)
        documents.append(
            KnowledgeDocument(
                id=path.stem,
                title=title,
                category=category,
                content=content,
                doc_type=doc_type,
                date_updated=date_updated,
            )
        )
    return documents


def chunk_knowledge_documents(
    documents: list[KnowledgeDocument],
    target_chunk_size: int = 220,
    strategy: str = "fixed",
) -> list[KnowledgeChunk]:
    """Chunk documents using the given strategy.

    strategy="fixed"    — paragraph-accumulation up to target_chunk_size chars (original)
    strategy="semantic" — one sentence-boundary chunk per paragraph; never merges across
                          paragraph boundaries, preserving natural topic breaks
    """
    if strategy == "semantic":
        return _chunk_semantic(documents)
    return _chunk_fixed(documents, target_chunk_size)


# Current strategy: accumulate paragraphs until target_chunk_size chars
# Parameters: target_chunk_size=220 chars, paragraph delimiter="\n"
# Overlap: none
def _chunk_fixed(
    documents: list[KnowledgeDocument],
    target_chunk_size: int = 220,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        paragraphs = [part.strip() for part in document.content.split("\n") if part.strip()]
        current_parts: list[str] = []
        current_length = 0
        chunk_index = 0
        current_section = document.title
        for paragraph in paragraphs:
            # Track section headers (## Sub-section) to populate source_section
            if paragraph.startswith("## "):
                current_section = paragraph.removeprefix("## ").strip()
                continue
            projected = current_length + len(paragraph) + (1 if current_parts else 0)
            if current_parts and projected > target_chunk_size:
                chunk_index += 1
                chunk_text = " ".join(current_parts).strip()
                chunks.append(
                    KnowledgeChunk(
                        document_id=document.id,
                        chunk_text=chunk_text,
                        metadata={
                            "chunk_index": str(chunk_index),
                            "category": document.category,
                            "doc_type": document.doc_type,
                            "source_section": current_section,
                            "chunk_strategy": "fixed",
                        },
                    )
                )
                current_parts = [paragraph]
                current_length = len(paragraph)
            else:
                current_parts.append(paragraph)
                current_length = projected
        if current_parts:
            chunk_index += 1
            chunks.append(
                KnowledgeChunk(
                    document_id=document.id,
                    chunk_text=" ".join(current_parts).strip(),
                    metadata={
                        "chunk_index": str(chunk_index),
                        "category": document.category,
                        "doc_type": document.doc_type,
                        "source_section": current_section,
                        "chunk_strategy": "fixed",
                    },
                )
            )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ', '? ', '! ' boundaries."""
    import re

    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_semantic(documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
    """Each paragraph becomes its own chunk, preserving natural topic boundaries.

    If a paragraph contains multiple sentences, each sentence is kept together
    as a single chunk (no cross-paragraph merging).
    """
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        paragraphs = [part.strip() for part in document.content.split("\n") if part.strip()]
        chunk_index = 0
        current_section = document.title
        for paragraph in paragraphs:
            if paragraph.startswith("## "):
                current_section = paragraph.removeprefix("## ").strip()
                continue
            chunk_index += 1
            chunks.append(
                KnowledgeChunk(
                    document_id=document.id,
                    chunk_text=paragraph,
                    metadata={
                        "chunk_index": str(chunk_index),
                        "category": document.category,
                        "doc_type": document.doc_type,
                        "source_section": current_section,
                        "chunk_strategy": "semantic",
                    },
                )
            )
    return chunks


def deduplicate_chunks(
    chunks: list[KnowledgeChunk],
    embeddings: list[list[float]],
    threshold: float = 0.95,
) -> tuple[list[KnowledgeChunk], list[list[float]], int]:
    """Remove near-duplicate chunks where cosine similarity >= threshold.

    Runs at index time so duplicates never enter the database.
    Returns (kept_chunks, kept_embeddings, n_dropped).
    """
    if not chunks:
        return chunks, embeddings, 0

    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    kept_indices: list[int] = []
    for i, emb in enumerate(embeddings):
        is_dup = False
        for j in kept_indices:
            if cosine(emb, embeddings[j]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    n_dropped = len(chunks) - len(kept_indices)
    return (
        [chunks[i] for i in kept_indices],
        [embeddings[i] for i in kept_indices],
        n_dropped,
    )


def _voyage_embed_with_retry(client, texts: list[str], input_type: str) -> list[list[float]]:
    import time

    import voyageai

    for attempt in range(5):
        try:
            result = client.embed(texts, model="voyage-3-lite", input_type=input_type)
            return result.embeddings
        except voyageai.error.RateLimitError:
            wait = 22 * (attempt + 1)
            print(f"  Voyage rate limit — waiting {wait}s (attempt {attempt + 1}/5)...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Voyage rate limit persists after 5 retries")


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    import voyageai

    client = voyageai.Client(api_key=api_key)
    return _voyage_embed_with_retry(client, texts, "document")


def embed_queries(texts: list[str], api_key: str) -> list[list[float]]:
    import voyageai

    client = voyageai.Client(api_key=api_key)
    return _voyage_embed_with_retry(client, texts, "query")


def embed_query(text: str, api_key: str) -> list[float]:
    return embed_queries([text], api_key=api_key)[0]


def import_knowledge_to_postgres(
    database_url: str,
    knowledge_dir: str,
    voyage_api_key: str | None = None,
    chunk_strategy: str = "fixed",
) -> dict[str, int]:
    import json

    import psycopg

    documents = load_knowledge_documents(knowledge_dir)
    chunks = chunk_knowledge_documents(documents, strategy=chunk_strategy)

    embeddings: list[list[float]] | None = None
    n_dropped = 0
    if voyage_api_key:
        embeddings = embed_texts([c.chunk_text for c in chunks], api_key=voyage_api_key)
        chunks, embeddings, n_dropped = deduplicate_chunks(chunks, embeddings)
        if n_dropped:
            import sys

            print(f"  deduplication: dropped {n_dropped} near-duplicate chunk(s)", file=sys.stderr)

    document_query = """
        insert into knowledge_documents (id, title, category, content, doc_type, date_updated)
        values (%s, %s, %s, %s, %s, %s)
        on conflict (id) do update set
            title = excluded.title,
            category = excluded.category,
            content = excluded.content,
            doc_type = excluded.doc_type,
            date_updated = excluded.date_updated
    """
    delete_chunks_query = "delete from knowledge_chunks where document_id = %s"
    chunk_query = """
        insert into knowledge_chunks (document_id, chunk_text, metadata, embedding)
        values (%s, %s, %s::jsonb, %s::vector)
    """
    chunk_query_no_embedding = """
        insert into knowledge_chunks (document_id, chunk_text, metadata)
        values (%s, %s, %s::jsonb)
    """

    chunks_by_document: dict[str, list[KnowledgeChunk]] = defaultdict(list)
    chunk_index_map: dict[tuple[str, int], int] = {}
    for global_idx, chunk in enumerate(chunks):
        local_idx = len(chunks_by_document[chunk.document_id])
        chunk_index_map[(chunk.document_id, local_idx)] = global_idx
        chunks_by_document[chunk.document_id].append(chunk)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for document in documents:
                cur.execute(
                    document_query,
                    (
                        document.id,
                        document.title,
                        document.category,
                        document.content,
                        document.doc_type,
                        document.date_updated,
                    ),
                )
                cur.execute(delete_chunks_query, (document.id,))
                doc_chunks = chunks_by_document[document.id]
                for local_idx, chunk in enumerate(doc_chunks):
                    global_idx = chunk_index_map[(chunk.document_id, local_idx)]
                    if embeddings is not None:
                        embedding_str = "[" + ",".join(str(v) for v in embeddings[global_idx]) + "]"
                        cur.execute(
                            chunk_query,
                            (
                                chunk.document_id,
                                chunk.chunk_text,
                                json.dumps(chunk.metadata),
                                embedding_str,
                            ),
                        )
                    else:
                        cur.execute(
                            chunk_query_no_embedding,
                            (chunk.document_id, chunk.chunk_text, json.dumps(chunk.metadata)),
                        )
    return {"documents": len(documents), "chunks": len(chunks), "dropped_duplicates": n_dropped}
