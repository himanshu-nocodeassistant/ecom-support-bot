create extension if not exists vector;

create table if not exists support_orders (
    order_id text primary key,
    customer_name text not null,
    status text not null,
    shipping_date timestamptz null,
    delivery_estimate timestamptz null,
    item text not null,
    delivered boolean not null default false
);

create table if not exists knowledge_documents (
    id text primary key,
    title text not null,
    category text not null,
    content text not null
);

-- embedding column uses 512 dims (voyage-3-lite)
-- migration for existing deployments: ALTER TABLE knowledge_chunks ALTER COLUMN embedding TYPE vector(512);
create table if not exists knowledge_chunks (
    id bigserial primary key,
    document_id text not null references knowledge_documents(id) on delete cascade,
    chunk_text text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(512),
    search_vector tsvector generated always as (to_tsvector('english', chunk_text)) stored
);

create index if not exists idx_support_orders_status on support_orders(status);
create index if not exists idx_knowledge_chunks_search_vector on knowledge_chunks using gin(search_vector);
create index if not exists idx_knowledge_chunks_embedding on knowledge_chunks using hnsw (embedding vector_cosine_ops);
