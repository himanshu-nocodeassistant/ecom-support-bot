-- Migration 7: Persistent Customer Memory
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks)

-- 7a: Customer identity
create table if not exists customers (
    customer_id uuid primary key default gen_random_uuid(),
    email       text not null unique,
    name        text not null,
    created_at  timestamptz not null default now()
);

create table if not exists customer_sessions (
    session_id      text primary key,
    customer_id     uuid not null references customers(customer_id) on delete cascade,
    started_at      timestamptz not null default now(),
    last_active_at  timestamptz not null default now()
);

create index if not exists idx_customer_sessions_customer_id
    on customer_sessions(customer_id);

-- 7b: Durable conversation history (replaces in-process SESSION_MEMORY dict)
create table if not exists conversation_turns (
    id          bigserial primary key,
    session_id  text not null,
    role        text not null check (role in ('user', 'assistant')),
    content     text not null,
    tool_events jsonb null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_conversation_turns_session_id
    on conversation_turns(session_id, created_at);

-- 7c: Cross-session memory facts
create table if not exists customer_memory (
    id                bigserial primary key,
    customer_id       uuid not null references customers(customer_id) on delete cascade,
    fact_type         text not null,  -- 'order_preference' | 'issue_history' | 'product_interest' | 'communication_style'
    fact_text         text not null,
    confidence        numeric(3,2) not null check (confidence between 0 and 1),
    source_session_id text null,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    unique (customer_id, fact_type)   -- one fact per type per customer (upsert by type)
);

create index if not exists idx_customer_memory_customer_id
    on customer_memory(customer_id);

-- 7e: Customer order linkage
create table if not exists customer_orders (
    customer_id uuid not null references customers(customer_id) on delete cascade,
    order_id    text not null,
    linked_at   timestamptz not null default now(),
    primary key (customer_id, order_id)
);

create index if not exists idx_customer_orders_customer_id
    on customer_orders(customer_id, linked_at desc);
