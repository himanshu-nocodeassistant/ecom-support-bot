-- Migration 5c: add doc_type and date_updated to knowledge_documents
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks)

alter table knowledge_documents
    add column if not exists doc_type text not null default 'guide';

alter table knowledge_documents
    add column if not exists date_updated date null;

-- Index for metadata category filtering (5e)
create index if not exists idx_knowledge_chunks_metadata_category
    on knowledge_chunks using gin ((metadata -> 'category'));
