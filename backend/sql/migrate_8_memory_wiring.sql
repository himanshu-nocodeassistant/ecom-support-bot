-- Migration 8: Memory wiring — add expires_at TTL to customer_memory
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks)

-- 8a: Add expires_at column with a 90-day default for existing rows
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customer_memory' AND column_name = 'expires_at'
    ) THEN
        ALTER TABLE customer_memory
            ADD COLUMN expires_at timestamptz NOT NULL
                DEFAULT (now() + interval '90 days');
    END IF;
END $$;

-- 8b: Index for efficient TTL filtering
CREATE INDEX IF NOT EXISTS idx_customer_memory_expires_at
    ON customer_memory(expires_at);
