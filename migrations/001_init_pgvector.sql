CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_documents (
    id UUID PRIMARY KEY,
    company_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    effective_from DATE,
    effective_to DATE,
    source_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (company_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_policy_documents_company
    ON policy_documents (company_id);

CREATE TABLE IF NOT EXISTS policy_chunks (
    id UUID PRIMARY KEY,
    company_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (company_id, document_id)
        REFERENCES policy_documents (company_id, document_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_company
    ON policy_chunks (company_id);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding
    ON policy_chunks
    USING hnsw (embedding vector_cosine_ops);
