CREATE TABLE IF NOT EXISTS runs (
    run_id            uuid PRIMARY KEY,
    split             text NOT NULL CHECK (split IN ('development', 'check')),
    agent_version     text NOT NULL,
    prompt_version    text NOT NULL,
    dataset_version   text NOT NULL,
    evaluator_version text NOT NULL,
    fact_version      text NOT NULL,
    provider          text NOT NULL,
    model             text NOT NULL,
    settings          jsonb NOT NULL,
    file_hashes       jsonb NOT NULL,
    source_revision   text,
    status            text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    counts            jsonb,
    metrics           jsonb,          -- null for check runs
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);

CREATE TABLE IF NOT EXISTS answer_results (
    run_id        uuid NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    answer_id     text NOT NULL,
    status        text NOT NULL CHECK (status IN ('ok', 'error')),
    error_type    text,
    error_message text,
    latency_ms    integer,
    trace_id      text,
    claim_count   integer NOT NULL DEFAULT 0,
    scores        jsonb,              -- per-answer extraction/joint F1 on development, null on check
    PRIMARY KEY (run_id, answer_id)
);

CREATE TABLE IF NOT EXISTS claim_results (
    run_id         uuid NOT NULL,
    answer_id      text NOT NULL,
    claim_id       text NOT NULL,
    start_char     integer NOT NULL CHECK (start_char >= 0),
    end_char       integer NOT NULL CHECK (end_char > start_char),
    quote          text NOT NULL,
    label          text NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
    fact_ids       jsonb NOT NULL,
    reason         text NOT NULL,
    matched_gold   boolean,           -- development only
    expected_label text,              -- development only
    label_correct  boolean,           -- development only
    PRIMARY KEY (run_id, answer_id, start_char, end_char),
    UNIQUE (run_id, claim_id),
    FOREIGN KEY (run_id, answer_id) REFERENCES answer_results (run_id, answer_id) ON DELETE CASCADE
);
