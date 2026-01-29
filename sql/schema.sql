-- SampleSuperstore (CSV has 13 columns)
-- Raw -> Processed -> Predictions

CREATE TABLE IF NOT EXISTS raw_superstore (
  id BIGSERIAL PRIMARY KEY,

  ship_mode TEXT,
  segment TEXT,
  country TEXT,
  city TEXT,
  state TEXT,
  postal_code TEXT,
  region TEXT,
  category TEXT,
  sub_category TEXT,

  sales NUMERIC,
  quantity INTEGER,
  discount NUMERIC,
  profit NUMERIC,

  source_file TEXT,
  ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_superstore (
  id BIGSERIAL PRIMARY KEY,
  raw_id BIGINT REFERENCES raw_superstore(id),

  ship_mode TEXT,
  segment TEXT,
  country TEXT,
  city TEXT,
  state TEXT,
  postal_code TEXT,
  region TEXT,
  category TEXT,
  sub_category TEXT,

  sales NUMERIC,
  quantity INTEGER,
  discount NUMERIC,
  profit NUMERIC,

  -- optional engineered features
  profit_margin NUMERIC,

  processed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  raw_id BIGINT REFERENCES raw_superstore(id),

  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  target TEXT NOT NULL,          -- "profit_margin" (final target)
  prediction NUMERIC NOT NULL,

  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_superstore_category ON raw_superstore(category);
CREATE INDEX IF NOT EXISTS idx_processed_superstore_raw_id ON processed_superstore(raw_id);
CREATE INDEX IF NOT EXISTS idx_predictions_raw_id ON predictions(raw_id);
