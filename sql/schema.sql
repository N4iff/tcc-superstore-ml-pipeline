-- Superstore DB Schema (PostgreSQL)
-- Tables: raw_orders, processed_orders, predictions

CREATE TABLE IF NOT EXISTS raw_orders (
  id BIGSERIAL PRIMARY KEY,
  row_id INTEGER,
  order_id TEXT,
  order_date DATE,
  ship_date DATE,
  ship_mode TEXT,
  customer_id TEXT,
  segment TEXT,
  country TEXT,
  city TEXT,
  state TEXT,
  postal_code TEXT,
  region TEXT,
  product_id TEXT,
  category TEXT,
  sub_category TEXT,
  product_name TEXT,
  sales NUMERIC,
  quantity INTEGER,
  discount NUMERIC,
  profit NUMERIC,
  source_file TEXT,
  ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_orders (
  id BIGSERIAL PRIMARY KEY,
  raw_id BIGINT REFERENCES raw_orders(id),
  order_date DATE,
  ship_date DATE,
  ship_mode TEXT,
  segment TEXT,
  city TEXT,
  state TEXT,
  region TEXT,
  category TEXT,
  sub_category TEXT,
  sales NUMERIC,
  quantity INTEGER,
  discount NUMERIC,
  profit NUMERIC,
  -- optional engineered features
  days_to_ship INTEGER,
  profit_margin NUMERIC,
  processed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  raw_id BIGINT REFERENCES raw_orders(id),
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  target TEXT NOT NULL,              -- e.g., "sales" or "profit"
  prediction NUMERIC NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_orders_order_id ON raw_orders(order_id);
CREATE INDEX IF NOT EXISTS idx_processed_orders_raw_id ON processed_orders(raw_id);
CREATE INDEX IF NOT EXISTS idx_predictions_raw_id ON predictions(raw_id);
