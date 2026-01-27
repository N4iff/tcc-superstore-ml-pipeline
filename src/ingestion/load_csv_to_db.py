import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "SampleSuperstore.csv"

TABLE_NAME = "raw_superstore"

CSV_TO_DB_COLS = {
    "Ship Mode": "ship_mode",
    "Segment": "segment",
    "Country": "country",
    "City": "city",
    "State": "state",
    "Postal Code": "postal_code",
    "Region": "region",
    "Category": "category",
    "Sub-Category": "sub_category",
    "Sales": "sales",
    "Quantity": "quantity",
    "Discount": "discount",
    "Profit": "profit",
}


def get_conn():
    load_dotenv(PROJECT_ROOT / ".env")  # local only, not committed
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "superstore")
    user = os.getenv("DB_USER", "superstore_user")
    password = os.getenv("DB_PASSWORD", "REDACTED")

    return psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    )


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in CSV_TO_DB_COLS.keys() if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV columns mismatch. Missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep only expected columns and rename them
    df = df[list(CSV_TO_DB_COLS.keys())].rename(columns=CSV_TO_DB_COLS)

    # Clean types (safe conversions)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    for col in ["sales", "discount", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # postal_code as text (keep as string)
    df["postal_code"] = df["postal_code"].astype(str)

    # Basic NA handling: keep None so Postgres accepts NULL
    df = df.where(pd.notnull(df), None)

    return df


def insert_rows(conn, df: pd.DataFrame, source_file: str):
    cols = list(df.columns) + ["source_file"]

    # Convert numpy/pandas types -> plain Python types (int/float/str/None)
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].apply(lambda x: x.item() if hasattr(x, "item") else x)

    df2["source_file"] = source_file

    values = list(df2.itertuples(index=False, name=None))

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} ({", ".join(cols)})
        VALUES %s
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values, page_size=1000)



def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    df = load_csv(csv_path)

    print(f"Loaded CSV: {csv_path}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    conn = get_conn()
    try:
        conn.autocommit = False

        # Optional: avoid double-loading by truncating raw table first (comment out if you don't want that)
        # with conn.cursor() as cur:
        #     cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY;")

        insert_rows(conn, df, source_file=csv_path.name)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            total = cur.fetchone()[0]
        print(f"✅ Inserted {len(df):,} rows into {TABLE_NAME}. Total rows now: {total:,}")

    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
