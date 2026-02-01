import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


RAW_TABLE = "raw_superstore"
PROCESSED_TABLE = "processed_superstore"


def get_conn():
    load_dotenv(PROJECT_ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "superstore"),
        user=os.getenv("DB_USER", "superstore_user"),
        password=os.getenv("DB_PASSWORD", "REDACTED"),
    )


def get_table_columns(cur, table_name: str):
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position;
        """,
        (table_name,),
    )
    return [r[0] for r in cur.fetchall()]


def main():
    conn = get_conn()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            raw_cols = set(get_table_columns(cur, RAW_TABLE))
            proc_cols = get_table_columns(cur, PROCESSED_TABLE)
            proc_cols_set = set(proc_cols)

            # Expressions we know how to build from raw
            expr_map = {
                # linkage
                "raw_id": "r.id",

                # same fields (cleaned)
                "ship_mode": "NULLIF(TRIM(r.ship_mode), '')",
                "segment": "NULLIF(TRIM(r.segment), '')",
                "country": "NULLIF(TRIM(r.country), '')",
                "city": "NULLIF(TRIM(r.city), '')",
                "state": "NULLIF(TRIM(r.state), '')",
                "postal_code": "NULLIF(TRIM(r.postal_code), '')",
                "region": "NULLIF(TRIM(r.region), '')",
                "category": "NULLIF(TRIM(r.category), '')",
                "sub_category": "NULLIF(TRIM(r.sub_category), '')",

                "sales": "r.sales",
                "quantity": "r.quantity",
                "discount": "r.discount",
                "profit": "r.profit",

                "source_file": "r.source_file",

                # engineered features
                "profit_margin": "CASE WHEN r.sales IS NULL OR r.sales = 0 THEN NULL ELSE (r.profit / r.sales) END",
                "sales_log": "CASE WHEN r.sales IS NULL THEN NULL ELSE LN(r.sales + 1) END",
                "discount_bucket": """
                    CASE
                      WHEN r.discount IS NULL THEN NULL
                      WHEN r.discount = 0 THEN '0'
                      WHEN r.discount <= 0.10 THEN '0-10%'
                      WHEN r.discount <= 0.20 THEN '10-20%'
                      WHEN r.discount <= 0.30 THEN '20-30%'
                      ELSE '30%+'
                    END
                """.strip(),

                # timestamps (if exists in processed table)
                "processed_at": "NOW()",
            }

            # Build insert list only for columns that exist in processed table AND we can compute
            insert_cols = []
            select_exprs = []

            # Ignore identity/PK columns
            skip_cols = {"id"}  # processed table PK

            for c in proc_cols:
                if c in skip_cols:
                    continue
                if c in expr_map:
                    # ensure raw dependency exists when needed
                    if expr_map[c].startswith("r.") and c not in raw_cols and c not in ("raw_id", "processed_at"):
                        continue
                    insert_cols.append(c)
                    select_exprs.append(f"{expr_map[c]} AS {c}")

            if not insert_cols:
                raise RuntimeError(
                    "No matching columns to insert. Check processed table schema vs raw table."
                )

            # 1) Truncate processed (clean rebuild)
            cur.execute(f"TRUNCATE TABLE {PROCESSED_TABLE} RESTART IDENTITY;")

            # 2) Insert from raw
            sql = f"""
                INSERT INTO {PROCESSED_TABLE} ({", ".join(insert_cols)})
                SELECT {", ".join(select_exprs)}
                FROM {RAW_TABLE} r;
            """
            cur.execute(sql)

            # 3) Counts
            cur.execute(f"SELECT COUNT(*) FROM {RAW_TABLE};")
            raw_count = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {PROCESSED_TABLE};")
            proc_count = cur.fetchone()[0]

        conn.commit()
        print(f"✅ Processed rebuild done. raw={raw_count:,} processed={proc_count:,}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
