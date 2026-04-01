import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def main() -> None:
    """
    Creates the SeaBorg PostgreSQL schema objects (table + indexes).

    Reads:
      - DATABASE_URL from environment / .env

    Side effects:
      - Connects to PostgreSQL
      - Creates table `argo_profiles` and indexes if they do not exist
      - Prints "Database ready." on success
    """

    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is not set. Add it to .env (see .env.example) and retry."
        )

    engine = create_engine(database_url, future=True)

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS argo_profiles (
        id SERIAL PRIMARY KEY,
        float_id VARCHAR(20) NOT NULL,
        date TIMESTAMP NOT NULL,
        latitude FLOAT,
        longitude FLOAT,
        depth_m FLOAT,
        temp_c FLOAT,
        salinity FLOAT,
        oxygen FLOAT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

    create_idx_float_id_sql = """
    CREATE INDEX IF NOT EXISTS idx_float_id ON argo_profiles (float_id);
    """

    create_idx_date_sql = """
    CREATE INDEX IF NOT EXISTS idx_date ON argo_profiles (date);
    """

    create_idx_lat_lon_sql = """
    CREATE INDEX IF NOT EXISTS idx_lat_lon ON argo_profiles (latitude, longitude);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
            conn.execute(text(create_idx_float_id_sql))
            conn.execute(text(create_idx_date_sql))
            conn.execute(text(create_idx_lat_lon_sql))
    except Exception as e:
        raise SystemExit(f"Failed to initialise database: {e}") from e

    print("Database ready.")


if __name__ == "__main__":
    main()
