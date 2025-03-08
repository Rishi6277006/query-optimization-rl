import psycopg2
import pandas as pd

# Database connection parameters
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "rishithakker"  # Replace with your username
DB_PASSWORD = ""          # Leave empty if no password is set
DB_NAME = "tpch_db"

# Path to the TPC-H .tbl files
TPCH_DIR = "/Users/rishithakker/Downloads/tpch-dbgen-master/"

# Table schemas
TABLE_SCHEMAS = {
    "lineitem": [
        ("l_orderkey", "BIGINT"),
        ("l_partkey", "BIGINT"),
        ("l_suppkey", "BIGINT"),
        ("l_linenumber", "INTEGER"),
        ("l_quantity", "DECIMAL(15,2)"),
        ("l_extendedprice", "DECIMAL(15,2)"),
        ("l_discount", "DECIMAL(15,2)"),
        ("l_tax", "DECIMAL(15,2)"),
        ("l_returnflag", "CHAR(1)"),
        ("l_linestatus", "CHAR(1)"),
        ("l_shipdate", "DATE"),
        ("l_commitdate", "DATE"),
        ("l_receiptdate", "DATE"),
        ("l_shipinstruct", "CHAR(25)"),
        ("l_shipmode", "CHAR(10)"),
        ("l_comment", "VARCHAR(44)")
    ],
    "orders": [
        ("o_orderkey", "BIGINT"),
        ("o_custkey", "BIGINT"),
        ("o_orderstatus", "CHAR(1)"),
        ("o_totalprice", "DECIMAL(15,2)"),
        ("o_orderdate", "DATE"),
        ("o_orderpriority", "CHAR(15)"),
        ("o_clerk", "CHAR(15)"),
        ("o_shippriority", "INTEGER"),
        ("o_comment", "VARCHAR(79)")
    ]
    # Add schemas for other tables (e.g., customer, part, supplier, etc.)
}

def create_table(cursor, table_name):
    """Create a table in PostgreSQL based on the schema."""
    columns = ", ".join([f"{col} {dtype}" for col, dtype in TABLE_SCHEMAS[table_name]])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
    cursor.execute(query)

def load_data(cursor, table_name, file_path):
    """Load data from a .tbl file into a PostgreSQL table."""
    # Read the .tbl file into a pandas DataFrame
    df = pd.read_csv(file_path, sep="|", header=None, names=[col for col, _ in TABLE_SCHEMAS[table_name]])

    # Remove rows with NaN values (caused by trailing delimiters)
    df = df.dropna()

    # Insert data into the table
    columns = ", ".join([col for col, _ in TABLE_SCHEMAS[table_name]])
    placeholders = ", ".join(["%s"] * len(TABLE_SCHEMAS[table_name]))
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    for _, row in df.iterrows():
        cursor.execute(insert_query, tuple(row))

def main():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    cursor = conn.cursor()

    try:
        # Load lineitem table
        print("Loading lineitem table...")
        create_table(cursor, "lineitem")
        load_data(cursor, "lineitem", f"{TPCH_DIR}/lineitem.tbl")

        # Load orders table
        print("Loading orders table...")
        create_table(cursor, "orders")
        load_data(cursor, "orders", f"{TPCH_DIR}/orders.tbl")

        # Commit changes
        conn.commit()
        print("Data loaded successfully!")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()