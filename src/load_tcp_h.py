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
    ],
    "customer": [
        ("c_custkey", "BIGINT"),
        ("c_name", "VARCHAR(25)"),
        ("c_address", "VARCHAR(40)"),
        ("c_nationkey", "BIGINT"),
        ("c_phone", "CHAR(15)"),
        ("c_acctbal", "DECIMAL(15,2)"),
        ("c_mktsegment", "CHAR(10)"),
        ("c_comment", "VARCHAR(117)")
    ],
    "part": [
        ("p_partkey", "BIGINT"),
        ("p_name", "VARCHAR(55)"),
        ("p_mfgr", "CHAR(25)"),
        ("p_brand", "CHAR(10)"),
        ("p_type", "VARCHAR(25)"),
        ("p_size", "INTEGER"),
        ("p_container", "CHAR(10)"),
        ("p_retailprice", "DECIMAL(15,2)"),
        ("p_comment", "VARCHAR(23)")
    ],
    "supplier": [
        ("s_suppkey", "BIGINT"),
        ("s_name", "CHAR(25)"),
        ("s_address", "VARCHAR(40)"),
        ("s_nationkey", "BIGINT"),
        ("s_phone", "CHAR(15)"),
        ("s_acctbal", "DECIMAL(15,2)"),
        ("s_comment", "VARCHAR(101)")
    ]
}

def create_table(cursor, table_name):
    """Create a table in PostgreSQL based on the schema."""
    columns = ", ".join([f"{col} {dtype}" for col, dtype in TABLE_SCHEMAS[table_name]])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
    cursor.execute(query)

def load_data(cursor, table_name, file_path):
    """Load data from a .tbl file into a PostgreSQL table."""
    try:
        # Read the .tbl file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="|", header=None, names=[col for col, _ in TABLE_SCHEMAS[table_name]])
        print(f"Read {len(df)} rows from {file_path}")
        print("First few rows of the DataFrame:")
        print(df.head())  # Print the first few rows of the DataFrame
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Remove rows with NaN values (caused by trailing delimiters)
    df = df.dropna()
    print(f"After dropping NaN rows: {len(df)} rows remaining")

    # Insert data into the table
    columns = ", ".join([col for col, _ in TABLE_SCHEMAS[table_name]])
    placeholders = ", ".join(["%s"] * len(TABLE_SCHEMAS[table_name]))
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    print(f"Insert query: {insert_query}")

    for i, row in df.iterrows():
        try:
            cursor.execute(insert_query, tuple(row))
            print(f"Inserted row {i + 1}: {tuple(row)}")
        except Exception as e:
            print(f"Error inserting row {i + 1}: {tuple(row)}. Error: {e}")
            break

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
        # Load tables
        print("Loading lineitem table...")
        create_table(cursor, "lineitem")
        load_data(cursor, "lineitem", f"{TPCH_DIR}/lineitem_fixed.tbl")

        print("Loading orders table...")
        create_table(cursor, "orders")
        load_data(cursor, "orders", f"{TPCH_DIR}/orders_fixed.tbl")

        print("Loading customer table...")
        create_table(cursor, "customer")
        load_data(cursor, "customer", f"{TPCH_DIR}/customer_fixed.tbl")

        print("Loading part table...")
        create_table(cursor, "part")
        load_data(cursor, "part", f"{TPCH_DIR}/part_fixed.tbl")

        print("Loading supplier table...")
        create_table(cursor, "supplier")
        load_data(cursor, "supplier", f"{TPCH_DIR}/supplier_fixed.tbl")

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