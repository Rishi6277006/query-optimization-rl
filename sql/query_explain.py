import psycopg2

# Database connection parameters
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "rishithakker"  # Replace with your username
DB_PASSWORD = ""          # Leave empty if no password is set
DB_NAME = "tpch_db"

def get_execution_plan(query):
    """Run EXPLAIN (FORMAT JSON) on a query and return the execution plan."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()

        # Run EXPLAIN (FORMAT JSON)
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        cursor.execute(explain_query)
        result = cursor.fetchone()[0]  # Fetch the JSON result

        # Debugging: Print the raw result
        print("Raw Result:")
        print(result)

        # Handle the result based on its type
        if isinstance(result, list):
            execution_plan = result  # Already a list of dictionaries
        elif isinstance(result, dict):
            execution_plan = [result]  # Wrap the dictionary in a list
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

        return execution_plan

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        cursor.close()
        conn.close()

def extract_features(execution_plan):
    """Extract key features from the execution plan."""
    features = {}

    # Extract top-level plan details
    plan = execution_plan[0]["Plan"]
    features["Node Type"] = plan.get("Node Type", "Unknown")
    features["Startup Cost"] = plan.get("Startup Cost", 0)
    features["Total Cost"] = plan.get("Total Cost", 0)

    # Extract relation name (if applicable)
    features["Relation Name"] = plan.get("Relation Name", "N/A")

    # Extract scan type (if applicable)
    if "Scan" in features["Node Type"]:
        features["Scan Type"] = features["Node Type"]

    # Extract sub-plans (if any)
    if "Plans" in plan:
        features["Sub-Plans"] = len(plan["Plans"])
    else:
        features["Sub-Plans"] = 0

    return features

def main():
    # Example query
    query = """
    SELECT l_orderkey, SUM(l_quantity)
    FROM lineitem
    WHERE l_shipdate >= '1996-01-01' AND l_shipdate < '1997-01-01'
    GROUP BY l_orderkey;
    """

    # Get the execution plan
    execution_plan = get_execution_plan(query)
    if execution_plan:
        print("\nExecution Plan:")
        print(execution_plan)

        # Extract features
        features = extract_features(execution_plan)
        print("\nExtracted Features:")
        for key, value in features.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()