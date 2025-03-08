from sqlalchemy import create_engine
import pandas as pd

class DatabaseConnector:
    def __init__(self, host, port, user, password, database):
        # Create a connection string
        self.connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.connection_string)
    
    def execute_query(self, query):
        """
        Execute a SQL query and return the result as a DataFrame.
        """
        with self.engine.connect() as connection:
            result = pd.read_sql(query, connection)
        return result
    
    def explain_query(self, query):
        """
        Run the EXPLAIN command on a query and return the execution plan.
        """
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        with self.engine.connect() as connection:
            result = connection.execute(explain_query)
            plan = result.fetchone()[0]  # Extract the JSON plan
        return plan