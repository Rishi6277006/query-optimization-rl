import gym
from gym import spaces
import psycopg2
import numpy as np
import json
import subprocess
import re

# Database connection parameters
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "rishithakker"  # Replace with your username
DB_PASSWORD = ""          # Leave empty if no password is set
DB_NAME = "tpch_db"

# Define bin edges for each feature
bins = [
    [0, 1],                          # Node Type: Aggregate (binary)
    [0, 1],                          # Node Type: Seq Scan (binary)
    np.linspace(0, 500000, 20),      # Total Cost (20 bins)
    np.linspace(0, 1000000, 20),     # Plan Rows (20 bins)
    np.linspace(0, 100, 10),         # Plan Width (10 bins)
    [0, 1],                          # Sub-Plans (binary)
    [0, 1],                          # Hash Join Indicator (binary)
    [0, 1],                          # Index Scan Indicator (binary)
    np.linspace(0, 1000000, 20),     # Actual Rows (20 bins)
    np.linspace(0, 100, 10),         # Actual Loops (10 bins)
    np.linspace(0, 10, 5),           # Number of Sub-Plans (5 bins)
]

num_bins_per_feature = [len(b) - 1 for b in bins]  # Number of bins for each feature
total_states = np.prod(num_bins_per_feature)  # Total number of states
print(f"Total number of states: {total_states}")


class QueryOptimizationEnv(gym.Env):
    def __init__(self):
        super(QueryOptimizationEnv, self).__init__()

        # Initialize environment attributes
        self.current_cost = None
        self.best_cost = float('inf')  # Initialize best_cost to a large value
        self.bins = bins

        # Define the action space
        self.action_space = spaces.Discrete(3)  # 3 possible actions

        # Define the state space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, np.inf, np.inf, np.inf, np.inf, 1, 1, 1], dtype=np.float32),
            shape=(9,),  # Number of features in the state
            dtype=np.float32
        )

        # Initialize the database connection
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            self.conn = None
            self.cursor = None

        # Initialize the query
        self.query = """
        SELECT o_orderkey, l_linenumber, SUM(l_quantity)
        FROM lineitem
        JOIN orders ON l_orderkey = o_orderkey
        WHERE l_shipdate >= '1996-01-01' AND l_shipdate < '1997-01-01'
        GROUP BY o_orderkey, l_linenumber;
        """

    def fetch_initial_state(self):
        """
        Fetch the initial query plan state.
        
        :return: JSON-formatted query plan.
        """
        query = f"""
        {self.query}
        """
        return self.execute_query(query)

    def fetch_current_state(self, modified_query):
        """
        Fetch the current query plan state after applying an action.
        
        :param modified_query: Modified SQL query.
        :return: JSON-formatted query plan.
        """
        query = f"""
        {modified_query}
        """
        return self.execute_query(query)


    def execute_query(self, query):
        try:
            # Split the query into SET commands and the actual query
            set_commands = []
            actual_query = ""
            for line in query.splitlines():
                if line.strip().upper().startswith("SET"):
                    set_commands.append(line.strip())
                else:
                    actual_query += line + "\n"

            # Start a transaction
            self.cursor.execute("BEGIN;")

            # Execute the SET commands
            for set_command in set_commands:
                self.cursor.execute(set_command)

            # Execute the actual query with EXPLAIN
            self.cursor.execute(f"EXPLAIN (FORMAT JSON) {actual_query}")
            result = self.cursor.fetchone()[0]

            # Commit the transaction
            self.cursor.execute("COMMIT;")

            print("Query executed successfully.")
            return result

        except Exception as e:
            # Rollback the transaction in case of an error
            self.cursor.execute("ROLLBACK;")
            print(f"Error executing query: {e}")
            return None

    def extract_cost_from_plan(self, query_plan):
        """
        Extract the total cost from the query plan.
        
        :param query_plan: Parsed JSON query plan.
        :return: Total cost as a float.
        """
        try:
            # Ensure the query plan is a list
            if isinstance(query_plan, list) and len(query_plan) > 0:
                total_cost = query_plan[0]["Plan"]["Total Cost"]
                return float(total_cost)
            else:
                raise ValueError("Unexpected type for query plan.")
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error extracting cost from query plan: {e}")
            return None
        
    def calculate_reward(self, new_cost, new_execution_time=None):
        if new_cost is None:  # Invalid query plan
            return -100  # Large penalty for invalid actions
        
        if new_cost == self.current_cost:  # No change in query plan
            return -0.1  # Small penalty for no change
        
        cost_reduction = (self.current_cost - new_cost) / self.current_cost
        if cost_reduction > 0:  # Cost reduced
            reward = cost_reduction * 100  # Scale up the reward for larger reductions
        else:  # Cost increased
            reward = cost_reduction * 50  # Scale up the penalty for increased cost
        
        # Bonus for achieving a new best cost
        if new_cost < self.best_cost:
            self.best_cost = new_cost
            reward += 20  # Bonus for achieving a new best cost
        
        if new_execution_time is not None:  # Incorporate execution time
            reward -= new_execution_time  # Penalize longer execution times
        
        return reward

    def extract_state(self, query_plan):
        try:
            plan = query_plan[0]["Plan"]
            state = [
                1 if plan.get("Node Type") == "Aggregate" else 0,  # Node Type: Aggregate
                1 if plan.get("Node Type") == "Seq Scan" else 0,   # Node Type: Seq Scan
                plan.get("Total Cost", 0),                        # Total Cost
                plan.get("Plan Rows", 0),                         # Plan Rows
                plan.get("Plan Width", 0),                        # Plan Width
                1 if "Subplan Name" in plan else 0,               # Sub-Plans
                1 if plan.get("Node Type") == "Hash Join" else 0, # Hash Join Indicator
                1 if plan.get("Node Type") == "Index Scan" else 0, # Index Scan Indicator
                plan.get("Actual Rows", 0),                       # Actual Rows
                plan.get("Actual Loops", 0),                      # Actual Loops
                len(plan.get("Plans", [])),                       # Number of Sub-Plans
            ]
            return np.array(state, dtype=np.float32)
        except (KeyError, TypeError) as e:
            print(f"Error extracting state from query plan: {e}")
            return None

    # Ensure discretization works correctly
    def discretize_state(self, state):
        if state is None:
            print("Cannot discretize state: State is None.")
            return None

        discretized = []
        for i, feature in enumerate(state):
            # Find the bin index for the current feature
            bin_index = np.digitize(feature, self.bins[i]) - 1
            # Ensure the bin index is within bounds
            bin_index = max(0, min(bin_index, len(self.bins[i]) - 2))
            discretized.append(bin_index)

        # Convert the discretized state into a unique state index
        state_index = 0
        multiplier = 1
        for i, bin_index in enumerate(reversed(discretized)):
            state_index += bin_index * multiplier
            multiplier *= len(self.bins[i])

        # Ensure the state index is within the bounds of the total number of states
        total_states = np.prod([len(b) - 1 for b in self.bins])
        state_index = min(state_index, total_states - 1)

        print(f"State Features: {state}")
        print(f"Discretized Bins: {discretized}")
        print(f"State Index: {state_index}")

        return int(state_index)

    def reset(self):
        """Reset the environment to a new query."""
        print("Resetting environment...")
        raw_state = self.fetch_initial_state()
        if raw_state is None:
            raise ValueError("Failed to fetch initial state: Query plan is invalid or missing.")
        print("Raw state fetched successfully.")

        # Extract the initial cost from the query plan
        self.current_cost = self.extract_cost_from_plan(raw_state)
        if self.current_cost is None:
            raise ValueError("Failed to extract cost from query plan.")
        print(f"Initial cost: {self.current_cost}")

        # Extract state features (continuous values)
        state_features = self.extract_state(raw_state)
        if state_features is None:
            raise ValueError("Failed to extract state features.")
        print(f"State features: {state_features}")

        # Return the raw state features (continuous values)
        return state_features

    def step(self, action):
        """Execute one step in the environment given the action."""
        modified_query = self.apply_action(action)
        new_raw_state = self.fetch_current_state(modified_query)
        new_cost = self.extract_cost_from_plan(new_raw_state)

        # Calculate the reward
        reward = self.calculate_reward(new_cost)

        # Extract state features (continuous values)
        state_features = self.extract_state(new_raw_state)
        if state_features is None:
            raise ValueError("Failed to extract state features.")

        # Discretize the state
        next_state = self.discretize_state(state_features)
        print(f"Discretized State: {next_state}")

        # Check if done (for now, always False)
        done = False

        # Return the discretized state, reward, done flag, and additional info
        return next_state, reward, done, {"cost": new_cost}

    def apply_action(self, action):
        if action == 0:
            # Action 0: Force Hash Join
            modified_query = f"""
            SET LOCAL enable_hashjoin TO on;
            {self.query}
            """
        elif action == 1:
            # Action 1: Force Index Scan
            modified_query = f"""
            SET LOCAL enable_indexscan TO on;
            SET LOCAL enable_seqscan TO off;
            {self.query}
            """
        elif action == 2:
            # Action 2: Increase work_mem to encourage parallelism
            modified_query = f"""
            SET LOCAL work_mem TO '64MB';
            {self.query}
            """
        elif action == 3:
            # Action 3: Enable Nested Loop
            modified_query = f"""
            SET LOCAL enable_nestloop TO on;
            {self.query}
            """
        elif action == 4:
            # Action 4: Disable specific indexes
            modified_query = f"""
            SET LOCAL enable_indexscan TO off;
            {self.query}
            """
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return modified_query

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()