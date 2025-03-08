from query_optimization_env import QueryOptimizationEnv
from rl_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

# Define the list of possible actions
ACTIONS = [0, 1, 2]  # Corresponding to the 3 actions: Hash Join, Index Scan, and Increase work_mem

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

# Calculate the total number of states
num_bins_per_feature = [len(b) - 1 for b in bins]
total_states = np.prod(num_bins_per_feature)
print(f"Total number of states: {total_states}")

# Initialize the agent with the correct state space size
agent = QLearningAgent(
    actions=ACTIONS,
    state_space_size=total_states,  # Use the correct state space size
    learning_rate=0.2,
    discount_factor=0.8,
    epsilon=1.0,
    epsilon_decay=0.99,
    epsilon_min=0.2
)

def train_agent(env, agent, episodes=1000, early_stopping_threshold=10):
    best_reward = -float("inf")
    no_improvement_count = 0
    rewards = []

    for episode in range(episodes):
        state_features = env.reset()
        if state_features is None:
            print("Failed to reset environment. Skipping episode.")
            continue

        state = env.discretize_state(state_features)
        done = False
        total_reward = 0
        total_cost = 0
        step_count = 0

        while not done:
            step_count += 1
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            total_cost += info.get("cost", 0)

            if step_count > 100:  # Safeguard to prevent infinite steps
                print("Breaking out of the loop after 100 steps.")
                done = True

        agent.decay_epsilon()
        rewards.append(total_reward)

        # Early stopping logic
        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_threshold:
            print(f"Early stopping at episode {episode + 1}.")
            break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Best Reward = {best_reward}, No Improvement Count = {no_improvement_count}")

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

    print(f"Training completed after {episode + 1} episodes.")


def evaluate_agent(env, agent, test_queries):
    """Evaluate the trained RL agent on test queries."""
    total_reward = 0
    total_cost = 0

    for query in test_queries:
        env.query = query  # Set the test query
        state = env.reset()
        done = False

        print(f"\nEvaluating Query: {query}")

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            total_cost += info.get("cost", 0)

            # Log evaluation details
            print(f"  State: {state}, Action: {action}, Reward: {reward}, Cost: {info.get('cost', 0)}")

    print(f"\nEvaluation Results:")
    print(f"  Total Reward: {total_reward}")
    print(f"  Total Cost: {total_cost}")

if __name__ == "__main__":
    # Initialize environment and agent
    env = QueryOptimizationEnv()
    agent = QLearningAgent(
        actions=ACTIONS,
        state_space_size=total_states,
        learning_rate=0.2,
        discount_factor=0.8,
        epsilon=1.0,
        epsilon_decay=0.99,  # Slower decay
        epsilon_min=0.2  # Higher minimum epsilon
    )

    # Train the agent
    train_agent(env, agent, episodes=5000)

    # Save the Q-table
    agent.save_q_table("q_table.pkl")

    test_queries = [
    """
    SELECT o_orderkey, l_linenumber, SUM(l_quantity)
    FROM lineitem
    JOIN orders ON l_orderkey = o_orderkey
    WHERE l_shipdate >= '1996-01-01' AND l_shipdate < '1997-01-01'
    GROUP BY o_orderkey, l_linenumber;
    """,
    """
    SELECT c_custkey, c_name, SUM(o_totalprice)
    FROM customer
    JOIN orders ON c_custkey = o_custkey
    GROUP BY c_custkey, c_name;
    """,
    """
    SELECT p_partkey, p_name, SUM(l_quantity)
    FROM part
    JOIN lineitem ON p_partkey = l_partkey
    GROUP BY p_partkey, p_name;
    """,
    # Add more queries here
]

    # Evaluate the agent
    evaluate_agent(env, agent, test_queries)

    # Close the environment
    env.close()