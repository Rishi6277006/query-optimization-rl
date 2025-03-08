from query_optimization_env import QueryOptimizationEnv

def test_environment():
    # Create the environment
    env = QueryOptimizationEnv()

    try:
        # Reset the environment
        state = env.reset()
        print("Initial State:", state)

        # Take a random action
        action = env.action_space.sample()
        print("Action Taken:", action)

        # Step the environment
        next_state, reward, done, info = env.step(action)
        print("Next State:", next_state)
        print("Reward:", reward)

    finally:
        # Close the environment
        env.close()

if __name__ == "__main__":
    test_environment()