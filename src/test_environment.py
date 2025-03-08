from environment import QueryOptimizationEnv

def test_environment():
    env = QueryOptimizationEnv()
    
    # Reset the environment
    state = env.reset()
    print("Initial State:", state)
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        print(f"Step {i + 1}: Action={action}, Reward={reward}, State={state}")

if __name__ == "__main__":
    test_environment()