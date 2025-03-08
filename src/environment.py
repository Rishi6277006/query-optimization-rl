import gym
from gym import spaces
import numpy as np
import random

class QueryOptimizationEnv(gym.Env):
    def __init__(self):
        super(QueryOptimizationEnv, self).__init__()
        
        # Define the state space (query features + resource constraints)
        # For simplicity, we'll use a fixed-size vector of 10 features
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        # Define the action space (join order, index selection, parallelism)
        # For now, we'll assume 5 possible join orders
        self.action_space = spaces.Discrete(5)
        
        # Initialize state
        self.state = None
    
    def reset(self):
        """
        Reset the environment to a new query.
        Returns the initial state.
        """
        # Generate random query features for now
        self.state = np.random.rand(10)  # Replace with real query features later
        return self.state
    
    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action (int): The action taken by the agent (e.g., join order).
        Returns:
            state (array): New state after taking the action.
            reward (float): Reward for the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        # Simulate query execution based on the action
        cpu_usage, memory_usage, query_latency = self._execute_query(action)
        
        # Calculate reward
        reward = self.calculate_reward(cpu_usage, memory_usage, query_latency)
        
        # Generate a new state (for now, just random)
        self.state = np.random.rand(10)  # Replace with real state updates later
        
        # Check if done (for now, always False)
        done = False
        
        # Return the new state, reward, done flag, and additional info
        return self.state, reward, done, {}
    
    def _execute_query(self, action):
        """
        Simulate query execution based on the action.
        Returns CPU usage, memory usage, and query latency.
        """
        # Simulate resource usage and latency
        cpu_usage = random.uniform(10, 100)  # Random CPU usage between 10% and 100%
        memory_usage = random.uniform(10, 100)  # Random memory usage
        query_latency = random.uniform(0.1, 5)  # Random latency in seconds
        
        return cpu_usage, memory_usage, query_latency
    
    def calculate_reward(self, cpu_usage, memory_usage, query_latency):
        """
        Calculate the reward based on resource usage and latency.
        Lower resource usage and latency result in higher rewards.
        """
        cost_penalty = -(cpu_usage + memory_usage)  # Penalize high resource usage
        latency_penalty = -query_latency  # Penalize high latency
        return cost_penalty + latency_penalty