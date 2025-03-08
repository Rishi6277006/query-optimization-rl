import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, actions, state_space_size, learning_rate=0.2, discount_factor=0.8, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.2):
        self.actions = actions
        self.q_table = np.zeros((state_space_size, len(actions)))  # Initialize Q-table with correct size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Start with high exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        """Choose an action based on the current state."""
        if not isinstance(state, int):
            print(f"Invalid state type: {type(state)}. State must be an integer.")
            return random.choice(self.actions)  # Fallback to random action

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning formula."""
        if next_state >= self.q_table.shape[0]:  # Ensure next_state is within bounds
            next_state = self.q_table.shape[0] - 1
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_table.pkl"):
        """Save the Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """Load the Q-table from a file."""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)