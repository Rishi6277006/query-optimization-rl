# Query Optimization using Reinforcement Learning

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

This project demonstrates how **Reinforcement Learning (RL)** can be used to optimize SQL query execution plans in a database environment. The goal is to reduce query execution costs by dynamically adjusting query planner parameters such as join methods, index usage, and memory allocation.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Future Work](#future-work)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## **Overview**
Query optimization is a critical aspect of database systems, as it directly impacts performance. Traditional query optimizers rely on static rules and cost models, which may not always produce the most efficient execution plans. This project explores the use of **Reinforcement Learning (RL)** to dynamically optimize query execution plans based on real-time feedback.

The RL agent interacts with a PostgreSQL database and learns to adjust query planner parameters (e.g., enabling/disabling hash joins, index scans, etc.) to minimize query execution costs. The project uses the **TPC-H benchmark dataset** for training and evaluation.

---

## **Features**
- **Reinforcement Learning Environment**: A custom Gym environment for query optimization.
- **Q-Learning Agent**: Implements a Q-learning algorithm to optimize query execution plans.
- **PostgreSQL Integration**: Connects to a PostgreSQL database to execute and analyze queries.
- **Dynamic Query Tuning**: Adjusts query planner parameters (e.g., `enable_hashjoin`, `enable_indexscan`, `work_mem`) to reduce execution costs.
- **Training and Evaluation**: Includes scripts for training the RL agent and evaluating its performance on test queries.

---

## **Installation**
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/query-optimization-rl.git
   cd query-optimization-rl
Set up a PostgreSQL database:
Install PostgreSQL and create a database (e.g., tpch_db).
Load the TPC-H dataset using the provided script (load_tcp_h.py).
Install dependencies:
bash
Copy
pip install -r requirements.txt
Run the training script:
bash
Copy
python src/train_rl_agent.py
Usage

1. Training the RL Agent

To train the RL agent, run the following command:

bash
Copy
python src/train_rl_agent.py
The agent will interact with the PostgreSQL database, optimize query execution plans, and save the trained Q-table.

2. Evaluating the Agent

To evaluate the trained agent on test queries, use the following script:

bash
Copy
python src/test_environment.py
3. Custom Queries

You can modify the test queries in src/test_environment.py to evaluate the agent on different workloads.

Results

Training Progress

The following graph shows the training progress of the RL agent over multiple episodes. The agent learns to reduce query execution costs by adjusting query planner parameters.

<img width="642" alt="Screenshot 2025-03-08 at 10 57 51 AM" src="https://github.com/user-attachments/assets/ba7fb53e-bf51-4688-8e1d-aaf41900a330" />

Performance Metrics

Best Reward Achieved: 236.54
Average Cost Reduction: 15%
Future Work

Deep Reinforcement Learning: Replace Q-learning with Deep Q-Networks (DQN) or Policy Gradient methods for better performance.
State Space Refinement: Experiment with additional state features (e.g., query complexity, database statistics).
Multi-Query Optimization: Extend the agent to optimize multiple queries simultaneously.
Integration with Real Databases: Test the agent on real-world database workloads.
Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

OpenAI Gym for the RL environment framework.
TPC-H Benchmark for the dataset.
PostgreSQL for the database backend.
Contact

For questions or feedback, feel free to reach out:

Name: Rishikesh Thakker
Email: thakker834@gmail.com
LinkedIn: www.linkedin.com/in/rishikesh-thakker-318078275

