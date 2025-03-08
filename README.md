# **Query Optimization using Reinforcement Learning**  

![GitHub License](https://img.shields.io/github/license/Rishi6277006/query-optimization-rl)  
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)  

This project explores how **Reinforcement Learning (RL)** can optimize SQL query execution plans in **PostgreSQL** databases. By dynamically tuning query planner parameters—such as join methods, index usage, and memory allocation—the RL agent aims to **reduce query execution costs** and improve performance.  

## **Table of Contents**  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Future Work](#future-work)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [Contact](#contact)  

## **Overview**  

Query optimization is a fundamental challenge in database systems, as inefficient execution plans can lead to **poor performance**. Traditional query optimizers rely on **static cost models**, which may not always produce the best execution strategies.  

This project leverages **Reinforcement Learning (RL)** to dynamically optimize SQL queries based on real-time feedback. The RL agent interacts with **PostgreSQL** and learns to adjust query planner parameters (e.g., `enable_hashjoin`, `enable_indexscan`, `work_mem`) to **minimize execution costs**.  

We use the **TPC-H benchmark dataset** for training and evaluation.  

## **Features**  

✅ **Custom Reinforcement Learning Environment** – Built using OpenAI Gym for query optimization.  
✅ **Q-Learning Agent** – Implements a Q-learning algorithm to optimize query execution plans.  
✅ **PostgreSQL Integration** – Connects to a PostgreSQL database to execute and analyze queries.  
✅ **Dynamic Query Tuning** – Adjusts planner parameters in real-time to optimize performance.  
✅ **Training & Evaluation Scripts** – Includes scripts to train the RL agent and evaluate performance.  

## **Installation**  

Follow these steps to set up the project:  

### **1. Clone the repository**  
```bash
git clone https://github.com/your-username/query-optimization-rl.git
cd query-optimization-rl
```  

### **2. Set up PostgreSQL & Load the Dataset**  
- Install **PostgreSQL** and create a database (e.g., `tpch_db`).  
- Load the **TPC-H dataset** using the provided script:  
  ```bash
  python scripts/load_tpch.py
  ```  

### **3. Install dependencies**  
```bash
pip install -r requirements.txt
```  

## **Usage**  

### **Training the RL Agent**  
Run the following command to train the RL agent:  
```bash
python src/train_rl_agent.py
```  
The agent will interact with PostgreSQL, adjust query planner settings, and save the trained **Q-table**.  

### **Evaluating the Trained Agent**  
To test the agent on new queries, run:  
```bash
python src/test_environment.py
```  

### **Custom Queries**  
Modify `src/test_environment.py` to evaluate the agent on different workloads.  

## **Results**  

### **Training Progress**  
The RL agent learns to reduce query execution costs over multiple episodes:  

📈 

<img width="642" alt="Screenshot 2025-03-08 at 10 57 51 AM" src="https://github.com/user-attachments/assets/a589769a-5596-4002-8a89-752cc7669280" />


### **Performance Metrics**  
- **Best Reward Achieved:** `236.54`  
- **Average Cost Reduction:** `15%`  

## **Future Work**  

🚀 **Deep Reinforcement Learning** – Replace Q-learning with **DQN** or **Policy Gradient** methods.  
📊 **State Space Refinement** – Incorporate more features (e.g., query complexity, database stats).  
🔄 **Multi-Query Optimization** – Extend optimization to handle **multiple queries simultaneously**.  
🏢 **Real-World Databases** – Test the RL agent on production workloads.  

## **Contributing**  

We welcome contributions! To contribute:  

1. **Fork** the repository.  
2. Create a **new branch** (`git checkout -b feature/YourFeature`).  
3. **Commit** your changes (`git commit -m 'Add new feature'`).  
4. **Push** to your branch (`git push origin feature/YourFeature`).  
5. Open a **Pull Request**.  

## **License**  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

## **Acknowledgments**  

🙏 **Special thanks to:**  
- **OpenAI Gym** – RL environment framework.  
- **TPC-H Benchmark** – Dataset provider.  
- **PostgreSQL** – Database backend.  

## **Contact**  

📌 **Name:** Rishikesh Thakker  
📧 **Email:** [thakker834@gmail.com](mailto:thakker834@gmail.com)  
🔗 **LinkedIn:** [linkedin.com/in/rishikesh-thakker](www.linkedin.com/in/rishikesh-thakker-318078275)  
