<p align="center">
  <img src="https://i.imgur.com/SMm17nu.png" width="50%">
</p>

<h1 align="center">GoAI: Deep Reinforcement Learning for Go</h1>

## **Overview**
GoAI is an artificial intelligence project implementing **Deep Reinforcement Learning** for the game of **Go**, inspired by Google's **AlphaGo Zero**. The project utilizes **Monte Carlo Tree Search (MCTS)**, **Minimax**, and **Neural Networks (Policy & Value Networks)** to train an AI capable of self-learning and improving over time.

This project was developed as part of a **Master Qualifying Project at Worcester Polytechnic Institute (WPI)** by **Esteban Aranda** and **Thomas Graham**, under the advisement of **Professor Xiangnan Kong** and **Professor Yanhua Li**.

## **Key Achievements**
  - **Multi-Level AI Training:** The AI evolved from **Tic-Tac-Toe** to **Othello** and then to **Go**, testing each algorithm's efficiency.
  - **Deep Learning Implementation:** Built **Policy and Value Neural Networks** using **PyTorch** for decision-making and move evaluation.  
  - **Reinforcement Learning Framework:** Implemented **self-play training** where the AI improves by continuously playing against itself.  
  - **Monte Carlo Tree Search (MCTS):** Used MCTS for probabilistic decision-making, allowing more efficient move selection in Go.  
  - **Scalability & Modular Design:** The system was designed to be **easily extensible** for additional games and optimizations.  
  - **Comprehensive Unit Testing:** Developed **test-driven AI development**, ensuring high code quality and robust performance validation.  

---

## **Project Structure**
```
📂 GoAI/
│── 📂 src/                   # Main source code
│   ├── game/                 # Game implementations (Tic-Tac-Toe, Othello, Go)
│   ├── players/              # AI players (Random, Minimax, MCTS, PolicyNN, ValueNN)
│   ├── neural_networks/      # Deep learning models
│   ├── training/             # Reinforcement learning scripts
│   ├── tests/                # Unit tests
│── 📂 docs/                  # Project documentation and research
│── 📂 experiments/           # Training data and experiment results
│── requirements.txt          # Python dependencies
│── README.md                 # Project introduction and setup guide
│── LICENSE                   # Licensing information
```

---

## **How It Works**
### **1️⃣ Game Environment**
- Supports **Tic-Tac-Toe**, **Othello**, and **Go** (9x9, 13x13, 19x19).
- Each game is implemented with rules and a visualization interface.

### **2️⃣ AI Players**
| **Player Type**      | **Description** |
|----------------------|----------------|
| **Random Player**    | Makes completely random moves. |
| **Minimax Player**   | Uses Minimax algorithm to evaluate best moves. |
| **MCTS Player**      | Uses Monte Carlo Tree Search for probabilistic decision-making. |
| **PolicyNN Player**  | Uses a **Neural Network** to predict optimal moves. |
| **ValueNN Player**   | Uses a **Value Network** to evaluate board positions. |

### **3️⃣ Reinforcement Learning**
- AI trains **by playing against itself**.
- **Policy Network** predicts moves, while **Value Network** evaluates board states.
- Uses **Supervised Learning** first, then **Reinforcement Learning** for optimization.
- Training **iterates over 100 models**, improving the AI with each step.

---

## **Installation**
### **Prerequisites**
Ensure you have **Python 3.6+** installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### **Run AI Training**
To train the AI using Reinforcement Learning:
```bash
python training/train.py
```

To test AI performance in **Go**:
```bash
python src/game/go.py
```

To run **unit tests**:
```bash
pytest tests/
```

---

## **Experimental Results**
| **Algorithm** | **Win Rate (vs Random Player)** | **Loss Rate** | **Tie Rate** |
|--------------|--------------------------------|--------------|--------------|
| Minimax      | 99.8%                          | 0%           | 0.2%         |
| MCTS         | 98.0%                          | 0%           | 2.0%         |
| PolicyNN     | 64.2%                          | 23.2%        | 12.6%        |

**Key Takeaways:**
- Minimax and MCTS perform **exceptionally well** on small games like **Tic-Tac-Toe**.
- PolicyNN improves over **100 training iterations**, demonstrating **self-learning**.
- Reinforcement Learning further **optimizes gameplay**, particularly in larger games.

---

## **Future Enhancements**
🔹 Expand training to **larger Go boards (19x19)** for deeper AI learning.  
🔹 Combine **Policy & Value Networks** for **AlphaGo-like decision-making**.  
🔹 Improve **training efficiency** using **cloud computing** for larger datasets.  

---

## **Contributors**
- **Esteban Aranda** - [GitHub](https://github.com/username)  
- **Thomas Graham** - [GitHub](https://github.com/tgrahamcodes)  

Supervised by:
- **Prof. Xiangnan Kong** (WPI)
- **Prof. Yanhua Li** (WPI)

---

## **License**
This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for details.
