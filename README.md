# DecisionTree

Implementation of a binary decision tree classifier from scratch without using any external machine learning libraries.

## 📌 Description

This project implements a binary decision tree classifier using entropy and information gain as the splitting criteria. The algorithm supports deterministic tie-breaking rules and is constrained to a maximum depth of 2. It is designed for educational purposes and handles basic classification tasks with labeled data.

## 🧠 Features

- ID3-style decision tree
- Entropy-based split evaluation
- Information gain optimization
- Deterministic tie-breaking:
  - Lower dimension is preferred
  - If tied, lower split point is chosen
- Maximum depth of 2
- Written in pure Python
- No external dependencies

## 📁 File Structure

- `decision_tree.py` – Main implementation of the classifier
- `Node` class – Represents each node in the tree
- `Solution` class – Fits the tree and performs classification

## ▶️ How to Run

To use the model with a text input:

```bash
python3 your_script.py < input00.txt
