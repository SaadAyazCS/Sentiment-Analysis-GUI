# Sentiment Analysis GUI Project

## 📌 Project Overview
This is my project on **Sentiment Analysis**, built with a graphical user interface (GUI) using Python's **Tkinter**. It combines machine learning with heuristic-based sentiment scoring to analyze text inputs or CSV files.

---

## ⚙️ Core Functionalities
- 📁 Upload a CSV dataset with labeled sentiment data (text + label columns)
- 🧠 Automatically detects relevant columns (text and sentiment)
- 📊 Trains a sentiment classification model using **TF-IDF** and **LinearSVC**
- 📈 Shows **accuracy score**, **classification report**, and charts:
  - Confusion matrix
  - Accuracy comparison bar chart
- 🔎 Predict sentiment for:
  - Typed user input
  - Bulk prediction from a CSV file
- 💾 Saves output results and charts to the **Desktop**
- 🧹 Handles invalid input (e.g. empty, numeric-only)

---

## 🚀 How to Use
1. Click **"Upload Dataset"** to select a CSV file  
   → Must have one column for text and one for sentiment labels.
2. Click **"Train Model"**  
   → It fits the classifier and saves results/charts to the desktop.
3. Type a sentence and click **"Predict Sentiment"** for a quick test.
4. Use **"Predict File"** to upload a CSV for bulk prediction.
5. Use **"Reset"** to clear the interface and start over.

---

## 🧪 Libraries Used
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `vaderSentiment`
- `tkinter` (for GUI)
- `PIL` (for image handling)

---

## 💡 Suggestions for Improvement
- Add progress bar during training and prediction
- Include model saving/loading to avoid retraining every time
- Experiment with advanced models or ensembles (e.g., VotingClassifier)
- Use transformer models (e.g., BERT) for better accuracy
- Extend support for non-English languages

---
