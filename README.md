# Sentiment Analysis on Amazon Product Reviews

A complete NLP pipeline that classifies Amazon product reviews as **Positive** or **Negative** using machine learning — with a live interactive demo built in Streamlit.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-analysis-amazon-appuct-reviews.streamlit.app/)

---

## Live Demo

🚀 **[https://sentiment-analysis-amazon-appuct-reviews.streamlit.app/](https://sentiment-analysis-amazon-appuct-reviews.streamlit.app/)**

Type any product review and get an instant sentiment prediction with confidence score.

---

## Project Overview

| | |
|---|---|
| **Task** | Binary sentiment classification (Positive / Negative) |
| **Dataset** | Amazon Product Reviews (`reviewText`, `Positive`) |
| **Best Model** | LinearSVC + TF-IDF |
| **Test Accuracy** | ~93% |
| **Test F1 Score** | ~95% |

---

## Repository Structure

```
├── Sentiment Analysis on Amazon Product Reviews.ipynb   # Full project notebook
├── app.py                                               # Streamlit demo app
└── requirements.txt                                     # Python dependencies
```

---

## Notebook Walkthrough

### 1. Dataset Overview
- Binary sentiment labels: `1` = Positive, `0` = Negative
- Source: Amazon product reviews dataset

### 2. Data Preprocessing
- Handle missing values
- Lowercase, remove URLs / HTML / punctuation / digits
- Tokenize, remove stop words, lemmatize

### 3. Model Selection
Four models compared:
- Logistic Regression
- Multinomial Naive Bayes
- SVM (LinearSVC)
- Random Forest

### 4. Model Training
- **Vectorization:** TF-IDF with unigrams + bigrams (`ngram_range=(1,2)`, `max_features=15000`)
- All models trained on an 80/20 stratified split

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix for each model

### 6. Hyperparameter Tuning
- **Grid Search** on Logistic Regression (`C`, `penalty`)
- **Randomized Search** on LinearSVC (`C`, `max_iter`)

### 7. Comparative Analysis
- Performance comparison table and bar charts
- Strengths & weaknesses of each model

### 8. Conclusion
- SVM and Logistic Regression outperform tree-based models on sparse TF-IDF features
- Bigrams significantly improve sentiment capture
- Future work: LSTM/GRU with GloVe embeddings, BERT fine-tuning

---

## Run Locally

```bash
git clone https://github.com/clindoe/sentiment-analysis-amazon-product-reviews.git
cd sentiment-analysis-amazon-product-reviews
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
