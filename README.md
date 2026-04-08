# Naive Bayes — Google Play Store Sentiment Analysis

## Overview
This project builds a **sentiment classifier** for Google Play Store reviews using Naive Bayes, with comparisons to Random Forest, Logistic Regression, and Gradient Boosting.

## Dataset
- **Source:** [4GeeksAcademy Naive Bayes Tutorial](https://github.com/4GeeksAcademy/naive-bayes-project-tutorial)
- **891 reviews** from Google Play Store applications
- Variables: `package_name` (dropped), `review` (text), `polarity` (0=negative, 1=positive)
- Class distribution: 584 negative (65.5%), 307 positive (34.5%)

## Methodology
1. Text preprocessing: strip whitespace, lowercase, remove `package_name`
2. CountVectorizer with English stop words for word count matrix
3. Train all three Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB
4. Hyperparameter tuning: alpha smoothing, TF-IDF vs Count, ngram range
5. GridSearchCV with Pipeline (vectorizer + classifier)
6. Random Forest optimization for comparison
7. Alternative models: Logistic Regression, Gradient Boosting

## Results
| Model | Test Acc | Test F1 | AUC-ROC |
|-------|----------|---------|---------|
| GaussianNB | 0.8156 | 0.7179 | 0.7832 |
| **MultinomialNB (default)** | **0.8547** | **0.7593** | 0.8987 |
| BernoulliNB | 0.7821 | 0.5806 | 0.8973 |
| MultinomialNB (optimized) | 0.7374 | 0.4198 | 0.9145 |
| Random Forest | 0.7598 | 0.5275 | 0.9024 |
| Logistic Regression | 0.8324 | 0.7273 | **0.9217** |
| Gradient Boosting | 0.7542 | 0.5849 | 0.8174 |

## Key Findings
- **MultinomialNB** is the best Naive Bayes for word counts — achieves highest accuracy (0.8547) and F1 (0.7593)
- **Logistic Regression** achieves the highest AUC-ROC (0.9217), outperforming all NB variants
- GaussianNB is not appropriate for sparse word count matrices
- Default MultinomialNB (alpha=1.0) with CountVectorizer is a strong, fast baseline
- Small dataset (891 reviews) limits gains from complex ensemble methods

## Project Structure
```
naive_bayes/
├── data/
│   ├── raw/playstore_reviews.csv
│   ├── interim/
│   └── processed/
├── models/naive_bayes_sentiment.pkl
├── src/
│   ├── explore.ipynb
│   └── app.py
├── requirements.txt
└── README.md
```

## Usage
```bash
pip install -r requirements.txt
python src/app.py
```
