# 📱 SMS Spam Detection using Machine Learning & NLP

> A complete NLP pipeline that classifies SMS messages as **Spam** or **Ham (Not Spam)** using text preprocessing, feature engineering, and a comparative evaluation of 11 machine learning classifiers — with **Multinomial Naïve Bayes** selected as the final model for its high precision.

---

## 📌 Project Overview

- Built a binary SMS spam classifier using the classic **spam.csv** dataset
- Covered the full ML workflow: data cleaning → EDA → text preprocessing → feature extraction → model comparison → deployment
- Evaluated **11 classifiers** and selected the best based on **Accuracy** and **Precision**

---

## 📂 Dataset

- **Source:** `spam.csv` (UCI SMS Spam Collection)
- **Total Records:** 5,572 messages (before cleaning)
- **After Deduplication:** 5,169 messages (403 duplicates removed)
- **Class Distribution:**
  - Ham (0): 4,516 messages — **87.37%**
  - Spam (1): 653 messages — **12.63%**
- **Label Encoding:** `ham → 0` | `spam → 1`
- **Preprocessing:** Removed 3 unnamed/redundant columns, renamed columns to `target` and `text`

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized class distribution using a **pie chart** — confirmed dataset is **heavily imbalanced** (87.37% ham vs 12.63% spam)
- Engineered three statistical features per message:
  - `num_characters` — total character count
  - `num_words` — total word count (via NLTK tokenizer)
  - `num_sentences` — total sentence count
- Compared feature distributions between spam and ham using **histograms** and **pair plots**
- Analyzed feature correlations using a **heatmap**

**Feature Statistics — Overall:**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| num_characters | 78.98 | 58.24 | 2 | 910 |
| num_words | 18.46 | 13.32 | 1 | 220 |
| num_sentences | 1.97 | 1.45 | 1 | 38 |

**Key Insight — Ham vs Spam comparison:**

| Feature | Ham (avg) | Spam (avg) |
|---------|-----------|------------|
| num_characters | 70.46 | 137.89 |
| num_words | 17.12 | 27.67 |
| num_sentences | 1.82 | 2.97 |

> Spam messages are consistently **longer**, with more words and sentences than ham messages.

---

## 🔧 Text Preprocessing

A custom `transform_text()` function was built to apply the full NLP pipeline:

- Converted all text to **lowercase**
- **Tokenized** text using NLTK word tokenizer
- Removed **special characters** — kept only alphanumeric tokens
- Removed **stop words** and **punctuation** (using NLTK stopwords + Python `string` library)
- Applied **Porter Stemming** to reduce words to their root form

---

## 🔍 Exploratory Text Analysis (Post-Preprocessing)

- Generated **Word Clouds** separately for spam and ham messages to visualize dominant terms
- Plotted **top 30 most frequent words** in spam and ham corpora using bar plots
- **Spam corpus size:** 9,939 words | **Ham corpus size:** 35,404 words
- Spam corpus keywords: `free`, `call`, `txt`, `claim`, `win`, `prize`, etc.
- Ham corpus keywords: conversational, everyday language

---

## ⚙️ Feature Extraction

- Extracted features using **TF-IDF Vectorizer** with `max_features=3000`
- Also used **CountVectorizer (Bag of Words)** — produced a vocabulary of **6,708 features** on the cleaned corpus
- **Train/Test Split:** 80% training | 20% testing (`random_state=2`)

---

## 🏗️ Models Evaluated

11 classifiers were trained and compared:

| # | Model | Notes |
|---|-------|-------|
| 1 | **Gaussian Naïve Bayes** | Baseline NB variant |
| 2 | **Multinomial Naïve Bayes** | Best balance of accuracy & precision |
| 3 | **Bernoulli Naïve Bayes** | Binary feature NB variant |
| 4 | **Support Vector Classifier (SVC)** | Sigmoid kernel, gamma=1.0 |
| 5 | **K-Nearest Neighbors** | Distance-based classifier |
| 6 | **Decision Tree** | max_depth=5 |
| 7 | **Random Forest** | 50 estimators |
| 8 | **AdaBoost** | 50 estimators |
| 9 | **Bagging Classifier** | 50 estimators |
| 10 | **Extra Trees Classifier** | 50 estimators |
| 11 | **XGBoost** | 50 estimators |

---

## 📈 Model Performance Comparison

> Models sorted by **Precision** (descending) — the primary metric to minimize false spam flags on legitimate messages.

| Model | Accuracy | Precision |
|-------|:--------:|:---------:|
| K-Nearest Neighbors | 90.62% | **100.00%** |
| Logistic Regression | 97.20% | **100.00%** |
| Random Forest | 97.20% | **100.00%** |
| Extra Trees Classifier ⭐ | **97.68%** | 99.14% |
| XGBoost | 97.39% | 97.44% |
| Bernoulli Naïve Bayes | 97.00% | 97.35% |
| AdaBoost | 96.32% | 94.64% |
| Gradient Boosting | 94.39% | 94.44% |
| Decision Tree | 92.65% | 94.29% |
| Bagging Classifier | 96.22% | 91.60% |
| Multinomial Naïve Bayes | 96.42% | 83.44% |
| Gaussian Naïve Bayes | 88.01% | 53.15% |
| SVC (Sigmoid) | 92.65% | 74.22% |

**Naïve Bayes variants (CountVectorizer baseline):**

| Model | Accuracy | Precision | Confusion Matrix |
|-------|:--------:|:---------:|-----------------|
| Gaussian NB | 88.01% | 53.15% | TP=118, FP=104, TN=792, FN=20 |
| Multinomial NB | 96.42% | 83.44% | TP=126, FP=25, TN=871, FN=12 |
| Bernoulli NB | 97.00% | 97.35% | TP=110, FP=3, TN=893, FN=28 |

> Precision was prioritized as the key metric — a legitimate message marked as spam is a worse outcome than a spam message slipping through.

---

## ✅ Final Model: Multinomial Naïve Bayes (Deployed)

- Selected for deployment due to its **speed, simplicity, and suitability** for text classification pipelines
- Achieved **96.42% accuracy** and **83.44% precision** with CountVectorizer
- While KNN, LR, RF, and ETC had higher precision, MNB was chosen for its lightweight inference and ease of integration
- Exported using **Pickle** for deployment:
  - `vectorizer.pkl` — saved TF-IDF vectorizer
  - `model.pkl` — saved trained MNB model

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, NLTK, Scikit-learn, XGBoost, Matplotlib, Seaborn, WordCloud, Pickle
- **NLP Tools:** NLTK (tokenization, stopwords, stemming), TF-IDF Vectorizer
- **ML Models:** Scikit-learn, XGBoost
