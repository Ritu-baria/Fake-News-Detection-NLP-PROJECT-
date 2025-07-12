# Fake-News-Detection-NLP-PROJECT-# 📰 Fake News Detection using Logistic Regression and NLP

Detecting fake news using Natural Language Processing (NLP) and Machine Learning. This project uses a dataset of real and fake news articles to train a classification model that distinguishes between the two.

---

## 📌 Project Overview

With the rise of misinformation, detecting fake news has become a critical problem. In this project, we use logistic regression with text vectorization to classify news articles as **real** or **fake**.

---

## 📂 Dataset

We use the [Fake and Real News Dataset](https://www.kaggle.com/datasets/mrisdal/fake-news) from Kaggle. It contains:

- **True.csv**: Real news articles
- **Fake.csv**: Fake news articles

Each file has:
- `title`: Title of the news
- `text`: Full article content
- `subject`, `date` (not used)
  
We add a `label` column (1 = real, 0 = fake) to train the model.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- NLTK (stopwords)
- Scikit-learn (CountVectorizer, Logistic Regression)

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
