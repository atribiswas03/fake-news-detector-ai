# fake_news_pipeline.py

import os
import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

def load_and_prepare_data():
    data_frames = []

    # 1. Load True.csv (real news)
    if os.path.exists("True.csv"):
        true_df = pd.read_csv("True.csv")
        true_df['label'] = 1
        data_frames.append(true_df)
    else:
        print("⚠️ True.csv not found.")

    # 2. Load Fake.csv (fake news)
    if os.path.exists("Fake.csv"):
        fake_df = pd.read_csv("Fake.csv")
        fake_df['label'] = 0
        data_frames.append(fake_df)
    else:
        print("⚠️ Fake.csv not found.")

    # 3. Load news_dataset.csv (user feedback)
    feedback_path = "news_dataset.csv"
    if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
        feedback_df = pd.read_csv(feedback_path)
        if 'text' in feedback_df.columns and 'label' in feedback_df.columns:
            feedback_df = feedback_df[['text', 'label']]
            data_frames.append(feedback_df)
        else:
            print("⚠️ news_dataset.csv missing 'text' or 'label' columns.")
    else:
        print("⚠️ news_dataset.csv is empty or missing.")

    if not data_frames:
        raise ValueError("❌ No valid datasets found. Cannot train.")

    # Combine all data
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Combine title+text if both present (for True/Fake)
    if 'title' in combined_df.columns and 'text' in combined_df.columns:
        combined_df['text'] = combined_df['title'].astype(str) + " " + combined_df['text'].astype(str)

    if 'text' not in combined_df.columns or 'label' not in combined_df.columns:
        raise ValueError("❌ Required columns ('text', 'label') missing after merge.")

    # Clean text
    combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)

    # Shuffle and return
    combined_df = combined_df[['text', 'label']].sample(frac=1).reset_index(drop=True)
    return combined_df['text'], combined_df['label']


def train_model(X, y, show_metrics=True):
    """
    Trains and saves the logistic regression model and TF-IDF vectorizer
    using provided features (X) and labels (y).
    """
    # Check class balance
    if len(set(y)) < 2:
        print("❌ Cannot train model: only one class present in data:", set(y))
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        ngram_range=(1, 2),
        min_df=1
    )

    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(C=1.5, max_iter=1000, solver='liblinear')
    model.fit(X_tfidf, y)

    if show_metrics:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        if len(set(y_train)) < 2:
            print("⚠️ Skipping evaluation: train set only has one class.")
        else:
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            print("✅ Model Retrained on Approved Data")
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
            print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save updated model and vectorizer
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model and vectorizer saved.")

    return model, vectorizer


def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = load(MODEL_PATH)
        vectorizer = load(VECTORIZER_PATH)
        print("\u2705 Model and vectorizer loaded from disk.")
        return model, vectorizer
    else:
        print("\u26a0\ufe0f Model files not found. Please train the model first.")
        return None, None

def predict_news(news, model, vectorizer, threshold=0.55):
    news_clean = clean_text(news)
    news_vec = vectorizer.transform([news_clean])
    probs = model.predict_proba(news_vec)[0]

    label = "Real" if probs[1] >= threshold else "Fake"
    predicted_prob = probs[1] if label == "Real" else probs[0]
    confidence = round(predicted_prob * 100, 2)

    return label, confidence

def main():
    print("\n\ud83d\udce6 Loading and preparing data...")
    X, y = load_and_prepare_data()

    print("\n\ud83d\udd27 Training model...")
    model, vectorizer = train_model()

    while True:
        user_input = input("\n\ud83d\udcf0 Enter news text (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("\ud83d\udc4b Exiting.")
            break

        label, confidence = predict_news(user_input, model, vectorizer)
        print(f"\u2705 Prediction: The news is **{label}** with {confidence}% confidence.")

if __name__ == "__main__":
    main()
