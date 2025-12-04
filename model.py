import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset (local training only; not used on Streamlit Cloud)
tweets_path = "data/Tweets.csv"
df = pd.read_csv(tweets_path)

df = df[["text", "airline_sentiment"]].dropna()

sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(sentiment_map)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"[^a-z\s]", "", text)  # punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
nb_clf = MultinomialNB()

ensemble = VotingClassifier(
    estimators=[("rf", rf_clf), ("nb", nb_clf)], voting="soft"
)

ensemble.fit(X_train_tfidf, y_train)

y_pred = ensemble.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("ensemble_model.pkl", "wb") as f:
    pickle.dump({"model": ensemble, "vectorizer": tfidf}, f)

print("Saved ensemble_model.pkl")
