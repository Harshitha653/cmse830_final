# model.py
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
tweets_path = "data/Tweets.csv"
df = pd.read_csv(tweets_path)

# Keep only relevant columns
df = df[['text', 'airline_sentiment']].dropna()

# Map sentiment to numeric
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(sentiment_map)

# 2. Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# 3. Train-test split
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Define models
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
nb_clf = MultinomialNB()

# 6. Ensemble Voting Classifier
ensemble = VotingClassifier(
    estimators=[('rf', rf_clf), ('nb', nb_clf)],
    voting='soft'
)

# 7. Train ensemble
ensemble.fit(X_train_tfidf, y_train)

# 8. Evaluate
y_pred = ensemble.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save model and vectorizer
with open("ensemble_model.pkl", "wb") as f:
    pickle.dump({'model': ensemble, 'vectorizer': tfidf}, f)

print("Ensemble model saved as ensemble_model.pkl")
