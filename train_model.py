import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset/ai_vs_human_text.csv")

# Keep only needed columns
df = df[["text", "label"]]

# Convert labels
df["label"] = df["label"].map({
    "AI-generated": 1,
    "Human-written": 0
})

# Drop nulls & duplicates
df = df.dropna().drop_duplicates()

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()

    # REMOVE LABEL LEAKAGE (IMPORTANT)
    text = text.replace("ai-generated content sample", "")
    text = text.replace("human-written text sample", "")

    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Keep only alphabets
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["text"] = df["text"].apply(clean_text)

# ---------- SPLIT ----------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- MODELS ----------
# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_vec, y_train)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# ---------- EVALUATION ----------
y_pred = lr_model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# ---------- SAVE ----------
pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(nb_model, open("nb_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(" Training complete. Models saved.")