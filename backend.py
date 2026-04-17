import pickle
import re

# LOAD MODELS
lr_model = pickle.load(open("model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# CLEAN TEXT
def clean_text(text):
    text = text.lower()

    text = text.replace("ai-generated content sample", "")
    text = text.replace("human-written text sample", "")

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# PREDICT
def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    lr_pred = lr_model.predict(vec)[0]
    lr_prob = lr_model.predict_proba(vec)[0]
    lr_conf = max(lr_prob)

    nb_pred = nb_model.predict(vec)[0]
    nb_prob = nb_model.predict_proba(vec)[0]
    nb_conf = max(nb_prob)

    return (lr_pred, lr_conf), (nb_pred, nb_conf)

# TOP WORDS
def get_top_words(n=10):
    feature_names = vectorizer.get_feature_names_out()
    coef = lr_model.coef_[0]

    top_ai = sorted(zip(coef, feature_names), reverse=True)[:n]
    top_human = sorted(zip(coef, feature_names))[:n]

    return top_ai, top_human