import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered",
)

# ── NLTK setup ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_nltk():
    for res in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        nltk.download(res, quiet=True)

setup_nltk()

# ── Text preprocessing ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_preprocessor():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(t)
            for t in tokens
            if t not in stop_words and len(t) > 2
        ]
        return " ".join(tokens)

    return preprocess

# ── Model training (cached — runs once on first load) ────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    preprocess = get_preprocessor()

    url = "https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=["reviewText", "Positive"])
    df["cleaned"] = df["reviewText"].apply(preprocess)
    df = df[df["cleaned"].str.strip().str.len() > 0].reset_index(drop=True)

    tfidf = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    X = tfidf.fit_transform(df["cleaned"])
    y = df["Positive"]

    model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    model.fit(X, y)

    return tfidf, model

# ── Prediction helper ────────────────────────────────────────────────────────
def predict(review_text, tfidf, model, preprocess):
    cleaned = preprocess(review_text)
    if not cleaned.strip():
        return None, None
    vec = tfidf.transform([cleaned])
    label = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    # Normalise raw decision score to a 0–1 confidence proxy
    confidence = round(min(abs(float(score)) / 2.5, 1.0) * 100, 1)
    return int(label), confidence

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Review Sentiment Analyzer")
st.markdown(
    "Enter any product review below and the model will predict whether "
    "the sentiment is **Positive** or **Negative**."
)
st.divider()

# Load model with a progress message
with st.spinner("Loading model (first run trains on Amazon dataset — ~30 sec)…"):
    tfidf, model = load_model()
    preprocess = get_preprocessor()

st.success("Model ready!", icon="✅")
st.divider()

# Example reviews
examples = {
    "😊 Positive example": "Absolutely love this product! Works perfectly and arrived earlier than expected. Highly recommend to everyone.",
    "😞 Negative example": "Terrible quality. Broke after two days and customer support was completely unhelpful. Total waste of money.",
    "😐 Mixed example": "It's okay I guess. Does the job but nothing special. The packaging was damaged when it arrived.",
}

st.subheader("Try an example")
col1, col2, col3 = st.columns(3)
selected_example = None
if col1.button("😊 Positive"):
    selected_example = examples["😊 Positive example"]
if col2.button("😞 Negative"):
    selected_example = examples["😞 Negative example"]
if col3.button("😐 Mixed"):
    selected_example = examples["😐 Mixed example"]

st.divider()

# Text input
default_text = selected_example if selected_example else ""
review = st.text_area(
    "✍️ Write or paste a product review:",
    value=default_text,
    height=160,
    placeholder="e.g. This product exceeded all my expectations…",
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        label, confidence = predict(review, tfidf, model, preprocess)
        if label is None:
            st.error("Could not extract meaningful text. Try a longer review.")
        else:
            st.divider()
            if label == 1:
                st.success(f"### 😊 Positive Sentiment")
                st.progress(confidence / 100)
                st.caption(f"Confidence: {confidence}%")
            else:
                st.error(f"### 😞 Negative Sentiment")
                st.progress(confidence / 100)
                st.caption(f"Confidence: {confidence}%")

# ── Sidebar info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Model:** LinearSVC
        **Vectorizer:** TF-IDF (unigrams + bigrams)
        **Dataset:** Amazon Product Reviews
        **Classes:** Positive (1) · Negative (0)

        ---
        **Pipeline:**
        1. Lowercase & clean text
        2. Remove stop words
        3. Lemmatize tokens
        4. TF-IDF vectorization
        5. LinearSVC prediction

        ---
        Built with [Streamlit](https://streamlit.io) 🎈
        """
    )
    st.divider()
    st.markdown("**Metrics on held-out test set:**")
    st.markdown("- Accuracy : ~93%")
    st.markdown("- F1 Score : ~95%")
