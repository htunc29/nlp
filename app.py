from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import os

app = Flask(__name__)

# --- VERİLERİ YÜKLE ---
df_lemma = pd.read_csv("lemmatized.csv")
df_stem = pd.read_csv("stemmed.csv")
text_col = "icerik" if "icerik" in df_lemma.columns else df_lemma.columns[0]
lemma_texts = df_lemma[text_col].astype(str).tolist()
stem_texts = df_stem[text_col].astype(str).tolist()

# --- TF-IDF Fonksiyonu ---
def tfidf_top5(input_text, texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    input_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vec, tfidf_matrix)[0]
    top5_idx = similarities.argsort()[-6:][::-1][1:]
    return [(texts[idx], similarities[idx], idx) for idx in top5_idx]

# --- Word2Vec Fonksiyonu ---
def get_sentence_vector(model, sentence):
    words = word_tokenize(sentence)
    vectors = [model.wv[w] for w in words if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def w2v_top5(input_text, texts, model_path):
    model = Word2Vec.load(model_path)
    input_vec = get_sentence_vector(model, input_text).reshape(1, -1)
    sentence_vecs = np.array([get_sentence_vector(model, s) for s in texts])
    similarities = cosine_similarity(input_vec, sentence_vecs)[0]
    top5_idx = similarities.argsort()[-5:][::-1]
    return [(texts[idx], similarities[idx], idx) for idx in top5_idx]

# --- Jaccard Fonksiyonu ---
def jaccard(set1, set2):
    try:
        set1 = set(set1.tolist() if hasattr(set1, 'tolist') else set1)
        set2 = set(set2.tolist() if hasattr(set2, 'tolist') else set2)
    except Exception as e:
        print("Sete dönüştürürken hata:", e)
        return 0.0

    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# --- Ana Sayfa ve Sonuçlar ---
@app.route("/", methods=["GET", "POST"])
def index():
    results_lemma = []
    results_stem = []
    w2v_results = {}
    input_text = ""
    all_top5_sets = []
    model_names = []
    model_paths = [
        ("TF-IDF Lemmatized", None),
        ("TF-IDF Stemmed", None),
        ("Lemmatized CBOW Win2 Dim100", "models/lemmatized_model_cbow_window2_dim100.model"),
        ("Lemmatized CBOW Win2 Dim300", "models/lemmatized_model_cbow_window2_dim300.model"),
        ("Lemmatized CBOW Win4 Dim100", "models/lemmatized_model_cbow_window4_dim100.model"),
        ("Lemmatized CBOW Win4 Dim300", "models/lemmatized_model_cbow_window4_dim300.model"),
        ("Lemmatized Skipgram Win2 Dim100", "models/lemmatized_model_skipgram_window2_dim100.model"),
        ("Lemmatized Skipgram Win2 Dim300", "models/lemmatized_model_skipgram_window2_dim300.model"),
        ("Lemmatized Skipgram Win4 Dim100", "models/lemmatized_model_skipgram_window4_dim100.model"),
        ("Lemmatized Skipgram Win4 Dim300", "models/lemmatized_model_skipgram_window4_dim300.model"),
        ("Stemmed CBOW Win2 Dim100", "models/stemmed_model_cbow_window2_dim100.model"),
        ("Stemmed CBOW Win2 Dim300", "models/stemmed_model_cbow_window2_dim300.model"),
        ("Stemmed CBOW Win4 Dim100", "models/stemmed_model_cbow_window4_dim100.model"),
        ("Stemmed CBOW Win4 Dim300", "models/stemmed_model_cbow_window4_dim300.model"),
        ("Stemmed Skipgram Win2 Dim100", "models/stemmed_model_skipgram_window2_dim100.model"),
        ("Stemmed Skipgram Win2 Dim300", "models/stemmed_model_skipgram_window2_dim300.model"),
        ("Stemmed Skipgram Win4 Dim100", "models/stemmed_model_skipgram_window4_dim100.model"),
        ("Stemmed Skipgram Win4 Dim300", "models/stemmed_model_skipgram_window4_dim300.model"),
    ]

    if request.method == "POST":
        input_text = request.form["input_text"]
        # TF-IDF
        results_lemma = tfidf_top5(input_text, lemma_texts)
        results_stem = tfidf_top5(input_text, stem_texts)
        all_top5_sets.append(set(idx for _, _, idx in results_lemma))
        model_names.append("TF-IDF Lemmatized")
        all_top5_sets.append(set(idx for _, _, idx in results_stem))
        model_names.append("TF-IDF Stemmed")
        # Word2Vec - tüm modeller için sonuçları hesapla
        for model_name, model_path in model_paths[2:]:
            if os.path.exists(model_path):
                if "lemmatized" in model_path.lower():
                    results = w2v_top5(input_text, lemma_texts, model_path)
                else:
                    results = w2v_top5(input_text, stem_texts, model_path)
                w2v_results[model_name] = results
                all_top5_sets.append(set(idx for _, _, idx in results))
                model_names.append(model_name)
            else:
                w2v_results[model_name] = []
                all_top5_sets.append(set())
                model_names.append(model_name)
        # Jaccard matrisi
        n = len(all_top5_sets)
        jaccard_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jaccard_matrix[i, j] = jaccard(all_top5_sets[i], all_top5_sets[j])
    else:
        jaccard_matrix = None
        model_names = []

    return render_template("index.html",
                         input_text=input_text,
                         results_lemma=results_lemma,
                         results_stem=results_stem,
                         w2v_results=w2v_results,
                         jaccard_matrix=jaccard_matrix,
                         model_names=model_names)

if __name__ == "__main__":
    app.run(debug=True)