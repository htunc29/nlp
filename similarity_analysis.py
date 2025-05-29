import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os
import csv

w2v_models = [
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

df_lemma = pd.read_csv("lemmatized.csv")
df_stem = pd.read_csv("stemmed.csv")

lemma_sentences = df_lemma["original_sentence"].astype(str).tolist()
stem_sentences = df_stem["original_sentence"].astype(str).tolist()

text_col = "icerik" if "icerik" in df_lemma.columns else "processed_tokens"

input_text = df_lemma[text_col].iloc[0]
print("Giriş metni:\n", input_text)

def tfidf_top5(input_text, texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    input_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vec, tfidf_matrix)[0]
    top5_idx = similarities.argsort()[-6:][::-1][1:]  # ilk sırada kendisi olur
    return [(idx, texts[idx], similarities[idx]) for idx in top5_idx]

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
    return [(idx, texts[idx], similarities[idx]) for idx in top5_idx]  # idx başa eklendi!

results = []


lemma_texts = df_lemma[text_col].astype(str).tolist()
tfidf_lemma = tfidf_top5(input_text, lemma_texts)
results.append(set(idx for idx, _, _ in tfidf_lemma))


stem_texts = df_stem[text_col].astype(str).tolist()
tfidf_stem = tfidf_top5(input_text, stem_texts)
results.append(set(idx for idx, _, _ in tfidf_stem))

# Word2Vec Modelleri
for name, path in w2v_models:
    if not os.path.exists(path):
        print(f"Model bulunamadı: {path}")
        results.append(set())
        continue
    # Hangi veriyle çalışacağına dikkat et!
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_texts, path)
    else:
        top5 = w2v_top5(input_text, stem_texts, path)
    results.append(set(idx for idx, _, _ in top5))
def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

n = len(results)
jaccard_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        jaccard_matrix[i, j] = jaccard(results[i], results[j])

print("Jaccard Benzerlik Matrisi:")
print(jaccard_matrix)

print("\nTF-IDF (Lemmatized) ilk 5:")
for idx, metin, skor in tfidf_lemma:
    print(f"{idx}: {skor:.4f} - {lemma_sentences[idx][:80]}...")

print("\nTF-IDF (Stemmed) ilk 5:")
for idx, metin, skor in tfidf_stem:
    print(f"{idx}: {skor:.4f} - {stem_sentences[idx][:80]}...")

model_names = [
    "TF-IDF (Lemmatized)",
    "TF-IDF (Stemmed)",
    *[name for name, _ in w2v_models]  # Word2Vec modellerinin isimleri
]

# DataFrame oluştur
jaccard_df = pd.DataFrame(
    jaccard_matrix,
    index=model_names,
    columns=model_names
)

# CSV olarak kaydet
jaccard_df.to_csv("jaccard_similarity_matrix.csv", float_format="%.4f")
# Model isimlerini hazırla
model_names = [
    "TF-IDF (Lemmatized)",
    "TF-IDF (Stemmed)",
    *[name for name, _ in w2v_models]
]

# DataFrame oluştur
jaccard_df = pd.DataFrame(
    jaccard_matrix,
    index=model_names,
    columns=model_names
)

# HTML olarak kaydet
html = jaccard_df.style\
    .format("{:.4f}")\
    .background_gradient(cmap='Blues')\
    .set_caption("Jaccard Benzerlik Matrisi")\
    .set_table_styles([{
        'selector': 'caption',
        'props': [
            ('font-size', '16pt'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]
    }])\
    .to_html()

with open("jaccard_similarity_matrix.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Jaccard benzerlik matrisi 'jaccard_similarity_matrix.html' dosyasına kaydedildi.")
print("Jaccard benzerlik matrisi 'jaccard_similarity_matrix.csv' dosyasına kaydedildi.")
for i, (name, path) in enumerate(w2v_models):
    print(f"\nWord2Vec {name} ilk 5:")
    if not os.path.exists(path):
        print("Model bulunamadı.")
        continue
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_texts, path)
        for idx, metin, skor in top5:
            print(f"{idx}: {skor:.4f} - {lemma_sentences[idx][:80]}...")
    else:
        top5 = w2v_top5(input_text, stem_texts, path)
        for idx, metin, skor in top5:
            print(f"{idx}: {skor:.4f} - {stem_sentences[idx][:80]}...")

# --- SONUÇLARI CSV ve HTML OLARAK KAYDET ---
def log_top5_to_csv_and_html(tfidf_lemma, tfidf_stem, w2v_models, lemma_sentences, stem_sentences, all_w2v_top5, input_text):
    # CSV
    with open("benzerlik_top5.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Sıra", "Index", "Skor", "Metin"])
        # TF-IDF Lemmatized
        for i, (idx, _, skor) in enumerate(tfidf_lemma, 1):
            writer.writerow(["TF-IDF (Lemmatized)", i, idx, f"{skor:.4f}", lemma_sentences[idx][:120]])
        # TF-IDF Stemmed
        for i, (idx, _, skor) in enumerate(tfidf_stem, 1):
            writer.writerow(["TF-IDF (Stemmed)", i, idx, f"{skor:.4f}", stem_sentences[idx][:120]])
        # Word2Vec
        for (model_name, _), top5 in zip(w2v_models, all_w2v_top5):
            for i, (idx, _, skor) in enumerate(top5, 1):
                metin = lemma_sentences[idx][:120] if "lemmatized" in model_name.lower() else stem_sentences[idx][:120]
                writer.writerow([model_name, i, idx, f"{skor:.4f}", metin])

    # HTML
    html = '''<html><head><meta charset="UTF-8"><title>Benzerlik Sonuçları</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f7f9fa; color: #222; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 32px; }
    h1 { color: #2d6cdf; text-align: center; margin-bottom: 10px; }
    h2 { color: #444; border-left: 5px solid #2d6cdf; padding-left: 12px; margin-top: 36px; }
    .input-box { background: #eaf1fb; border-left: 5px solid #2d6cdf; padding: 18px 20px; margin-bottom: 30px; border-radius: 8px; font-size: 1.1em; }
    table { border-collapse: collapse; width: 100%; margin: 18px 0 32px 0; }
    th, td { border: 1px solid #b6c6e3; padding: 8px 12px; text-align: left; }
    th { background: #eaf1fb; color: #2d6cdf; font-weight: 600; }
    tr:nth-child(even) { background: #f4f8fc; }
    tr:hover { background: #e0eafc; }
    .model-title { margin-top: 32px; }
    </style></head><body><div class="container">
    '''
    html += f'<h1>Her Model İçin İlk 5 Benzer Metin</h1>'
    html += f'<div class="input-box"><b>Giriş Metni:</b><br>{input_text}</div>'
    # TF-IDF Lemmatized
    html += '<h2>TF-IDF (Lemmatized)</h2><table><tr><th>Sıra</th><th>Index</th><th>Skor</th><th>Metin</th></tr>'
    for i, (idx, _, skor) in enumerate(tfidf_lemma, 1):
        html += f'<tr><td>{i}</td><td>{idx}</td><td>{skor:.4f}</td><td>{lemma_sentences[idx][:120]}</td></tr>'
    html += '</table>'
    # TF-IDF Stemmed
    html += '<h2>TF-IDF (Stemmed)</h2><table><tr><th>Sıra</th><th>Index</th><th>Skor</th><th>Metin</th></tr>'
    for i, (idx, _, skor) in enumerate(tfidf_stem, 1):
        html += f'<tr><td>{i}</td><td>{idx}</td><td>{skor:.4f}</td><td>{stem_sentences[idx][:120]}</td></tr>'
    html += '</table>'
    # Word2Vec
    for (model_name, _), top5 in zip(w2v_models, all_w2v_top5):
        html += f'<h2>{model_name}</h2><table><tr><th>Sıra</th><th>Index</th><th>Skor</th><th>Metin</th></tr>'
        for i, (idx, _, skor) in enumerate(top5, 1):
            metin = lemma_sentences[idx][:120] if "lemmatized" in model_name.lower() else stem_sentences[idx][:120]
            html += f'<tr><td>{i}</td><td>{idx}</td><td>{skor:.4f}</td><td>{metin}</td></tr>'
        html += '</table>'
    html += '</div></body></html>'

    with open("benzerlik_top5.html", "w", encoding="utf-8") as f:
        f.write(html)

# Word2Vec sonuçlarını toplamak için:
all_w2v_top5 = []
for name, path in w2v_models:
    if not os.path.exists(path):
        all_w2v_top5.append([])
        continue
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_sentences, path)
    else:
        top5 = w2v_top5(input_text, stem_sentences, path)
    all_w2v_top5.append(top5)

# Fonksiyonu çağır
log_top5_to_csv_and_html(tfidf_lemma, tfidf_stem, w2v_models, lemma_sentences, stem_sentences, all_w2v_top5, input_text)