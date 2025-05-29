import pandas as pd
import re
import zeyrek
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Türkçe stopword ve araçlar
lemmatizer = zeyrek.MorphAnalyzer()
stemmer = TurkishStemmer()
stop_words = set(stopwords.words('turkish'))

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', sentence)
    tokens = word_tokenize(sentence)
    filtered_tokens = [
        token for token in tokens
        if token.isalpha()
        and token not in stop_words
        and len(token) > 2
    ]
    lemmatized_tokens = []
    for token in filtered_tokens:
        try:
            analysis = lemmatizer.analyze(token)
            if analysis and analysis[0]:
                lemma = analysis[0][0][1]
                lemmatized_tokens.append(lemma)
        except:
            lemmatized_tokens.append(token)
    return lemmatized_tokens

def main():
    # Karar metinlerini oku
    df = pd.read_csv("uyap_output.csv")
    filenames = df["filename"].tolist()
    texts = df["icerik"].dropna().astype(str).tolist()

    # Kullanıcıdan cümle al
    print("Bir olay/cümle girin:")
    user_input = input()
    user_processed = ' '.join(preprocess_sentence(user_input))

    # Her karar için cümleleri çıkar ve ön işle
    karar_cumleleri = []  # (dosya_adı, cümle)
    for fname, text in zip(filenames, texts):
        cumleler = sent_tokenize(text)
        for cumle in cumleler:
            temiz = ' '.join(preprocess_sentence(cumle))
            if temiz.strip():
                karar_cumleleri.append((fname, temiz))

    # Tüm karar cümleleri + kullanıcı cümlesi
    all_cumleler = [c[1] for c in karar_cumleleri] + [user_processed]

    # TF-IDF vektörizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(all_cumleler)

    user_vec = tfidf_matrix[-1]
    karar_vecs = tfidf_matrix[:-1]

    # Her karar cümlesi ile benzerlik
    similarities = cosine_similarity(user_vec, karar_vecs)[0]
    max_idx = np.argmax(similarities)
    en_benzer_dosya = karar_cumleleri[max_idx][0]
    en_benzer_cumle = karar_cumleleri[max_idx][1]
    en_benzer_oran = similarities[max_idx]

    print(f"\nEn yüksek benzerlik oranı: {en_benzer_oran:.4f}")
    print(f"En benzer karar dosyası: {en_benzer_dosya}")
    print(f"En benzer karar cümlesi: {en_benzer_cumle}")

if __name__ == "__main__":
    main() 