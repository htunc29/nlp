import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer,PorterStemmer
from snowballstemmer import TurkishStemmer
import zeyrek
import wikipedia

tqdm.pandas()
wikipedia.set_lang("tr")
page=wikipedia.page("Mustafa Kemal Atatürk")
texts=page.content

lemmatizer = zeyrek.MorphAnalyzer()
stemmer = TurkishStemmer()

stop_words = set(stopwords.words('turkish'))

sentences=sent_tokenize(texts)

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)  # Cümleyi kelimelere ayır
    # Sadece harf olan kelimeleri al ve stopword'leri çıkar
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    lemmatized_tokens = [lemmatizer.analyze(token) for token in filtered_tokens]  # Lemmatize etme
    stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]  # Stemleme
    
    return lemmatized_tokens, stemmed_tokens

# Her cümleyi tokenleştir, lemmatize et ve stemle
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)

# İlk 5 cümleyi yazdıralım
zeyrek_lemmatatize=[]
for tokenize in tokenized_corpus_lemmatized:
    for word_analysis in tokenize:  # Her kelime için analiz sonuçları
        # İlk analiz sonucunun ilk lemmasını al
        first_lemma = word_analysis[0][0][1]  # [0] ilk analiz, [1] lemma bilgisi
        zeyrek_lemmatatize.append(first_lemma)
print(f"Cümle  - Lemma: {zeyrek_lemmatatize[:10]}")
print(f"Cümle  - Stemmed: {tokenized_corpus_stemmed[0][:10]}")


