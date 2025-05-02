import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer,PorterStemmer
from snowballstemmer import TurkishStemmer



tqdm.pandas()


df = pd.read_csv("uyap_output.csv")
texts = df["text"].dropna().progress_apply(str)  
texts = texts.tolist()  

lemmatizer = WordNetLemmatizer()
stemmer = TurkishStemmer()

stop_words = set(stopwords.words('turkish'))

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)  # Cümleyi kelimelere ayır
    # Sadece harf olan kelimeleri al ve stopword'leri çıkar
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]  # Lemmatize etme
    stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]  # Stemleme
    
    return lemmatized_tokens, stemmed_tokens


sentences = []
for text in tqdm(texts, desc="Cümleler oluşturuluyor"):
    sentences.extend(sent_tokenize(text))  




# Her cümleyi tokenleştir, lemmatize et ve stemle
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)
print(f"Stem {tokenized_corpus_stemmed[:10]}")
print(f"Lemma {tokenized_corpus_lemmatized[:10]}")