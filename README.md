# UYAP Doküman Analizi Projesi

Bu proje, UYAP (Ulusal Yargı Ağı Bilişim Sistemi) dokümanlarının analizi için geliştirilmiş bir doğal dil işleme (NLP) projesidir.

## Proje Açıklaması

Proje, Türkçe hukuki metinlerin analizi için çeşitli NLP tekniklerini kullanmaktadır. Temel olarak şu işlemleri gerçekleştirir:

1. Metin ön işleme (lemmatization ve stemming)
2. Word2Vec model eğitimi
3. TF-IDF analizi
4. Kelime benzerliği hesaplamaları

## Kullanılan Teknolojiler

- Python 3.x
- Gensim (Word2Vec)
- NLTK
- Scikit-learn
- Zeyrek (Türkçe morfolojik analiz)
- SnowballStemmer
- Pandas
- NumPy

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. NLTK veri setlerini indirin:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Dosya Yapısı

- `uyap_output.csv`: Ham veri seti
- `lemmatized.csv`: Lemmatization işlemi uygulanmış metinler
- `stemmed.csv`: Stemming işlemi uygulanmış metinler
- `trainmodel.py`: Word2Vec model eğitimi
- `tfidfmodel.py`: TF-IDF analizi
- `model.py`: Eğitilmiş modellerin kullanımı
- `cleaner.py`: Veri temizleme işlemleri
- `zeyrekdeneme.py`: Zeyrek kütüphanesi denemeleri
- `zipf.py`: Zipf yasası analizi

## Kullanım

1. Veri ön işleme:
```bash
python trainmodel.py
```

2. TF-IDF analizi:
```bash
python tfidfmodel.py
```

3. Model kullanımı:
```bash
python model.py
```

## Model Parametreleri

Projede farklı parametrelerle eğitilmiş Word2Vec modelleri bulunmaktadır:

- CBOW ve Skip-gram mimarileri
- Pencere boyutları: 2 ve 4
- Vektör boyutları: 100 ve 300
- Lemmatization ve stemming ön işleme teknikleri

