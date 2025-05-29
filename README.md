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
- `zipf.py`: Zipf yasası analiz

## Kullanım
1. Veri Temizleme
```bash
python cleaner.py
```


2. Veri ön işleme:
```bash
python trainmodel.py
```

3. TF-IDF analizi:
```bash
python tfidfmodel.py
```

4. Model kullanımı:
```bash
python model.py
```

## Model Parametreleri

Projede farklı parametrelerle eğitilmiş Word2Vec modelleri bulunmaktadır:

- CBOW ve Skip-gram mimarileri
- Pencere boyutları: 2 ve 4
- Vektör boyutları: 100 ve 300
- Lemmatization ve stemming ön işleme teknikleri

# Ödev-2: Metin Benzerliği Hesaplama ve Değerlendirme

## Amaç

Bu projede, **Doğal Dil İşleme** dersi kapsamında, metinler arası benzerliği ölçmek amacıyla TF-IDF ve Word2Vec modelleri kullanılmıştır. Farklı model yapılandırmaları ile elde edilen sonuçlar karşılaştırılmış, modellerin performansları hem sayısal hem de anlamsal olarak değerlendirilmiştir.

## Kullanılan Veri Seti

- **Dosyalar:**  
  - `lemmatized.csv`  
  - `stemmed.csv`  
  (Bu dosyalar, Ödev-1 çıktılarıdır.)

- **Veri Yapısı:**  
  - `document_id`: Metne ait benzersiz kimlik  
  - `content`: Metnin işlenmiş (lema veya kök) hali

Her dosya, iki sütundan oluşan bir CSV formatındadır.

## Modeller

### TF-IDF
- `tf-idf_lemmatized`
- `tf-idf_stemmed`

### Word2Vec
Toplam 16 model:
- 8 adet lemmatized (CBOW/SkipGram, farklı window ve vektör boyutları)
- 8 adet stemmed (CBOW/SkipGram, farklı window ve vektör boyutları)

| Model Türü | Yöntem   | Window | Vektör Boyutu |
|------------|----------|--------|---------------|
| Lemmatized | CBOW     | 5, 10  | 100, 300      |
| Lemmatized | SkipGram | 5, 10  | 100, 300      |
| Stemmed    | CBOW     | 5, 10  | 100, 300      |
| Stemmed    | SkipGram | 5, 10  | 100, 300      |

## Yöntem

- **Benzerlik Hesaplama:**  
  - Cosine Similarity ile metinler arası benzerlik ölçülmüştür.
- **Model Tutarlılığı:**  
  - Jaccard benzerliği ile farklı modellerin ilk 5 benzer metin seçimlerinin tutarlılığı analiz edilmiştir.

### Örnek Benzerlik Hesaplama Fonksiyonu

```python
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(vector1, vector2):
    """
    İki vektör arasındaki kosinüs benzerliğini hesaplar.
    """
    return cosine_similarity([vector1], [vector2])[0][0]
```

### Jaccard Benzerliği Hesaplama

```python
def jaccard_similarity(list1, list2):
    """
    İki küme/listenin Jaccard benzerliğini hesaplar.
    """
    set1, set2 = set(list1), set(list2)
    return len(set1 & set2) / len(set1 | set2)
```

## Çalıştırma Talimatları

### Gereksinimler

- Python 3.x
- pandas
- scikit-learn
- gensim
- numpy

Gereksinimleri yüklemek için:
```bash
pip install -r requirements.txt
```

### Projeyi Çalıştırmak için

```bash
python similarity_analysis.py
```

### Örnek Giriş

Veri setinden bir örnek metin (`doc1`) ile benzerlik analizi yapılabilir.  
`similarity_analysis.py` çalıştırıldığında, seçilen metin için en benzer 5 metin ve skorları ekrana ve dosyalara yazdırılır.

## Sonuçlar ve Değerlendirme

### Her Model İçin İlk 5 Benzer Metin
Her model için giriş metnine en çok benzeyen ilk 5 metin ve bu metinlere ait benzerlik skorları, hem tablo hem de HTML formatında detaylı şekilde sunulmuştur.  
Detaylı tablo ve metinler için bakınız: [benzerlik_top5.html](./benzerlik_top5.html) ve [benzerlik_top5.csv](./benzerlik_top5.csv)

### Benzerlik Skorlarının Tablo veya Matrislerle Gösterimi
Her modelin ilk 5 benzer metni ve skorları, ilgili dosyalarda tablo olarak sunulmuştur.  
Ayrıca, farklı modellerin ilk 5 benzer metin seçimlerinin ne kadar örtüştüğünü gösteren Jaccard benzerlik matrisi de hazırlanmıştır.  
Jaccard matrisi için bakınız: [jaccard_similarity_matrix.html](./jaccard_similarity_matrix.html) ve [jaccard_similarity_matrix.csv](./jaccard_similarity_matrix.csv)

### Hangi Model(ler) Daha Başarılıydı? Yorumlar
- **TF-IDF Modelleri:**  
  TF-IDF tabanlı modeller, özellikle kısa ve anahtar kelime odaklı metinlerde yüksek başarı göstermiştir. Lemmatize edilmiş verilerle çalışan TF-IDF modeli, genellikle daha anlamlı ve tutarlı sonuçlar üretmiştir.
- **Word2Vec Modelleri:**  
  Word2Vec tabanlı modeller, metinlerin anlamsal yakınlığını daha iyi yakalayabilmektedir. Özellikle CBOW ve SkipGram yöntemlerinin yüksek boyutlu (ör. 300) ve geniş pencere (window) değerleriyle eğitilmiş halleri, benzerlik skorlarında öne çıkmıştır.
- **Genel Değerlendirme:**  
  TF-IDF, kelime bazlı benzerlikte hızlı ve etkili sonuçlar verirken; Word2Vec, anlamsal yakınlık ve bağlamı yakalamada daha başarılıdır. Özellikle karmaşık ve uzun metinlerde Word2Vec modelleri daha anlamlı sonuçlar üretmiştir.

### Model Yapılandırmalarının (Pencere Boyutu, Boyut Sayısı vb.) Başarıya Etkisi
- **Pencere (Window) Boyutu:**  
  Daha büyük pencere boyutları (ör. 10), kelimeler arası bağlamı daha geniş tutarak, anlamsal benzerliği artırmıştır. Ancak çok büyük pencere değerlerinde, bazı anlamsal kaymalar gözlenmiştir.
- **Vektör Boyutu (Dim):**  
  300 boyutlu vektörler, 100 boyutlu olanlara göre genellikle daha yüksek benzerlik skorları üretmiştir. Yüksek boyut, kelimeler arası ilişkileri daha iyi modelleyebilmiştir.
- **CBOW vs SkipGram:**  
  SkipGram modelleri, nadir kelimeler ve uzun metinlerde daha başarılı olurken; CBOW, sık kullanılan kelimeler ve kısa metinlerde daha iyi performans göstermiştir.

### Sonuçların Görselleştirilmesi ve Dosya Referansları
- Her modelin ilk 5 benzer metni ve skorları için:  
  → [benzerlik_top5.html](./benzerlik_top5.html)  
  → [benzerlik_top5.csv](./benzerlik_top5.csv)
- Jaccard benzerlik matrisi için:  
  → [jaccard_similarity_matrix.html](./jaccard_similarity_matrix.html)  
  → [jaccard_similarity_matrix.csv](./jaccard_similarity_matrix.csv)

---

## Sonuç ve Öneriler

### Genel Çıkarımlar
Bu çalışmada, metin benzerliği hesaplama ve değerlendirme amacıyla TF-IDF ve Word2Vec tabanlı çeşitli modeller karşılaştırılmıştır. Elde edilen sonuçlar göstermiştir ki:
- **TF-IDF modelleri**, özellikle kısa, anahtar kelime odaklı ve doğrudan kelime eşleşmesinin önemli olduğu durumlarda hızlı ve etkili sonuçlar vermektedir.
- **Word2Vec modelleri** ise, metinler arası anlamsal yakınlığı ve bağlamsal ilişkileri daha iyi yakalayabilmekte, özellikle uzun ve karmaşık metinlerde daha anlamlı benzerlikler sunmaktadır.
- Model yapılandırmalarında pencere boyutu ve vektör boyutunun artırılması, genellikle benzerlik skorlarını ve anlamsal tutarlılığı olumlu yönde etkilemiştir.

### Hangi Model, Hangi Tür Görevler İçin Daha Uygun?
- **TF-IDF**  
  - Kısa metinler, anahtar kelime arama, hızlı filtreleme ve temel metin benzerliği analizlerinde tercih edilmelidir.
  - Özellikle kelime bazlı arama ve özet çıkarma gibi görevlerde yüksek performans gösterir.
- **Word2Vec**  
  - Anlamsal yakınlık gerektiren, bağlamın önemli olduğu, uzun ve karmaşık metinlerin karşılaştırılmasında daha uygundur.
  - Hukuki metinler, literatür taramaları, öneri sistemleri ve anlamsal arama gibi görevlerde öne çıkar.
  - SkipGram, nadir kelimeler ve uzun metinlerde; CBOW ise sık kullanılan kelimeler ve kısa metinlerde daha iyi sonuçlar verebilir.

### Öneriler
- Görev türüne göre model seçimi yapılmalıdır.  
  Anahtar kelime odaklı, hızlı ve basit analizler için TF-IDF; anlamsal ve bağlamsal analizler için Word2Vec tercih edilmelidir.
- Model yapılandırmaları (pencere boyutu, vektör boyutu) ihtiyaca göre optimize edilmelidir.
- Daha yüksek doğruluk ve tutarlılık için, farklı model sonuçlarının birleştirildiği hibrit yaklaşımlar da değerlendirilebilir.

---

## Referanslar

- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [gensim](https://radimrehurek.com/gensim/)
- [numpy](https://numpy.org/)


## Ek Notlar

- Proje dili Türkçedir.
- Kod ve açıklamalar profesyonel ve anlaşılır şekilde hazırlanmıştır.
- Geliştirme ve testler Windows 10 ortamında yapılmıştır.

---

Teşekkürler!  
Herhangi bir sorunuz veya katkınız için lütfen iletişime geçin.

