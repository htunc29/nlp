import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from typing import List, Dict

def load_and_preprocess_data(csv_file: str, text_column: str) -> str:
    """CSV dosyasını yükler ve metin verilerini ön işler"""
    df = pd.read_csv(csv_file)
    return ' '.join(df[text_column].dropna().astype(str))

def tokenize_text(text: str) -> List[str]:
    """Metni tokenize eder ve temizler"""
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    return nltk.word_tokenize(cleaned_text, language="turkish")

def calculate_word_frequencies(words: List[str]) -> Dict[str, int]:
    """Kelimelerin frekanslarını hesaplar"""
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

def plot_zipf_law(sorted_freqs: List[int], title: str) -> None:
    """Zipf yasasını görselleştirir"""
    ranks = np.arange(1, len(sorted_freqs) + 1)
    
    # Stil ayarları
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    
    # Grafik oluşturma
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Ana veri çizimi
    scatter = ax.loglog(ranks, sorted_freqs, 
                       marker="o", 
                       linestyle="none", 
                       markersize=5,
                       alpha=0.6,
                       color="#2E86C1",
                       label="Kelime Frekansları")
    
    # Teorik Zipf çizgisi (referans için)
    theoretical_line = ax.loglog(ranks, sorted_freqs[0] / ranks,
                               linestyle="--",
                               color="#E74C3C",
                               alpha=0.7,
                               label="Teorik Zipf Çizgisi")
    
    # Eksen etiketleri ve başlık
    ax.set_xlabel("Kelime Sırası (log)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frekans (log)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Izgara ve arka plan
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_facecolor('#F8F9F9')
    fig.patch.set_facecolor('white')
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Kenar boşlukları
    plt.tight_layout()
    
    # Grafik gösterimi
    plt.show()

def analyze_zipf_law(csv_file: str, text_column: str, title: str) -> None:
    """Verilen CSV dosyası için Zipf yasası analizi yapar"""
    # Veriyi yükle ve ön işle
    text = load_and_preprocess_data(csv_file, text_column)
    
    # Tokenize et
    words = tokenize_text(text)
    
    # Frekansları hesapla
    word_freq = calculate_word_frequencies(words)
    
    # Frekansları sırala
    sorted_freqs = sorted(word_freq.values(), reverse=True)
    
    # Görselleştir
    plot_zipf_law(sorted_freqs, title)

def main():
    # Orijinal metin analizi
    analyze_zipf_law(
        "uyap_output.csv",
        "icerik",
        "UYAP Karar Metinleri İçin Zipf Yasası"
    )
    
    # Stemmed metin analizi
    analyze_zipf_law(
        "stemmed.csv",
        "processed_tokens",
        "UYAP Karar Metinleri Stemmed İçin Zipf Yasası"
    )
    
    # Lemmatized metin analizi
    analyze_zipf_law(
        "lemmatized.csv",
        "processed_tokens",
        "UYAP Karar Metinleri Lemmatize İçin Zipf Yasası"
    )

if __name__ == "__main__":
    main()
