import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re

# CSV dosyasını oku
df = pd.read_csv("uyap_output.csv")


all_text = ' '.join(df['icerik'].dropna().astype(str)) 


words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', all_text.lower()), language="turkish")


word_freq = {}
for word in words:
    word_freq[word] = word_freq.get(word, 0) + 1


sorted_freqs = sorted(word_freq.values(), reverse=True)


ranks = np.arange(1, len(sorted_freqs) + 1)


plt.figure(figsize=(8, 6))
plt.loglog(ranks, sorted_freqs, marker="o", linestyle="none", markersize=4, alpha=0.7, color="r")
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")
plt.title("UYAP Karar Metinleri İçin Zipf Yasası")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
