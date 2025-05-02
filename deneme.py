from tqdm import tqdm
import time

# Basit bir yükleme barı örneği
for i in tqdm(range(100), desc="Yükleme", ncols=100):
    time.sleep(0.1)  # Simülasyon olarak her iterasyon arasında 0.1 saniye bekleme
