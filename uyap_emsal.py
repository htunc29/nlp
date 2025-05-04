import requests
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

class UyapEmsal:
    def __init__(self):
        self.base_url = "https://emsal.uyap.gov.tr"
        self.search_endpoint = "/aramalist"
        self.document_endpoint = "/getDokuman"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Dökümanların kaydedileceği klasör
        self.documents_dir = "uyap_documents"
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
        
        # CAPTCHA yönetimi için değişkenler
        self.request_count = 0
        self.last_captcha_time = None
        self.max_requests = 5  # CAPTCHA öncesi maksimum istek sayısı
        self.captcha_wait_time = 300  # CAPTCHA sonrası bekleme süresi (saniye)

    def check_captcha(self, response_data: Dict) -> bool:
        """
        Yanıtta CAPTCHA kontrolü var mı kontrol eder
        """
        if not response_data:
            return False
            
        metadata = response_data.get("metadata", {})
        error_message = metadata.get("FMTE", "")
        return "DisplayCaptcha" in error_message

    def handle_captcha(self) -> None:
        """
        CAPTCHA durumunu yönetir
        """
        current_time = time.time()
        
        if self.last_captcha_time:
            elapsed_time = current_time - self.last_captcha_time
            if elapsed_time < self.captcha_wait_time:
                wait_time = self.captcha_wait_time - elapsed_time
                print(f"\nCAPTCHA bekleme süresi devam ediyor. {int(wait_time)} saniye daha bekleniyor...")
                time.sleep(wait_time)
        
        self.last_captcha_time = current_time
        self.request_count = 0
        print(f"\nCAPTCHA tespit edildi. {self.captcha_wait_time} saniye bekleniyor...")
        time.sleep(self.captcha_wait_time)

    def search_decisions(self, keyword: str, page_size: int = 100, page_number: int = 2) -> Optional[Dict]:
        """
        UYAP'ta emsal karar araması yapar
        """
        payload = {
            "data": {
                "aranan": keyword,
                "arananKelime": "***",
                "pageSize": page_size,
                "pageNumber": page_number
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.search_endpoint}",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            
            if self.check_captcha(data):
                self.handle_captcha()
                return self.search_decisions(keyword, page_size, page_number)
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"Arama sırasında hata oluştu: {e}")
            return None

    def get_document(self, document_id: str) -> Optional[Dict]:
        """
        Belirli bir kararın detaylarını getirir
        """
        self.request_count += 1
        
        if self.request_count >= self.max_requests:
            print(f"\nMaksimum istek sayısına ({self.max_requests}) ulaşıldı. Kısa bir süre bekleniyor...")
            time.sleep(10)  # 10 saniye bekle
            self.request_count = 0
            
        try:
            response = requests.get(
                f"{self.base_url}{self.document_endpoint}",
                params={"id": document_id},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if self.check_captcha(data):
                self.handle_captcha()
                return self.get_document(document_id)
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"Doküman getirme sırasında hata oluştu: {e}")
            return None

    def create_safe_filename(self, decision_info: Dict) -> str:
        """
        Güvenli bir dosya adı oluşturur
        """
        # Dosya adındaki geçersiz karakterleri değiştir
        esas_no = decision_info['esasNo'].replace('/', '_')
        karar_no = decision_info['kararNo'].replace('/', '_')
        return f"{esas_no}_{karar_no}_{decision_info['id']}"

    def save_document(self, document: Dict, decision_info: Dict) -> None:
        """
        Dökümanı HTML ve JSON formatında kaydeder
        """
        try:
            # JSON dosyasını kaydet
            json_filename = self.create_safe_filename(decision_info) + ".json"
            json_filepath = os.path.join(self.documents_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=4)
            
            # HTML içeriğini kontrol et ve kaydet
            html_content = document.get("data", "")
            if html_content and html_content.strip():
                soup = BeautifulSoup(html_content, 'html.parser')
                clean_html = soup.prettify()
                
                # HTML dosyasını kaydet
                html_filename = self.create_safe_filename(decision_info) + ".html"
                html_filepath = os.path.join(self.documents_dir, html_filename)
                
                with open(html_filepath, 'w', encoding='utf-8') as f:
                    f.write(clean_html)
                print(f"Dökümanlar başarıyla kaydedildi: {html_filename} ve {json_filename}")
            else:
                print(f"HTML içeriği boş, sadece JSON kaydedildi: {json_filename}")
                
        except Exception as e:
            print(f"Döküman kaydedilirken hata oluştu: {e}")

def main():
    uyap = UyapEmsal()
    
    try:
        total_pages = 6753  # 675215 / 100 = 6752.15, yani 6753 sayfa
        total_decisions_processed = 0
        
        for page in range(1, total_pages + 1):
            print(f"\nSayfa {page}/{total_pages} işleniyor...")
            
            # Arama yap
            search_result = uyap.search_decisions("borç", page_number=page)
            
            if search_result and search_result.get("data", {}).get("data"):
                decisions = search_result["data"]["data"]
                page_decisions = len(decisions)
                total_decisions_processed += page_decisions
                
                print(f"Bu sayfada {page_decisions} karar bulundu.")
                print(f"Toplam işlenen karar sayısı: {total_decisions_processed}")
                
                for index, decision in enumerate(decisions, 1):
                    print(f"\nKarar {index}/{page_decisions} (Sayfa {page})")
                    print(f"Daire: {decision['daire']}")
                    print(f"Esas No: {decision['esasNo']}")
                    print(f"Karar No: {decision['kararNo']}")
                    print(f"Karar Tarihi: {decision['kararTarihi']}")
                    
                    try:
                        # Her kararın detaylarını al
                        document = uyap.get_document(decision['id'])
                        if document:
                            # Dökümanı kaydet
                            uyap.save_document(document, decision)
                        else:
                            print("Doküman alınamadı")
                    except Exception as e:
                        print(f"İşlem sırasında hata oluştu: {e}")
                        continue
                    
                    # API'yi yormamak için kısa bir bekleme
                    time.sleep(2)
            else:
                print(f"Sayfa {page} için sonuç alınamadı.")
                
            # Sayfalar arası daha uzun bir bekleme
            time.sleep(5)
                
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {e}")
    finally:
        print(f"\nToplam {total_decisions_processed} karar işlendi.")

if __name__ == "__main__":
    main() 