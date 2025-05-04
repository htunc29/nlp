import os
import csv
from bs4 import BeautifulSoup
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleaner.log'),
        logging.StreamHandler()
    ]
)

class DocumentCleaner:
    def __init__(self, folder_path: str = "uyap_documents", output_csv: str = "uyap_output.csv"):
        self.folder_path = Path(folder_path)
        self.output_csv = output_csv
        self.patterns = [
            r"(GEREĞİ (GÖRÜŞÜLÜP )?DÜŞÜNÜLDÜ.*)",
            r"(G\s*E\s*R\s*E\s*Ğ\s*İ\s*\s*D\s*Ü\s*Ş\s*Ü\s*N\s*Ü\s*L\s*D\s*Ü.*)",
            r"(GEREĞİ (GÖRÜŞÜLÜP )?DÜŞÜNÜLDÜ.*?)(?=\n[A-Z]{2,}|$)",
            r"(GEREĞİ (GÖRÜŞÜLÜP )?DÜŞÜNÜLDÜ.*?)(?=\n\d{2}\.\d{2}\.\d{4}|$)"
        ]
        
        self.metadata_patterns = {
            "mahkeme": [
                r"(?:MAHKEMESİ|Mahkemesi|MAHKEME|Mahkeme)[:\s]+(.+?)(?=\n|$)",
                r"(\d+\.\s*[A-ZÇĞİÖŞÜ\s]+DAİRESİ)",
                r"([A-ZÇĞİÖŞÜ\s]+MAHKEMESİ)"
            ],
            "dosya_no": [
                r"(?:DOSYA NO|Dosya No|Dosya no|Dosya No:|Dosya No :)[:\s]+(.+?)(?=\n|$)",
                r"(\d{4}/\d+)\s*Esas"
            ],
            "karar_no": [
                r"(?:KARAR NO|Karar No|Karar no|Karar No:|Karar No :)[:\s]+(.+?)(?=\n|$)",
                r"(\d{4}/\d+)\s*Karar"
            ],
            "tarih": [
                r"(?:TARİHİ|Tarih|TARİH|Tarih:|Tarih :)[:\s]+(.+?)(?=\n|$)",
                r"(\d{2}/\d{2}/\d{4})"
            ],
            "dava_konusu": [
                r"(?:DAVANIN KONUSU|DAVA KONUSU|DAVA|Dava|Davanın Konusu|Dava Konusu)[:\s]+(.+?)(?=\n|$)",
                r"DAVA KONUSU:\s*(.+?)(?=\n|$)"
            ]
        }

    def normalize_text(self, text: str) -> str:
        """Metindeki fazla boşlukları ve yeni satırları temizler"""
        # Birden fazla boşluğu tek boşluğa indir
        text = re.sub(r'\s+', ' ', text)
        # Birden fazla yeni satırı tek yeni satıra indir
        text = re.sub(r'\n+', '\n', text)
        # Baştaki ve sondaki boşlukları temizle
        text = text.strip()
        return text

    def clean_text(self, text: str) -> str:
        try:
            soup = BeautifulSoup(text, "html.parser")
            raw_text = soup.get_text(separator="\n")
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            full_text = "\n".join(lines)
            
            for pattern in self.patterns:
                match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                if match:
                    return self.normalize_text(match.group(1))
            
            return self.normalize_text(full_text)
        except Exception as e:
            logging.error(f"Metin temizleme hatası: {str(e)}")
            return self.normalize_text(text)

    def extract_field(self, text: str, patterns: List[str]) -> str:
        try:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return self.normalize_text(match.group(1))
            return ""
        except Exception as e:
            logging.error(f"Alan çıkarma hatası: {str(e)}")
            return ""

    def extract_metadata_and_text(self, cleaned_text: str, original_text: str) -> Dict[str, str]:
        try:
            metadata = {}
            for field, patterns in self.metadata_patterns.items():
                metadata[field] = self.extract_field(original_text, patterns)
            
            metadata["icerik"] = cleaned_text
            return metadata
        except Exception as e:
            logging.error(f"Metadata çıkarma hatası: {str(e)}")
            return {"icerik": cleaned_text}

    def process_files(self) -> None:
        try:
            all_data = []
            
            if not self.folder_path.exists():
                raise FileNotFoundError(f"Klasör bulunamadı: {self.folder_path}")
            
            for filename in self.folder_path.glob("*.html"):
                try:
                    with open(filename, "r", encoding="utf-8") as file:
                        html_content = file.read()
                        full_cleaned_text = self.clean_text(html_content)
                        original_cleaned_text = BeautifulSoup(html_content, "html.parser").get_text(separator="\n")
                        metadata = self.extract_metadata_and_text(full_cleaned_text, original_cleaned_text)
                        metadata["filename"] = filename.name
                        all_data.append(metadata)
                        logging.info(f"İşlenen dosya: {filename.name}")
                except Exception as e:
                    logging.error(f"Dosya işleme hatası ({filename.name}): {str(e)}")
                    continue

            # CSV'ye yaz
            with open(self.output_csv, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["filename", "mahkeme", "dosya_no", "karar_no", "tarih", "dava_konusu", "icerik"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(all_data)
                
            logging.info(f"Toplam {len(all_data)} dosya işlendi ve {self.output_csv} dosyasına kaydedildi.")
            
        except Exception as e:
            logging.error(f"Genel işleme hatası: {str(e)}")
            raise

if __name__ == "__main__":
    cleaner = DocumentCleaner()
    cleaner.process_files()
