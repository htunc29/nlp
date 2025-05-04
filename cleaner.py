import os
import csv
from bs4 import BeautifulSoup
import re

folder_path = "uyap_documents"
output_csv = "uyap_output.csv"

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    full_text = "\n".join(lines)
    
    # "Gereği düşünüldü"den sonrasını al
    patterns = [
        r"(GEREĞİ (GÖRÜŞÜLÜP )?DÜŞÜNÜLDÜ.*)",
        r"(G\s*E\s*R\s*E\s*Ğ\s*İ\s*\s*D\s*Ü\s*Ş\s*Ü\s*N\s*Ü\s*L\s*D\s*Ü.*)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            break
    if match:
        return match.group(1).strip()
    return full_text  # Eğer "Gereği düşünüldü" bulunamazsa tüm metni döndür

def extract_field(text, keywords):
    for keyword in keywords:
        pattern = rf"{keyword}[:\s]+(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def extract_metadata_and_text(cleaned_text, original_text):
    metadata = {}

    metadata["mahkeme"] = extract_field(original_text, ["MAHKEMESİ", "Mahkemesi"])
    metadata["dosya_no"] = extract_field(original_text, ["DOSYA NO", "Dosya No", "Dosya no"])
    metadata["karar_no"] = extract_field(original_text, ["KARAR NO", "Karar No"])
    metadata["tarih"] = extract_field(original_text, ["TARİHİ", "Tarih"])
    metadata["dava_konusu"] = extract_field(original_text, ["DAVANIN KONUSU", "DAVA KONUSU", "DAVA", "Dava", "Davanın Konusu"])
    
    metadata["icerik"] = cleaned_text  # Sadece "Gereği düşünüldü" sonrası

    return metadata

def process_files():
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                html_content = file.read()
                full_cleaned_text = clean_text(html_content)
                original_cleaned_text = BeautifulSoup(html_content, "html.parser").get_text(separator="\n")
                metadata = extract_metadata_and_text(full_cleaned_text, original_cleaned_text)
                metadata["filename"] = filename
                all_data.append(metadata)

    # CSV'ye yaz
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "mahkeme", "dosya_no", "karar_no", "tarih", "dava_konusu", "icerik"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

process_files()
