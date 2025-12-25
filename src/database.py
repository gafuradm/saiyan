import pdfplumber
import json
import os
from typing import List, Dict

class SimplePDFParser:
    """Простой парсер PDF для словаря HSK"""
    
    def read_pdf(self, pdf_path: str):
        """Читаем PDF и возвращаем текст"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Ошибка чтения PDF: {e}")
        return text
    
    def parse_hsk_words(self, text: str, level: int = 1):
        """Парсим слова из текста (самая простая версия)"""
        words = []
        
        # Разделяем текст по строкам
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Простейший парсинг - ищем китайские иероглифы
            # В реальности нужно адаптировать под формат твоего PDF!
            if any('\u4e00' <= char <= '\u9fff' for char in line):
                # Пример: "你好 nǐ hǎo hello"
                parts = line.split()
                
                if len(parts) >= 3:
                    word = {
                        "character": parts[0],  # китайский
                        "pinyin": parts[1],     # пиньинь
                        "translation": " ".join(parts[2:]),  # перевод
                        "hsk_level": level,
                        "learned": False,
                        "last_reviewed": None
                    }
                    words.append(word)
        
        return words

# Если PDF сложный, создай TXT файл вручную!
def create_simple_hsk_txt():
    """Создаем простой текстовый файл со словами если PDF не читается"""
    words = [
        {"character": "你好", "pinyin": "nǐ hǎo", "translation": "привет", "hsk_level": 1},
        {"character": "谢谢", "pinyin": "xiè xie", "translation": "спасибо", "hsk_level": 1},
        {"character": "再见", "pinyin": "zài jiàn", "translation": "до свидания", "hsk_level": 1},
        {"character": "中国", "pinyin": "zhōng guó", "translation": "Китай", "hsk_level": 2},
        {"character": "学习", "pinyin": "xué xí", "translation": "учиться", "hsk_level": 2},
    ]
    
    with open("data/hsk_words.json", "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    
    print("Создан файл data/hsk_words.json с 5 словами для теста")
    return words

if __name__ == "__main__":
    # Сначала создай простой файл для теста
    create_simple_hsk_txt()