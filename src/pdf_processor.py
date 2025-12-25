import pdfplumber
import json
import os
import re
from typing import List, Dict

print("=" * 60)
print("🎌 ПАРСЕР PDF СЛОВАРЕЙ HSK 1-6")
print("=" * 60)

class HSKPDFParser:
    def __init__(self):
        self.words = []
    
    def process_all_levels(self):
        """Обрабатываем HSK 1-6"""
        for level in range(1, 7):
            pdf_file = f"data/hsk{level}.pdf"
            if os.path.exists(pdf_file):
                print(f"\n📖 Обрабатываю HSK {level}...")
                words = self.process_pdf(pdf_file, level)
                print(f"   ✅ Найдено слов: {len(words)}")
                self.words.extend(words)
            else:
                print(f"\n⚠️  Файл не найден: {pdf_file}")
    
    def process_pdf(self, pdf_path: str, level: int) -> List[Dict]:
        """Обрабатываем один PDF файл"""
        words = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Разбиваем текст на строки
                    lines = text.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Пробуем разные форматы
                        word = self.parse_line(line, level)
                        if word:
                            words.append(word)
            
            print(f"   📄 Страниц обработано: {len(pdf.pages)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}")
        
        return words
    
    def parse_line(self, line: str, level: int) -> Dict:
        """Парсим строку с китайским словом"""
        try:
            # Убираем лишние пробелы
            line = re.sub(r'\s+', ' ', line)
            
            # Пытаемся найти китайские иероглифы
            chinese_match = re.search(r'[\u4e00-\u9fff]+', line)
            if not chinese_match:
                return None
            
            character = chinese_match.group(0)
            
            # Ищем пиньинь (латинские буквы с цифрами-тонами)
            pinyin_match = re.search(r'[a-zA-ZüÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\s]+[1-5]', line)
            pinyin = pinyin_match.group(0).strip() if pinyin_match else ""
            
            # Ищем перевод (всё после пиньиня или после иероглифов)
            translation = ""
            if pinyin:
                # Находим позицию после пиньиня
                pinyin_end = pinyin_match.end()
                translation = line[pinyin_end:].strip()
            else:
                # Если нет пиньиня, берём всё после иероглифов
                chinese_end = chinese_match.end()
                translation = line[chinese_end:].strip()
            
            # Очищаем перевод
            translation = re.sub(r'[\[\]()\d]', '', translation).strip()
            translation = translation.split(';')[0].split('.')[0].strip()
            
            # Определяем часть речи
            part_of_speech = self.detect_part_of_speech(translation)
            
            if len(character) > 0 and len(translation) > 0:
                return {
                    "character": character,
                    "pinyin": pinyin,
                    "translation": translation[:150],  # Ограничиваем длину
                    "hsk_level": level,
                    "part_of_speech": part_of_speech,
                    "frequency": "高频" if level <= 3 else "中频" if level <= 5 else "低频"
                }
        
        except Exception as e:
            # print(f"Ошибка парсинга строки: {line[:30]}... - {e}")
            pass
        
        return None
    
    def detect_part_of_speech(self, translation: str) -> str:
        """Определяем часть речи по переводу"""
        translation_lower = translation.lower()
        
        pos_patterns = {
            "глагол": ["гл", "verb", "v.", "делать", "ходить", "говорить", "смотреть"],
            "существительное": ["сущ", "noun", "n.", "предмет", "человек", "место", "вещь"],
            "прилагательное": ["прил", "adjective", "adj.", "красивый", "большой", "маленький"],
            "наречие": ["нар", "adverb", "adv.", "быстро", "медленно", "хорошо"],
            "местоимение": ["мест", "pronoun", "pron.", "я", "ты", "он", "она"],
            "числительное": ["числ", "numeral", "num.", "один", "два", "три", "первый"],
            "предлог": ["предл", "preposition", "prep.", "в", "на", "под", "над"],
            "союз": ["союз", "conjunction", "conj.", "и", "или", "но"],
            "частица": ["част", "particle", "part.", "же", "ли", "бы"]
        }
        
        for pos, patterns in pos_patterns.items():
            for pattern in patterns:
                if pattern in translation_lower:
                    return pos
        
        return "не указано"
    
    def save_results(self):
        """Сохраняем результаты"""
        if not self.words:
            print("\n❌ Не найдено ни одного слова!")
            return
        
        # Создаем папки
        os.makedirs("data", exist_ok=True)
        
        # Сохраняем все слова
        all_file = "data/hsk_all_words.json"
        with open(all_file, "w", encoding="utf-8") as f:
            json.dump(self.words, f, ensure_ascii=False, indent=2)
        
        # Сохраняем по уровням
        for level in range(1, 7):
            level_words = [w for w in self.words if w["hsk_level"] == level]
            if level_words:
                level_file = f"data/hsk{level}_words.json"
                with open(level_file, "w", encoding="utf-8") as f:
                    json.dump(level_words, f, ensure_ascii=False, indent=2)
        
        # Показываем статистику
        self.show_stats()
        
        print(f"\n💾 Основной файл: {all_file}")
        for level in range(1, 7):
            level_words = [w for w in self.words if w["hsk_level"] == level]
            if level_words:
                print(f"📁 HSK {level}: {len(level_words)} слов -> data/hsk{level}_words.json")
    
    def show_stats(self):
        """Показываем статистику"""
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
        print("=" * 60)
        
        total = len(self.words)
        print(f"🎯 Всего слов: {total}")
        
        # По уровням
        print("\n📈 По уровням HSK:")
        for level in range(1, 7):
            level_words = [w for w in self.words if w["hsk_level"] == level]
            count = len(level_words)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  HSK {level}: {count:4d} слов ({percentage:.1f}%)")
        
        # По частям речи
        print("\n🔤 По частям речи:")
        pos_stats = {}
        for word in self.words:
            pos = word["part_of_speech"]
            pos_stats[pos] = pos_stats.get(pos, 0) + 1
        
        for pos, count in sorted(pos_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {pos:15s}: {count:4d} слов ({percentage:.1f}%)")
        
        # Примеры слов
        print("\n📝 Примеры слов:")
        for level in range(1, 4):  # Показываем HSK 1-3
            level_words = [w for w in self.words if w["hsk_level"] == level]
            if level_words:
                sample = level_words[:3]
                print(f"  HSK {level}: ", end="")
                for word in sample:
                    print(f"{word['character']} ({word['pinyin']}) = {word['translation'][:20]}...", end=" | ")
                print()

# Главная функция
def main():
    # Проверяем папку с PDF
    pdf_folder = "data"
    if not os.path.exists(pdf_folder):
        print(f"❌ Папка '{pdf_folder}' не существует!")
        print("\n📁 Создайте структуру папок:")
        print("saiyan/")
        print("├── data/")
        print("│   └── pdf/")
        print("│       ├── hsk1.pdf")
        print("│       ├── hsk2.pdf")
        print("│       ├── ...")
        print("│       └── hsk6.pdf")
        print("└── src/")
        print("    └── pdf_processor.py")
        return
    
    # Проверяем наличие файлов
    missing_files = []
    for level in range(1, 7):
        if not os.path.exists(f"data/hsk{level}.pdf"):
            missing_files.append(f"hsk{level}.pdf")
    
    if missing_files:
        print("⚠️  Не найдены файлы:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\n📁 Положите файлы в: {os.path.abspath('data')}/")
    
    # Запускаем парсер
    parser = HSKPDFParser()
    parser.process_all_levels()
    parser.save_results()
    
    print("\n" + "=" * 60)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("\n🚀 ДАЛЬШЕ:")
    print("1. Запусти сервер: python src/main.py")
    print("2. Открой в браузере: http://localhost:8000")
    print("3. Тестируй API: http://localhost:8000/docs")
    print("\n📚 Доступные команды API:")
    print("   • GET /stats - статистика базы")
    print("   • GET /search/你好 - поиск слова")
    print("   • GET /test/1 - тест HSK 1")
    print("   • GET /words/level/2 - слова HSK 2")
    print("=" * 60)

if __name__ == "__main__":
    main()