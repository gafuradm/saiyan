import json
import chromadb
from chromadb.config import Settings
import uuid

class SimpleVectorStore:
    """Простое хранилище для слов"""
    
    def __init__(self):
        # Создаем папку для базы данных
        self.db_path = "database/chroma_db"
        
        # Подключаемся к ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Создаем коллекцию для слов
        self.collection = self.client.get_or_create_collection(
            name="hsk_vocabulary",
            metadata={"description": "Слова HSK для изучения"}
        )
    
    def add_words(self, words: List[Dict]):
        """Добавляем слова в базу"""
        documents = []
        metadatas = []
        ids = []
        
        for word in words:
            # Создаем текст для поиска
            text = f"""
            Слово: {word['character']}
            Пиньинь: {word['pinyin']}
            Перевод: {word['translation']}
            Уровень HSK: {word['hsk_level']}
            """
            
            documents.append(text)
            metadatas.append(word)
            ids.append(str(uuid.uuid4()))
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Добавлено {len(words)} слов в базу данных")
    
    def search_words(self, query: str, n_results: int = 5):
        """Ищем слова по запросу"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    
    def get_words_by_level(self, level: int):
        """Получаем слова определенного уровня"""
        results = self.collection.get(
            where={"hsk_level": level}
        )
        return results

if __name__ == "__main__":
    # Тестируем
    store = SimpleVectorStore()
    
    # Загружаем слова из файла
    with open("data/hsk_words.json", "r", encoding="utf-8") as f:
        words = json.load(f)
    
    # Добавляем в базу
    store.add_words(words)
    
    # Ищем слова
    results = store.search_words("привет")
    print("Найдены слова:", results)