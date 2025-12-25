import os
import json
import asyncio
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GrammarExplainer:
    """AI-объяснитель грамматики китайского"""
    
    def __init__(self):
        self.client = self._get_client()
        self.grammar_topics = []  # Будет заполнено из main.py
        self.explanation_cache = {}
        
    def _get_client(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("⚠️  DeepSeek API ключ не найден. Использую резервные объяснения.")
            return None
        
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    async def explain_grammar(self, grammar_topic: Dict, user_level: str = "初") -> Dict:
        """
        Генерация AI-объяснения грамматической темы
        """
        # Если нет клиента или тем, возвращаем базовое объяснение
        if not self.client or not grammar_topic:
            return self._create_fallback_explanation(grammar_topic)
        
        cache_key = f"{grammar_topic.get('id', '')}_{user_level}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        try:
            prompt = self._create_grammar_prompt(grammar_topic, user_level)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Объясни тему грамматики: {grammar_topic.get('chinese', '')}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # Парсим JSON или создаем структуру
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                result = self._parse_ai_response(result_text)
            
            # Добавляем базовую информацию о теме
            result.update({
                "topic_id": grammar_topic.get("id", ""),
                "topic_chinese": grammar_topic.get("chinese", ""),
                "topic_pinyin": grammar_topic.get("pinyin", ""),
                "topic_english": grammar_topic.get("english", ""),
                "topic_russian": grammar_topic.get("russian", ""),
                "level": grammar_topic.get("level", "初"),
                "category": grammar_topic.get("category", ""),
                "tags": grammar_topic.get("tags", [])
            })
            
            # Кэшируем
            self.explanation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Ошибка генерации объяснения: {e}")
            return self._create_fallback_explanation(grammar_topic)
    
    def _create_grammar_prompt(self, topic: Dict, user_level: str) -> str:
        """Создание промпта для объяснения грамматики"""
        
        level_map = {
            "初": "начинающий (HSK 1-2)",
            "中": "средний (HSK 3-4)", 
            "高": "продвинутый (HSK 5-6)"
        }
        
        user_level_desc = level_map.get(user_level, "начинающий")
        
        return f"""Ты — эксперт-преподаватель китайского языка для русскоговорящих студентов.
Твоя задача — дать понятное, структурированное объяснение грамматической темы.

ИНФОРМАЦИЯ О ТЕМЕ:
- Китайское название: {topic.get('chinese', '')}
- Пиньинь: {topic.get('pinyin', '')}
- Английский перевод: {topic.get('english', '')}
- Русский перевод: {topic.get('russian', '')}
- Уровень сложности темы: {topic.get('level', '初')}
- Категория: {topic.get('category', '')}
- Описание: {topic.get('description', '')}
- Теги: {', '.join(topic.get('tags', []))}

УРОВЕНЬ УЧЕНИКА: {user_level_desc}

Верни ответ в формате JSON со следующими полями:
1. "topic_summary" - краткое резюме темы (2-3 предложения)
2. "basic_rule" - основное правило использования
3. "formula" - формула/структура грамматики (если применимо)
4. "examples" - массив из 3-5 примеров, каждый объект содержит:
   - "chinese": китайский текст
   - "pinyin": пиньинь
   - "translation": перевод на русский
   - "explanation": объяснение примера
5. "when_to_use" - когда использовать эту конструкцию
6. "common_mistakes" - частые ошибки русскоговорящих
7. "memory_tips" - советы по запоминанию
8. "practice_sentences" - 2-3 предложения для самостоятельной практики
9. "related_topics" - связанные темы грамматики

Важные требования:
1. Объясняй максимально просто и понятно
2. Приводи реальные, жизненные примеры
3. Учитывай уровень ученика ({user_level_desc})
4. Давай конкретные советы по запоминанию
5. Упомяни особенности для русскоговорящих

Формат: только JSON без дополнительного текста."""
    
    def _parse_ai_response(self, text: str) -> Dict:
        """Парсит текст AI в структурированный JSON"""
        import re
        
        # Пытаемся найти JSON в тексте
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Если не удалось, создаем структуру
        return {
            "topic_summary": text[:200] + "..." if len(text) > 200 else text,
            "basic_rule": "Смотри подробное объяснение выше",
            "formula": "Зависит от контекста использования",
            "examples": [
                {
                    "chinese": "这是一个例子",
                    "pinyin": "zhè shì yí gè lì zi",
                    "translation": "Это пример",
                    "explanation": "Базовый пример использования"
                }
            ],
            "when_to_use": "В соответствующих грамматических контекстах",
            "common_mistakes": "Избегайте дословного перевода с русского",
            "memory_tips": "Практикуйтесь с примерами ежедневно",
            "practice_sentences": ["Составьте свое предложение", "Переведите на китайский"],
            "related_topics": "Связанные грамматические конструкции"
        }
    
    def _create_fallback_explanation(self, topic: Dict) -> Dict:
        """Создание запасного объяснения если AI недоступен"""
        return {
            "topic_summary": f"{topic.get('chinese', '')} ({topic.get('english', '')}) - {topic.get('russian', '')}",
            "basic_rule": topic.get('description', 'Используйте согласно правилам грамматики'),
            "formula": f"Структура: зависит от использования {topic.get('chinese', 'темы')}",
            "examples": [
                {
                    "chinese": "示例句子",
                    "pinyin": "shì lì jù zi",
                    "translation": "пример предложения",
                    "explanation": "Базовый пример"
                }
            ],
            "when_to_use": "В соответствующих грамматических контекстах",
            "common_mistakes": "Избегайте дословного перевода с русского",
            "memory_tips": "Повторяйте несколько раз в день",
            "practice_sentences": ["练习用这个语法", "造自己的句子"],
            "related_topics": "Другие грамматические темы",
            "fallback": True,
            "note": "AI объяснение временно недоступно. Это базовое описание темы."
        }
    
    async def generate_practice(self, topic_id: str, difficulty: str = "medium") -> Dict:
        """Генерация практических упражнений"""
        try:
            # Находим тему
            topic = next((t for t in self.grammar_topics if t["id"] == topic_id), None)
            if not topic:
                return {"error": "Тема не найдена"}
            
            if not self.client:
                return self._create_fallback_exercises(topic)
            
            prompt = f"""Создай упражнения по китайской грамматике.

ТЕМА: {topic.get('chinese', '')} ({topic.get('pinyin', '')})
ОПИСАНИЕ: {topic.get('description', '')}
УРОВЕНЬ: {topic.get('level', '初')}
СЛОЖНОСТЬ УПРАЖНЕНИЙ: {difficulty}

Создай 4 типа упражнений:

1. МНОЖЕСТВЕННЫЙ ВЫБОР (3 вопроса):
   - Вопросы должны быть про конкретную тему
   - 4 варианта ответа (A, B, C, D)
   - Объяснение правильного ответа

2. ЗАПОЛНЕНИЕ ПРОПУСКОВ:
   - Текст с 3-4 пропусками
   - Использовать тему {topic.get('chinese', '')}

3. ИСПРАВЛЕНИЕ ОШИБОК:
   - 2-3 предложения с типичными ошибками
   - Объяснение ошибок

4. СОСТАВЛЕНИЕ ПРЕДЛОЖЕНИЙ:
   - Дать слова для составления предложений
   - Использовать тему грамматики

Верни ТОЛЬКО JSON в формате:
{{
    "multiple_choice": [
        {{
            "question": "текст вопроса",
            "options": ["A", "B", "C", "D"],
            "correct": "A",
            "explanation": "объяснение"
        }}
    ],
    "fill_in_blanks": "текст с пропусками",
    "error_correction": "текст с ошибками",
    "sentence_formation": "слова для составления предложений"
}}"""
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Ты создаешь упражнения по китайской грамматике. Отвечай ТОЛЬКО в формате JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # Очищаем и парсим JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                result = self._create_fallback_exercises(topic)
            
            # Добавляем метаданные
            result["topic"] = topic.get('chinese', '')
            result["topic_russian"] = topic.get('russian', '')
            result["difficulty"] = difficulty
            
            return result
            
        except Exception as e:
            print(f"Ошибка генерации упражнений: {e}")
            topic = next((t for t in self.grammar_topics if t["id"] == topic_id), None)
            return self._create_fallback_exercises(topic)
    
    def _create_fallback_exercises(self, topic: Dict) -> Dict:
        """Резервные упражнения"""
        return {
            "multiple_choice": [
                {
                    "question": f"Как правильно использовать {topic.get('chinese', 'эту грамматику')}?",
                    "options": [
                        "В утвердительных предложениях",
                        "Только в вопросах",
                        "Для выражения будущего времени",
                        "В отрицательных предложениях"
                    ],
                    "correct": "A",
                    "explanation": f"{topic.get('chinese', 'Эта конструкция')} используется в утвердительных предложениях"
                }
            ],
            "fill_in_blanks": f"Заполните пропуски, используя тему '{topic.get('chinese', '')}':\n\n1. 我昨天 ______ (делать) домашнюю работу。\n2. 他经常 ______ (использовать) эту конструкцию。",
            "error_correction": f"Исправьте ошибки в использовании темы '{topic.get('chinese', '')}':\n\n1. 我学中文在教室。\n2. 他吃饭了已经。",
            "sentence_formation": f"Составьте предложения, используя тему '{topic.get('chinese', '')}':\n\n我, 喜欢, 学习, 中文, 在, 学校",
            "note": "Упражнения сгенерированы автоматически"
        }
    
    async def answer_grammar_question(self, question: str, context: Dict = None) -> Dict:
        """Ответ на вопрос по грамматике"""
        if not self.client:
            return {"answer": "AI сервис временно недоступен. Проверьте настройки API."}
        
        try:
            prompt = "Ты — эксперт по грамматике китайского языка. Отвечай подробно и понятно."
            
            if context and context.get('topic'):
                prompt += f"\nКонтекст: тема '{context['topic'].get('chinese', '')}'"
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "answer": response.choices[0].message.content,
                "examples": "Изучите примеры в учебнике",
                "practice_tip": "Практикуйтесь ежедневно"
            }
            
        except Exception as e:
            return {"error": str(e)}

# Глобальный экземпляр
grammar_explainer = GrammarExplainer()