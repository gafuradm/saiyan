from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional
import re

load_dotenv()

class SmartChineseTranslator:
    """Умный переводчик с обучением"""
    
    def __init__(self):
        self.client = self._get_client()
        self.translation_cache = {}
        
    def _get_client(self):
        """Создаем клиент для DeepSeek"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API ключ не найден в .env файле")
        
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    async def smart_translate(self, text: str, user_level: int = 1, learning_style: str = "visual") -> Dict:
        """
        Умный перевод с объяснениями
        
        Args:
            text: Текст для перевода
            user_level: Уровень HSK пользователя
            learning_style: visual, auditory, kinesthetic
        
        Returns:
            Словарь с переводом и объяснениями
        """
        # Проверяем кэш
        cache_key = f"{text}_{user_level}_{learning_style}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Формируем промпт для AI
            system_prompt = f"""Ты — эксперт по китайскому языку и педагогике. Твоя задача не просто перевести текст, а ОБУЧИТЬ.

Пользователь: Уровень HSK {user_level}, стиль обучения: {learning_style}
Текст для перевода: "{text}"

Верни ответ в формате JSON с такими полями:
{{
    "original": оригинальный текст,
    "translation": литературный перевод,
    "word_by_word": дословный перевод по словам,
    "pinyin": пиньинь текста,
    "grammar_explanation": объяснение грамматики,
    "key_words": [
        {{
            "character": иероглиф,
            "pinyin": пиньинь,
            "translation": перевод,
            "explanation": объяснение (этимология, мнемоника),
            "memory_tip": совет по запоминанию,
            "hsk_level": уровень HSK
        }}
    ],
    "example_sentences": [
        {{
            "chinese": пример на китайском,
            "pinyin": пиньинь примера,
            "translation": перевод примера,
            "explanation": почему это хороший пример
        }}
    ],
    "study_tips": советы по изучению,
    "pronunciation_tips": советы по произношению,
    "common_mistakes": частые ошибки,
    "cultural_notes": культурный контекст,
    "difficulty_score": 1-10 (сложность для ученика),
    "next_steps": что учить дальше
}}

Особенности для уровня HSK {user_level}:
- Объясняй на {self._get_explanation_level(user_level)} уровне
- Используй примеры из HSK {user_level}
- Подчеркивай грамматические конструкции уровня {user_level}

Стиль обучения {learning_style}:
{self._get_learning_style_tips(learning_style)}

Будь детальным, но понятным!"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Парсим JSON ответ
            result = json.loads(response.choices[0].message.content)
            
            # Добавляем дополнительные вычисления
            result["characters_count"] = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            result["words_count"] = len(text.split())
            result["hsk_level_appropriate"] = user_level >= self._estimate_hsk_level(text)
            
            # Кэшируем результат
            self.translation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "original": text,
                "translation": "Ошибка перевода",
                "word_by_word": "",
                "pinyin": "",
                "grammar_explanation": "",
                "key_words": [],
                "example_sentences": [],
                "study_tips": "Попробуйте позже",
                "pronunciation_tips": "",
                "common_mistakes": "",
                "cultural_notes": "",
                "difficulty_score": 5,
                "next_steps": "Повторите базовые слова"
            }
    
    def _get_explanation_level(self, level: int) -> str:
        """Определяем уровень объяснений"""
        levels = {
            1: "очень простом, используй только базовые термины",
            2: "простом, минимальная грамматическая терминология", 
            3: "доступном, с простыми объяснениями грамматики",
            4: "подробном, с грамматическими терминами",
            5: "углубленном, с лингвистическими деталями",
            6: "экспертном, со сложными лингвистическими концепциями"
        }
        return levels.get(level, "доступном")
    
    def _get_learning_style_tips(self, style: str) -> str:
        """Советы по стилю обучения"""
        tips = {
            "visual": "• Используй визуальные аналогии\n• Рисуй ментальные карты\n• Цветовое кодирование иероглифов",
            "auditory": "• Фокусируйся на произношении\n• Используй ритм и рифмы\n• Придумывай песни",
            "kinesthetic": "• Связывай слова с движениями\n• Предлагай писать иероглифы\n• Используй жесты"
        }
        return tips.get(style, "")
    
    def _estimate_hsk_level(self, text: str) -> int:
        """Оценка уровня HSK текста"""
        # Простая эвристика: считаем количество сложных иероглифов
        simple_chars = set("的一是不人在有我他个大中要以会上们为子")  # HSK 1-2
        complex_chars = set(text) - simple_chars
        
        if len(complex_chars) > 10:
            return 4
        elif len(complex_chars) > 5:
            return 3
        elif len(complex_chars) > 2:
            return 2
        else:
            return 1
    
    async def analyze_pronunciation(self, text: str) -> Dict:
        """Анализ произношения текста"""
        try:
            system_prompt = """Ты — эксперт по китайскому произношению. Проанализируй текст и дай советы по произношению.
            
            Верни JSON:
            {
                "text": оригинальный текст,
                "pinyin": полная пиньинь,
                "tones": анализ тонов,
                "difficult_sounds": сложные звуки,
                "pronunciation_tips": советы,
                "common_errors": частые ошибки русскоговорящих,
                "practice_exercises": упражнения,
                "audio_advice": как работать с аудио
            }"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}
    
    async def generate_exercises(self, text: str, level: int = 1) -> Dict:
        """Генерация упражнений на основе текста"""
        try:
            system_prompt = f"""Создай упражнения для текста уровня HSK {level}.
            
            Текст: "{text}"
            
            Верни JSON:
            {{
                "fill_in_blanks": упражнение на заполнение пропусков,
                "matching": упражнение на сопоставление,
                "word_order": упражнение на порядок слов,
                "translation_exercise": упражнение на перевод,
                "writing_practice": упражнение на письмо,
                "conversation_topics": темы для разговора
            }}"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}

# Глобальный экземпляр переводчика
translator = SmartChineseTranslator()