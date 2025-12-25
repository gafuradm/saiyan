from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional
import re

load_dotenv()

class SmartChineseTranslator:
    """Smart Translator with Learning Capabilities"""
    
    def __init__(self):
        self.client = self._get_client()
        self.translation_cache = {}
        
    def _get_client(self):
        """Create a client for DeepSeek"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found in .env file")
        
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    async def smart_translate(self, text: str, user_level: int = 1, learning_style: str = "visual") -> Dict:
        """
        Smart translation with explanations
        
        Args:
            text: Text to translate
            user_level: User's HSK level
            learning_style: visual, auditory, kinesthetic
        
        Returns:
            Dictionary with translation and explanations
        """
        # Check cache
        cache_key = f"{text}_{user_level}_{learning_style}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Form AI prompt
            system_prompt = f"""You are an expert in Chinese language and pedagogy. Your task is not just to translate text, but to TEACH.

User: HSK level {user_level}, learning style: {learning_style}
Text to translate: "{text}"

Return the response in JSON format with these fields:
{{
    "original": original text,
    "translation": literary translation,
    "word_by_word": word-for-word translation,
    "pinyin": text pinyin,
    "grammar_explanation": grammar explanation,
    "key_words": [
        {{
            "character": character,
            "pinyin": pinyin,
            "translation": translation,
            "explanation": explanation (etymology, mnemonics),
            "memory_tip": memorization tip,
            "hsk_level": HSK level
        }}
    ],
    "example_sentences": [
        {{
            "chinese": example in Chinese,
            "pinyin": example pinyin,
            "translation": example translation,
            "explanation": why this is a good example
        }}
    ],
    "study_tips": study tips,
    "pronunciation_tips": pronunciation tips,
    "common_mistakes": common mistakes,
    "cultural_notes": cultural context,
    "difficulty_score": 1-10 (difficulty for student),
    "next_steps": what to learn next
}}

Features for HSK level {user_level}:
- Explain at {self._get_explanation_level(user_level)} level
- Use examples from HSK {user_level}
- Emphasize grammar structures of level {user_level}

Learning style {learning_style}:
{self._get_learning_style_tips(learning_style)}

Be detailed but clear!"""
            
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
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Add additional calculations
            result["characters_count"] = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            result["words_count"] = len(text.split())
            result["hsk_level_appropriate"] = user_level >= self._estimate_hsk_level(text)
            
            # Cache the result
            self.translation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "original": text,
                "translation": "Translation error",
                "word_by_word": "",
                "pinyin": "",
                "grammar_explanation": "",
                "key_words": [],
                "example_sentences": [],
                "study_tips": "Please try again later",
                "pronunciation_tips": "",
                "common_mistakes": "",
                "cultural_notes": "",
                "difficulty_score": 5,
                "next_steps": "Review basic vocabulary"
            }
    
    def _get_explanation_level(self, level: int) -> str:
        """Determine explanation level"""
        levels = {
            1: "very simple, use only basic terms",
            2: "simple, minimal grammar terminology", 
            3: "accessible, with simple grammar explanations",
            4: "detailed, with grammar terminology",
            5: "in-depth, with linguistic details",
            6: "expert, with complex linguistic concepts"
        }
        return levels.get(level, "accessible")
    
    def _get_learning_style_tips(self, style: str) -> str:
        """Learning style tips"""
        tips = {
            "visual": "• Use visual analogies\n• Draw mind maps\n• Color code characters",
            "auditory": "• Focus on pronunciation\n• Use rhythm and rhymes\n• Create songs",
            "kinesthetic": "• Connect words with movements\n• Suggest writing characters\n• Use gestures"
        }
        return tips.get(style, "")
    
    def _estimate_hsk_level(self, text: str) -> int:
        """Estimate text's HSK level"""
        # Simple heuristic: count complex characters
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
        """Analyze text pronunciation"""
        try:
            system_prompt = """You are an expert in Chinese pronunciation. Analyze the text and provide pronunciation tips.
            
            Return JSON:
            {
                "text": original text,
                "pinyin": full pinyin,
                "tones": tone analysis,
                "difficult_sounds": difficult sounds,
                "pronunciation_tips": tips,
                "common_errors": common errors for Russian speakers,
                "practice_exercises": exercises,
                "audio_advice": how to work with audio
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
        """Generate exercises based on text"""
        try:
            system_prompt = f"""Create exercises for text at HSK level {level}.
            
            Text: "{text}"
            
            Return JSON:
            {{
                "fill_in_blanks": fill-in-the-blank exercise,
                "matching": matching exercise,
                "word_order": word order exercise,
                "translation_exercise": translation exercise,
                "writing_practice": writing practice,
                "conversation_topics": conversation topics
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

# Global translator instance
translator = SmartChineseTranslator()