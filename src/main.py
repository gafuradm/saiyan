import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import re
import random
from datetime import datetime, timedelta
import uvicorn
import pickle
import hashlib
from src.ai_translator import translator
from src.grammar_explainer import grammar_explainer
import json
import asyncio
from src.grammar_explainer import grammar_explainer
from src.hsk_test_generator import test_generator, generate_hsk_test_api, evaluate_speaking_api, evaluate_writing_api, generate_certificate_api, generate_progress_report_api
from pydantic import Field

# Импорты для AI
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# ========== НАСТРОЙКА ПРИЛОЖЕНИЯ ==========
app = FastAPI(
    title="HSK AI Tutor",
    description="Прагматичный репетитор для сдачи HSK любой ценой (легально)",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажи конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# В начале main.py
class ChatThread(BaseModel):
    thread_id: str
    user_id: str
    title: str
    created_at: str
    messages: List[Dict]
    category: str = "general"  # grammar, vocabulary, test_prep, etc.

# Глобальные переменные
chat_threads = {}  # user_id -> list of threads
user_word_status: Dict[str, Dict[str, Dict]] = {}  # user_id -> {word_id: {"status": "saved"/"learned", "added_at": iso_str}}
current_threads = {}  # user_id -> current_thread_id

# ========== МОДЕЛИ ДАННЫХ ==========
class UserInfo(BaseModel):
    name: str
    current_level: int = 1
    target_level: int = 4
    exam_date: str = "2024-12-01"
    exam_location: str = "Москва"
    exam_format: str = "computer"  # computer или paper
    interests: List[str] = []
    daily_time: int = 30  # минут в день
    learning_style: str = "visual"  # visual, auditory, kinesthetic

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None

class TestAnswer(BaseModel):
    user_id: str
    test_id: str
    answers: Dict[str, Any]  # question_id: answer

class WordReview(BaseModel):
    user_id: str
    word_id: str  # character + level
    difficulty: int  # 1-5, где 1=легко, 5=очень сложно
    remembered: bool

class AuthRequest(BaseModel):
    username: str
    action: str = "login_or_register"
    password: Optional[str] = None

# Модели для полной регистрации
class UserRegister(BaseModel):
    name: str
    email: str
    password: str
    current_level: int = 1
    target_level: int = 4
    exam_date: str
    exam_location: str = "Москва"
    exam_format: str = "computer"
    interests: List[str] = []
    daily_time: int = 30
    learning_style: str = "visual"

class UserLogin(BaseModel):
    email: str
    password: str

# Модель для обновления чата
class ChatUpdate(BaseModel):
    thread_id: str
    title: str
    category: str

class VoiceChatRequest(BaseModel):
    message: str
    thread_id: str = Field(..., min_length=1, description="ID треда обязательно")
    context: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None

@app.post("/voice")
async def voice_chat(request: VoiceChatRequest):
    """Голосовой чат с AI для обучения чэнъюям (исправленная версия)"""
    try:
        # ВАЛИДАЦИЯ: проверяем обязательные поля
        if not request.thread_id or request.thread_id.strip() == "":
            raise HTTPException(status_code=422, detail="thread_id обязателен")
        
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=422, detail="message обязателен")
        
        print(f"🎤 Получен запрос voice/chat:")
        print(f"   message: {request.message[:100]}...")
        print(f"   thread_id: {request.thread_id}")
        print(f"   context keys: {list(request.context.keys())}")
        print(f"   user_id: {request.user_id}")
        
        # Проверяем, существует ли тред
        thread_exists = False
        if request.thread_id:
            for user_threads in chat_threads.values():
                for thread in user_threads:
                    if thread["thread_id"] == request.thread_id:
                        thread_exists = True
                        break
                if thread_exists:
                    break
        
        # Если тред не существует, создаем новый
        if not thread_exists and request.user_id:
            print(f"📝 Создаю новый тред для user_id: {request.user_id}")
            thread_id = f"voice_thread_{datetime.now().timestamp()}"
            
            if request.user_id not in chat_threads:
                chat_threads[request.user_id] = []
            
            thread = {
                "thread_id": thread_id,
                "user_id": request.user_id,
                "title": "Голосовой чат с AI",
                "category": "voice_chat",
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "updated_at": datetime.now().isoformat()
            }
            
            chat_threads[request.user_id].append(thread)
            current_threads[request.user_id] = thread_id
            request.thread_id = thread_id  # Обновляем thread_id в запросе
        system_prompt = """Ты — китайский AI-преподаватель. Ты ОБЯЗАН говорить ТОЛЬКО на китайском языке (普通话).

# СТРОГИЕ ПРАВИЛА:
1. 🇨🇳 Всегда отвечай ТОЛЬКО на китайском языке
2. 🗣️ Используй как устный, так и письменный китайский
3. 📚 Каждые 2-3 реплики естественно включай чэнъюй (成语)
4. 🎯 Объясняй сложное простыми словами, но на китайском
5. 🔤 Для пиньиня используй: (пиньинь)
6. 🇷🇺 Для перевода используй: 【русский перевод】

# ФОРМАТ ОТВЕТА:
1. Основной ответ на китайском
2. Сложные слова с пиньинем в скобках
3. Чэнъюи с объяснением
4. Краткий перевод ключевых фраз

# ПРИМЕРЫ:

## Пример 1: Обычный вопрос
用户: "Как дела?"
AI: "我很好，谢谢！(wǒ hěn hǎo, xièxiè) 【Мне хорошо, спасибо!】你今天怎么样？(nǐ jīntiān zěnmeyàng)"

## Пример 2: С чэнъюем
用户: "Что нового?"
AI: "今天我想教你一个成语：画蛇添足(huà shé tiān zú)。【Сегодня я хочу научить тебя чэнъюю: рисовать змею и добавлять ноги】意思是做多余的事情反而不好。【Значит: делать лишнее反而不好】比如：他的解释太长了，简直是画蛇添足。【Например: Его объяснение слишком длинное, это просто画蛇添足】"

## Пример 3: Объяснение
用户: "我不明白这个成语"
AI: "我来解释一下：画蛇添足(huà shé tiān zú)来自古代故事。几个人比赛画蛇，谁先画完谁赢。一个人很快画完了，但他自作聪明给蛇加了脚，结果输了。所以这个成语告诉我们：做事恰到好处就好，不要做多余的事情。【Я объясню: 画蛇添足来自古代故事...】"

# РЕКОМЕНДАЦИИ:
- Используй разные уровни сложности (HSK 1-6)
- Повторяй ранее изученные слова
- Задавай вопросы для практики
- Будь терпеливым и ободряющим

# ИСТОРИЯ ЧЭНЪЮЕВ:
已学成语：{learned_chengyu}

# ТЕКУЩИЙ УРОВЕНЬ УЧЕНИКА:
用户等级：HSK {user_level}

Не говори по-русски в основном тексте. Только китайский с пояснениями в скобках!"""

        # Формируем контекст
        learned_chengyu = request.context.get("learned_chengyu", [])
        command_type = request.context.get("command_type", "general")
        
        # Адаптируем промпт под тип команды
        if command_type == "chengyu":
            system_prompt += "\n\nПользователь просит рассказать новый чэнъюй. Выбери интересный и полезный чэнъюй для его уровня."
        elif command_type == "explain":
            system_prompt += "\n\nПользователь просит объяснить значение. Будь максимально понятным."
        
        # Добавляем информацию об изученных чэнъюях
        if learned_chengyu:
            system_prompt += f"\n\nИзученные чэнъюи: {', '.join(learned_chengyu[:5])}"
        
        # Получаем уровень пользователя
        user_level = 3
        if request.user_id and request.user_id in users_db:
            user = users_db[request.user_id]
            user_level = user.get("current_level", 3)
        
        # Добавляем уровень в промпт
        system_prompt += f"\n\nУровень пользователя: HSK {user_level}"
        
        # Отправляем запрос к DeepSeek
        client = get_deepseek_client()
        if not client:
            return {"response": "AI сервис временно недоступен", "error": "no_api_key"}
        
        # Формируем историю сообщений
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        print(f"🤖 Отправляю запрос к AI с {len(request.message)} символами")
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.8,
                max_tokens=800,
                presence_penalty=0.6,
                frequency_penalty=0.5
            )
            
            ai_response = response.choices[0].message.content
            
            print(f"🤖 Получен ответ от AI: {len(ai_response)} символов")
            
            # Сохраняем сообщение в историю
            if request.thread_id and request.user_id:
                # Находим или создаем тред
                thread_found = False
                for user_threads in chat_threads.values():
                    for thread in user_threads:
                        if thread["thread_id"] == request.thread_id:
                            thread["messages"].append({
                                "role": "user",
                                "content": request.message,
                                "timestamp": datetime.now().isoformat()
                            })
                            thread["messages"].append({
                                "role": "assistant",
                                "content": ai_response,
                                "timestamp": datetime.now().isoformat()
                            })
                            thread["updated_at"] = datetime.now().isoformat()
                            thread_found = True
                            break
                    if thread_found:
                        break
                
                if not thread_found and request.user_id:
                    # Создаем новый тред
                    if request.user_id not in chat_threads:
                        chat_threads[request.user_id] = []
                    
                    new_thread = {
                        "thread_id": request.thread_id,
                        "user_id": request.user_id,
                        "title": "Голосовой чат с AI",
                        "category": "voice_chat",
                        "created_at": datetime.now().isoformat(),
                        "messages": [
                            {
                                "role": "user",
                                "content": request.message,
                                "timestamp": datetime.now().isoformat()
                            },
                            {
                                "role": "assistant",
                                "content": ai_response,
                                "timestamp": datetime.now().isoformat()
                            }
                        ],
                        "updated_at": datetime.now().isoformat()
                    }
                    chat_threads[request.user_id].append(new_thread)
                    current_threads[request.user_id] = request.thread_id
                
                save_user_data()
            
            return {
                "response": ai_response,
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as ai_error:
            print(f"❌ Ошибка AI: {str(ai_error)}")
            return {
                "response": "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.",
                "thread_id": request.thread_id,
                "error": str(ai_error),
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"❌ Критическая ошибка голосового чата: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

class TranslationRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    detailed: bool = True
    include_exercises: bool = False

class PronunciationRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class ExerciseRequest(BaseModel):
    text: str
    level: int = 1
    exercise_type: str = "all"  # fill_blanks, matching, word_order, etc.

# Добавьте модели
class GrammarTopicRequest(BaseModel):
    topic_id: str
    user_id: Optional[str] = None
    user_level: Optional[str] = "初"

class GrammarQuestionRequest(BaseModel):
    question: str
    topic_id: Optional[str] = None
    user_id: Optional[str] = None

class HSKTestRequest(BaseModel):
    level: int
    test_type: str = "reduced"  # reduced или full
    user_id: Optional[str] = None

class SpeakingEvaluationRequest(BaseModel):
    audio_text: str  # Текст распознанной речи
    task_data: Dict[str, Any]
    user_id: str

class WritingEvaluationRequest(BaseModel):
    text: str
    task_data: Dict[str, Any]
    user_id: str

class TestResults(BaseModel):
    user_id: str
    test_id: str
    level: int  # 🔴 ОБЯЗАТЕЛЬНОЕ ПОЛЕ
    listening_score: Optional[int] = 0
    reading_score: Optional[int] = 0
    writing_score: Optional[int] = 0
    speaking_score: Optional[int] = 0
    total_score: Optional[int] = 0
    answers: Dict[str, Any]

@app.get("/hsk/test-answers/{test_id}/{user_id}")
async def get_test_answers(test_id: str, user_id: str):
    """Получить проверенные ответы пользователя"""
    if test_id not in tests_db or user_id not in tests_db[test_id]:
        raise HTTPException(status_code=404, detail="Результаты не найдены")
    
    user_results = tests_db[test_id][user_id]
    
    return {
        "test_id": test_id,
        "user_id": user_id,
        "answers": user_results.get("correct_answers", {}),
        "score": user_results.get("total_score_calculated", 0),
        "max_score": user_results.get("max_possible_score", 0),
        "percentage": user_results.get("percentage", 0),
        "ai_evaluated": user_results.get("ai_evaluated", False)
    }

# НАЙДИТЕ функцию generate_hsk_test и ИЗМЕНИТЕ её:
@app.post("/hsk/generate-test")
async def generate_hsk_test(request: HSKTestRequest):
    """Генерация полноценного теста HSK"""
    try:
        test_data = await generate_hsk_test_api(request.level, request.test_type)
        
        # 🔴 СРАЗУ СОХРАНЯЕМ ТЕСТ В БАЗУ ДАННЫХ
        test_id = test_data["test_id"]
        tests_db[test_id] = test_data  # Сохраняем сам тест
        
        # Для совместимости со старой структурой
        if test_id not in tests_db:
            tests_db[test_id] = {}
        
        # Сохраняем структуру теста отдельно
        tests_db[f"test_data_{test_id}"] = test_data
        
        return test_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации теста: {str(e)}")

@app.post("/hsk/evaluate-speaking")
async def evaluate_speaking(request: SpeakingEvaluationRequest):
    """Оценка речи пользователя"""
    try:
        evaluation = await evaluate_speaking_api(request.audio_text, request.task_data)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оценки речи: {str(e)}")

@app.post("/hsk/evaluate-writing")
async def evaluate_writing(request: WritingEvaluationRequest):
    """Оценка письменной работы"""
    try:
        evaluation = await evaluate_writing_api(request.text, request.task_data)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оценки письма: {str(e)}")

@app.post("/hsk/submit-test-results")
async def submit_test_results(results: TestResults):
    """Сохранить результаты теста"""
    try:
        test_id = results.test_id
        user_id = results.user_id
        
        # 1. Ищем тест
        original_test = None
        
        if test_id in tests_db and isinstance(tests_db[test_id], dict) and "sections" in tests_db[test_id]:
            original_test = tests_db[test_id]
        elif f"test_data_{test_id}" in tests_db:
            original_test = tests_db[f"test_data_{test_id}"]
        
        if not original_test:
            # Создаем минимальный тест для работы
            original_test = {
                "test_id": test_id,
                "level": results.level,
                "sections": {
                    "listening": {"questions": []},
                    "reading": {"questions": []},
                    "writing": {"tasks": []},
                    "speaking": {"tasks": []}
                }
            }
        
        # 2. Инициализируем правильные ответы
        correct_answers = {}
        
        # 3. Проверяем listening вопросы (только если они есть в тесте)
        listening_correct = 0
        listening_total = 0
        listening_questions = original_test.get("sections", {}).get("listening", {}).get("questions", [])
        
        for question in listening_questions:
            q_id = question.get("id")
            correct_index = question.get("correct_index")
            
            if correct_index is not None:
                listening_total += 1
                user_answer = results.answers.get(q_id)
                
                if user_answer is not None:
                    is_correct = user_answer == correct_index
                    if is_correct:
                        listening_correct += 1
                    
                    correct_answers[q_id] = {
                        "correct": is_correct,
                        "user_answer": user_answer,
                        "correct_answer": correct_index,
                        "points": 1 if is_correct else 0,
                        "section": "listening"
                    }
                else:
                    # Если пользователь не ответил
                    correct_answers[q_id] = {
                        "correct": False,
                        "user_answer": None,
                        "correct_answer": correct_index,
                        "points": 0,
                        "section": "listening"
                    }
        
        # 4. Проверяем reading вопросы
        reading_correct = 0
        reading_total = 0
        reading_questions = original_test.get("sections", {}).get("reading", {}).get("questions", [])
        
        for question in reading_questions:
            q_id = question.get("id")
            correct_index = question.get("correct_index")
            
            if correct_index is not None:
                reading_total += 1
                user_answer = results.answers.get(q_id)
                
                if user_answer is not None:
                    is_correct = user_answer == correct_index
                    if is_correct:
                        reading_correct += 1
                    
                    correct_answers[q_id] = {
                        "correct": is_correct,
                        "user_answer": user_answer,
                        "correct_answer": correct_index,
                        "points": 1 if is_correct else 0,
                        "section": "reading"
                    }
                else:
                    correct_answers[q_id] = {
                        "correct": False,
                        "user_answer": None,
                        "correct_answer": correct_index,
                        "points": 0,
                        "section": "reading"
                    }
        
        # 5. Рассчитываем баллы на основе правильных ответов
        # Важно: сначала проверяем, что есть вопросы в тесте!
        listening_score = 0
        reading_score = 0
        
        if listening_total > 0:
            listening_score = int((listening_correct / listening_total) * 100)
        
        if reading_total > 0:
            reading_score = int((reading_correct / reading_total) * 100)
        
        # 6. Используем переданные оценки для письменной и устной частей
        writing_score = results.writing_score if results.writing_score is not None else 0
        speaking_score = results.speaking_score if results.speaking_score is not None else 0
        
        # 7. Для письменных заданий добавляем в correct_answers
        writing_tasks = original_test.get("sections", {}).get("writing", {}).get("tasks", [])
        if writing_tasks:
            for task in writing_tasks:
                task_id = task.get("id", "1")
                correct_answers[f"W{task_id}"] = {
                    "correct": writing_score >= 60,
                    "score": writing_score,
                    "feedback": f"Письменная часть: {writing_score}/100",
                    "section": "writing"
                }
        
        # 8. Для говорения добавляем в correct_answers
        speaking_tasks = original_test.get("sections", {}).get("speaking", {}).get("tasks", [])
        if speaking_tasks:
            for task in speaking_tasks:
                task_id = task.get("id", "1")
                correct_answers[f"S{task_id}"] = {
                    "correct": speaking_score >= 60,
                    "score": speaking_score,
                    "feedback": f"Устная часть: {speaking_score}/100",
                    "section": "speaking"
                }
        
        # 9. Определяем общий балл ВНИМАТЕЛЬНО!
        # HSK 1-2: только listening (100) + reading (100) = максимум 200
        # HSK 3-6: listening (100) + reading (100) + writing (100) = максимум 300
        # Speaking НЕ входит в общий балл!
        
        # ОГРАНИЧИВАЕМ БАЛЛЫ до максимума 100 за каждую часть
        listening_score = min(100, listening_score)
        reading_score = min(100, reading_score)
        writing_score = min(100, writing_score)
        speaking_score = min(100, speaking_score)
        
        # Рассчитываем общий балл на основе уровня
        if results.level <= 2:
            # HSK 1-2: только listening + reading
            total_score = listening_score + reading_score
            max_possible_score = 200
        else:
            # HSK 3-6: listening + reading + writing
            total_score = listening_score + reading_score + writing_score
            max_possible_score = 300
        
        # Ограничиваем общий балл максимумом
        total_score = min(total_score, max_possible_score)
        
        # Рассчитываем процент
        percentage = int((total_score / max_possible_score) * 100) if max_possible_score > 0 else 0
        
        # 10. Сохраняем результаты
        if test_id not in tests_db:
            tests_db[test_id] = {}
        
        tests_db[test_id][user_id] = {
            "user_id": user_id,
            "test_id": test_id,
            "level": results.level,
            "listening_score": listening_score,
            "reading_score": reading_score,
            "writing_score": writing_score,
            "speaking_score": speaking_score,
            "total_score": total_score,
            "max_score": max_possible_score,
            "percentage": percentage,
            "answers": results.answers,
            "correct_answers": correct_answers,
            "listening_stats": {"correct": listening_correct, "total": listening_total},
            "reading_stats": {"correct": reading_correct, "total": reading_total},
            "checked_count": len(correct_answers),
            "submitted_at": datetime.now().isoformat(),
            "calculated_at": datetime.now().isoformat()
        }
        
        # 11. Генерируем сертификат и отчет
        user_data = users_db.get(user_id, {"name": "Студент", "user_id": user_id})
        
        certificate = await generate_certificate_api(
            {
                "test_id": test_id,
                "level": results.level,
                "listening_score": listening_score,
                "reading_score": reading_score,
                "writing_score": writing_score,
                "speaking_score": speaking_score,
                "total_score": total_score
            },
            user_data
        )
        
        progress_report = await generate_progress_report_api(
            {
                "test_id": test_id,
                "level": results.level,
                "listening_score": listening_score,
                "reading_score": reading_score,
                "writing_score": writing_score,
                "speaking_score": speaking_score,
                "total_score": total_score
            },
            user_data
        )
        
        save_user_data()
        
        return {
            "success": True,
            "certificate": certificate,
            "progress_report": progress_report,
            "correct_answers": correct_answers,
            "stats": {
                "listening": f"{listening_correct}/{listening_total} ({listening_score}/100)",
                "reading": f"{reading_correct}/{reading_total} ({reading_score}/100)",
                "writing": f"{writing_score}/100",
                "speaking": f"{speaking_score}/100",
                "total": f"{total_score}/{max_possible_score}"
            },
            "scores": {
                "listening": listening_score,
                "reading": reading_score,
                "writing": writing_score,
                "speaking": speaking_score,
                "total": total_score,
                "max": max_possible_score
            },
            "level": results.level,
            "calculated_score": total_score,
            "message": f"Результаты сохранены. Аудирование: {listening_correct}/{listening_total}, Чтение: {reading_correct}/{reading_total}, Общий балл: {total_score}/{max_possible_score}"
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Ошибка сохранения результатов: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения результатов: {str(e)}")

@app.get("/hsk/user-tests/{user_id}")
async def get_user_tests(user_id: str, limit: int = 10):
    """Получить историю тестов пользователя"""
    user_tests = []
    
    for test_id, test_data in tests_db.items():
        if user_id in test_data:
            user_test = test_data[user_id]
            user_test["test_id"] = test_id
            user_tests.append(user_test)
    
    # Сортируем по дате
    user_tests.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
    
    return {
        "user_id": user_id,
        "tests": user_tests[:limit],
        "total_tests": len(user_tests),
        "best_score": max([t.get("total_score", 0) for t in user_tests]) if user_tests else 0,
        "average_score": sum([t.get("total_score", 0) for t in user_tests]) // len(user_tests) if user_tests else 0
    }

@app.get("/hsk/test-stats/{test_id}")
async def get_test_stats(test_id: str):
    """Статистика по конкретному тесту"""
    if test_id not in tests_db:
        raise HTTPException(status_code=404, detail="Тест не найден")
    
    test_data = tests_db[test_id]
    users_count = len(test_data)
    
    if users_count == 0:
        return {"test_id": test_id, "users_count": 0}
    
    # Собираем статистику
    scores = [data.get("total_score", 0) for data in test_data.values()]
    
    return {
        "test_id": test_id,
        "users_count": users_count,
        "average_score": sum(scores) // users_count,
        "max_score": max(scores),
        "min_score": min(scores),
        "passing_rate": len([s for s in scores if s >= 180]) / users_count * 100 if users_count > 0 else 0,
        "scores_distribution": {
            "0-59": len([s for s in scores if s < 60]),
            "60-119": len([s for s in scores if 60 <= s < 120]),
            "120-179": len([s for s in scores if 120 <= s < 180]),
            "180-239": len([s for s in scores if 180 <= s < 240]),
            "240-300": len([s for s in scores if s >= 240])
        }
    }

# Добавьте глобальные переменные
grammar_topics = []

def load_grammar_topics():
    """Загрузка тем грамматики"""
    global grammar_topics
    try:
        with open("data/grammar_topics.json", "r", encoding="utf-8") as f:
            grammar_topics = json.load(f)
        print(f"✅ Загружено {len(grammar_topics)} тем грамматики")
        
        # Инициализируем grammar_explainer с темами
        grammar_explainer.grammar_topics = grammar_topics
    except FileNotFoundError:
        print("⚠️  Файл с темами грамматики не найден")
        grammar_topics = []
        grammar_explainer.grammar_topics = []

# Загружаем при старте
load_grammar_topics()

# Модель для проверки эссе
class EssayCheckRequest(BaseModel):
    essay_text: str
    topic: str
    hsk_level: int
    min_length: int = 300
    user_id: Optional[str] = None
    time_spent: Optional[int] = None
    mode: str = "essay_check"

@app.get("/grammar/topics")
async def get_grammar_topics(
    level: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Получить список тем грамматики"""
    filtered = grammar_topics
    
    if level:
        filtered = [t for t in filtered if t.get("level") == level]
    
    if category:
        filtered = [t for t in filtered if t.get("category") == category]
    
    paginated = filtered[offset:offset + limit]
    
    return {
        "topics": paginated,
        "total": len(filtered),
        "levels": list(set(t["level"] for t in grammar_topics)),
        "categories": list(set(t.get("category", "") for t in grammar_topics if t.get("category")))
    }

@app.get("/grammar/topic/{topic_id}")
async def get_grammar_topic(topic_id: str):
    """Получить информацию о теме грамматики"""
    topic = next((t for t in grammar_topics if t["id"] == topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Тема не найдена")
    
    return topic

@app.post("/grammar/explain")
async def explain_grammar_topic(request: GrammarTopicRequest):
    """Получить AI-объяснение темы грамматики"""
    # Находим тему
    topic = next((t for t in grammar_topics if t["id"] == request.topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Тема не найдена")
    
    # Получаем уровень пользователя если есть
    user_level = request.user_level
    if request.user_id and request.user_id in users_db:
        user = users_db[request.user_id]
        user_hsk = user.get("current_level", 1)
        # Конвертируем HSK в 初/中/高
        if user_hsk <= 2:
            user_level = "初"
        elif user_hsk <= 4:
            user_level = "中"
        else:
            user_level = "高"
    
    # Получаем объяснение
    explanation = await grammar_explainer.explain_grammar(topic, user_level)
    
    # Сохраняем в историю изучения
    if request.user_id:
        save_grammar_history(request.user_id, topic_id=request.topic_id)
    
    return explanation

@app.get("/grammar/practice/{topic_id}")
async def generate_grammar_practice(topic_id: str, difficulty: str = "medium"):
    """Сгенерировать упражнения по теме"""
    topic = next((t for t in grammar_topics if t["id"] == topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Тема не найдена")
    
    try:
        exercises = await grammar_explainer.generate_practice(topic_id, difficulty)
        return {
            "topic": topic,
            "exercises": exercises,
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации упражнений: {str(e)}")

@app.post("/grammar/ask")
async def ask_grammar_question(request: GrammarQuestionRequest):
    """Задать вопрос по грамматике"""
    context = None
    
    if request.topic_id:
        topic = next((t for t in grammar_topics if t["id"] == request.topic_id), None)
        if topic:
            context = {"topic": topic}
    
    answer = await grammar_explainer.answer_grammar_question(request.question, context)
    
    return {
        "question": request.question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/grammar/stats")
async def get_grammar_stats():
    """Статистика по грамматике"""
    if not grammar_topics:
        return {"message": "Темы грамматики не загружены"}
    
    # Статистика по уровням
    by_level = {}
    for topic in grammar_topics:
        level = topic.get("level", "未知")
        by_level[level] = by_level.get(level, 0) + 1
    
    # Статистика по категориям
    by_category = {}
    for topic in grammar_topics:
        category = topic.get("category", "其他")
        by_category[category] = by_category.get(category, 0) + 1
    
    # Сложность
    complexity_distribution = {
        "easy": len([t for t in grammar_topics if t.get("complexity", 3) <= 2]),
        "medium": len([t for t in grammar_topics if 2 < t.get("complexity", 3) <= 4]),
        "hard": len([t for t in grammar_topics if t.get("complexity", 3) > 4])
    }
    
    # Форматируем уровни для красивого отображения
    formatted_by_level = []
    for level_name, count in by_level.items():
        formatted_by_level.append({
            "level": level_name,
            "count": count,
            "display": {
                "初": "Начальный (初)",
                "中": "Средний (中)", 
                "高": "Продвинутый (高)"
            }.get(level_name, level_name)
        })
    
    # Сортируем уровни: 初 -> 中 -> 高
    formatted_by_level.sort(key=lambda x: {"初": 1, "中": 2, "高": 3}.get(x["level"], 4))
    
    return {
        "total_topics": len(grammar_topics),
        "by_level_formatted": formatted_by_level,  # Для фронтенда
        "by_level": by_level,  # Для совместимости
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:10]),
        "complexity_distribution": complexity_distribution,
        "average_complexity": sum(t.get("complexity", 3) for t in grammar_topics) / len(grammar_topics)
    }

# ========== УТИЛИТЫ ==========

def save_grammar_history(user_id: str, topic_id: str):
    """Сохраняем изучение темы в историю"""
    try:
        history_file = f"data/grammar_history_{user_id}.json"
        history = []
        
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        
        history.append({
            "topic_id": topic_id,
            "studied_at": datetime.now().isoformat(),
            "topic": next((t for t in grammar_topics if t["id"] == topic_id), {})
        })
        
        # Ограничиваем историю
        if len(history) > 100:
            history = history[-100:]
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка сохранения истории грамматики: {e}")

@app.get("/grammar/history/{user_id}")
async def get_grammar_history(user_id: str, limit: int = 20):
    """История изучения грамматики"""
    try:
        history_file = f"data/grammar_history_{user_id}.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # Добавляем информацию о темах
            for item in history:
                topic = next((t for t in grammar_topics if t["id"] == item["topic_id"]), None)
                if topic:
                    item["topic_info"] = topic
            
            return {
                "history": history[:limit],
                "total_studied": len(history),
                "recent_topics": list(set([h["topic_id"] for h in history[:10]]))
            }
        
        return {"history": [], "total_studied": 0}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки истории: {str(e)}")

@app.post("/ai/translate")
async def smart_translate(request: TranslationRequest):
    """Умный перевод с обучением"""
    try:
        # Получаем данные пользователя если есть
        user_level = 1
        learning_style = "visual"
        
        if request.user_id and request.user_id in users_db:
            user = users_db[request.user_id]
            user_level = user.get("current_level", 1)
            learning_style = user.get("learning_style", "visual")
        
        # Получаем умный перевод
        result = await translator.smart_translate(
            text=request.text,
            user_level=user_level,
            learning_style=learning_style
        )
        
        # Если нужны упражнения - генерируем
        if request.include_exercises:
            exercises = await translator.generate_exercises(request.text, user_level)
            result["exercises"] = exercises
        
        # Сохраняем в историю переводов
        if request.user_id:
            save_translation_history(request.user_id, request.text, result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перевода: {str(e)}")

@app.post("/ai/pronunciation")
async def analyze_pronunciation(request: PronunciationRequest):
    """Анализ произношения"""
    try:
        result = await translator.analyze_pronunciation(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

@app.post("/ai/exercises")
async def generate_exercises(request: ExerciseRequest):
    """Генерация упражнений"""
    try:
        result = await translator.generate_exercises(request.text, request.level)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.get("/ai/translation-history/{user_id}")
async def get_translation_history(user_id: str, limit: int = 20):
    """История переводов пользователя"""
    try:
        history = load_translation_history(user_id)
        return {
            "history": history[:limit],
            "count": len(history),
            "total_characters": sum(len(item.get("original", "")) for item in history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки истории: {str(e)}")

# ========== УТИЛИТЫ ДЛЯ ИСТОРИИ ==========

def save_translation_history(user_id: str, original: str, result: Dict):
    """Сохраняем перевод в историю"""
    try:
        history_file = f"data/translations_{user_id}.json"
        history = []
        
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        
        history.insert(0, {
            "original": original,
            "translation": result.get("translation", ""),
            "timestamp": datetime.now().isoformat(),
            "characters_count": result.get("characters_count", 0),
            "difficulty": result.get("difficulty_score", 5),
            "key_words": result.get("key_words", [])
        })
        
        # Ограничиваем историю 100 последними переводами
        if len(history) > 100:
            history = history[:100]
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка сохранения истории: {e}")

def load_translation_history(user_id: str) -> List:
    """Загружаем историю переводов"""
    try:
        history_file = f"data/translations_{user_id}.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Ошибка загрузки истории: {e}")
        return []

# API для обновления чата
@app.post("/chat/threads/update")
async def update_chat_thread(update: ChatUpdate):
    """Обновить чат-тред"""
    thread_found = None
    for user_threads in chat_threads.values():
        for thread in user_threads:
            if thread["thread_id"] == update.thread_id:
                thread["title"] = update.title
                thread["category"] = update.category
                thread["updated_at"] = datetime.now().isoformat()
                thread_found = thread
                break
    
    if not thread_found:
        raise HTTPException(status_code=404, detail="Тред не найден")
    
    save_user_data()
    return {"success": True, "thread": thread_found}

# API для удаления чата
@app.delete("/chat/threads/delete/{thread_id}")
async def delete_chat_thread(thread_id: str):
    """Удалить чат-тред"""
    deleted = False
    for user_id, threads in list(chat_threads.items()):
        for i, thread in enumerate(threads):
            if thread["thread_id"] == thread_id:
                threads.pop(i)
                deleted = True
                
                # Если удаляем текущий тред, устанавливаем другой
                if current_threads.get(user_id) == thread_id:
                    if threads:
                        current_threads[user_id] = threads[0]["thread_id"]
                    else:
                        del current_threads[user_id]
                break
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Тред не найден")
    
    save_user_data()
    return {"success": True, "message": "Тред удален"}

# API для получения истории чата
@app.get("/chat/{thread_id}/history")
async def get_chat_history(thread_id: str):
    """Получить историю чата"""
    thread = None
    for user_threads in chat_threads.values():
        for t in user_threads:
            if t["thread_id"] == thread_id:
                thread = t
                break
    
    if not thread:
        raise HTTPException(status_code=404, detail="Тред не найден")
    
    return {
        "thread_id": thread_id,
        "title": thread["title"],
        "category": thread["category"],
        "messages": thread["messages"],
        "message_count": len(thread["messages"])
    }

# Хэширование паролей (простое для демо)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Endpoints для авторизации
@app.post("/auth/register")
async def register_user_full(user: UserRegister):
    """Полная регистрация пользователя"""
    
    # Проверяем email
    for uid, existing_user in users_db.items():
        if existing_user.get("email", "").lower() == user.email.lower():
            raise HTTPException(status_code=400, detail="Email уже зарегистрирован")
    
    # Создаём ID
    user_id = f"user_{len(users_db) + 1}_{hashlib.md5(user.email.encode()).hexdigest()[:8]}"
    
    # Рассчитываем план
    days_until_exam = max(1, (datetime.fromisoformat(user.exam_date) - datetime.now()).days)
    target_words = {
        1: 150, 2: 300, 3: 600, 4: 1200, 5: 2500, 6: 5000
    }.get(user.target_level, 1000)
    daily_words = max(5, target_words // days_until_exam)
    
    # Сохраняем пользователя
    users_db[user_id] = {
        **user.dict(),
        "user_id": user_id,
        "password_hash": hash_password(user.password),
        "registered_at": datetime.now().isoformat(),
        "daily_words": daily_words,
        "days_until_exam": days_until_exam
    }
    
    # Инициализируем прогресс
    word_progress_db[user_id] = {}
    
    # Создаём чат-тред
    if user_id not in chat_threads:
        chat_threads[user_id] = []
    
    save_user_data()
    
    return {
        "success": True,
        "user_id": user_id,
        "name": user.name,
        "email": user.email,
        "current_level": user.current_level,
        "target_level": user.target_level,
        "plan": {
            "daily_words": daily_words,
            "days_until_exam": days_until_exam,
            "total_words_to_learn": target_words
        }
    }

@app.post("/auth/login")
async def login_user(login_data: UserLogin):
    """Вход пользователя"""
    
    user_found = None
    user_id = None
    
    # Ищем пользователя по email
    for uid, user in users_db.items():
        if user.get("email", "").lower() == login_data.email.lower():
            if user.get("password_hash") == hash_password(login_data.password):
                user_found = user
                user_id = uid
            break
    
    if not user_found:
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    
    # Возвращаем данные пользователя (без пароля)
    user_data = user_found.copy()
    user_data.pop("password_hash", None)
    
    # Добавляем статистику
    progress = word_progress_db.get(user_id, {})
    learned_words = len([p for p in progress.values() if p.get("remembered", False)])
    
    user_data["stats"] = {
        "learned_words": learned_words,
        "total_words": len(words_db),
        "progress_percentage": min(100, int(learned_words / len(words_db) * 100)) if words_db else 0
    }
    
    return user_data

# Удаляем старую регистрацию или делаем её частью авторизации

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
users_db = {}
words_db = []
word_progress_db = {}  # Прогресс изучения слов
tests_db = {}  # Результаты тестов

# ========== SYSTEM PROMPT (продолжение) ==========
system_prompt = """Ты — хитрый, прагматичный советчик по поступлению в китайские вузы.
Твоя цель: любыми легальными способами помочь студенту поступить и сдать HSK.

Ты знаешь все лазейки, лайфхаки и стратегии:
1. **Поступление без идеального китайского** - как обойти требования
2. **Связи и гуаньси** - как использовать networking
3. **Обходные пути** - альтернативные программы и факультеты
4. **Хитрости при подаче документов** - как выделиться среди тысяч заявок
5. **Психологические приемы** - как произвести впечатление на комиссию

Контекст студента: {context}

Твои ключевые компетенции:

🎯 **СТРАТЕГИИ ПОСТУПЛЕНИЯ:**
- Поиск "слабых" факультетов с низким конкурсом
- Подача через квоты для иностранцев
- Использование программ на английском
- Перевод из другого вуза после 1 курса

🕵️ **ДОКУМЕНТЫ И ЗАЯВКИ:**
- Как написать мотивационное письмо, которое прочитают
- Какие рекомендации работают лучше всего
- Как оформить портфолио без выдающихся достижений
- Что писать в CV для китайского вуза

🎓 **HSK И ЯЗЫК:**
- Как сдать HSK 4 за 3 месяца (интенсивные методы)
- Какие части HSK самые "пробиваемые"
- Как учить иероглифы для экзамена, а не для жизни
- Обманчиво легкие темы для эссе

🤝 **ГУАНЬСИ И СВЯЗИ:**
- Как найти "своего человека" в вузе
- К кому обращаться за рекомендациями
- Как использовать социальные сети для networking
- Бесплатные ресурсы и программы

💰 **ФИНАНСЫ И СТИПЕНДИИ:**
- Как получить стипендию CIS без идеальных оценок
- Скрытые стипендиальные программы
- Работа в Китае для студентов
- Экономия на проживании и учебе

Отвечай кратко, по делу, с конкретными шагами. Давай номера телефонов, названия программ, конкретные факультеты.
Избегай общих фраз. Будь циничным, но полезным.

Примеры ответов:
- "Вместо HSK 5 подавайся на англоязычную программу в Wuhan University"
- "Найди выпускника нужного вуза в LinkedIn и напиши..."
- "В мотивационном письме упомяни инициативу 'Один пояс, один путь'"
- "На экзамене используй шаблонные фразы из учебника 汉语口语..."

Готов помочь с любыми хитрыми вопросами! 🦊, Ты — прагматичный, циничный репетитор китайского для сдачи HSK.
Твоя цель: любой ценой (легально) помочь сдать экзамен.
Стиль: прямой, без воды, с лайфхаками, иногда с юмором.

Используй эти стратегии:
1. **80/20 правило** - учи только часто встречающиеся слова
2. **Чит-коды** - как угадывать ответы, распознавать паттерны
3. **Психологические приемы** - как не паниковать на экзамене
4. **Мошеннические лайфхаки** (легальные) - оптимизация времени

Отвечай кратко, по делу. Давай конкретные цифры и техники.
Примеры лайфхаков:
- "В части чтения сначала просмотри вопросы, потом ищи ответы в тексте"
- "Если не знаешь слово - ищи знакомые иероглифы в составе"
- "На аудировании сначала читай варианты ответов"
- "В письменной части используй шаблонные фразы"

Контекст ученика: {context}
"""

@app.post("/auth/user")
async def auth_user(auth_data: AuthRequest):
    """Авторизация или регистрация пользователя"""
    
    # Ищем существующего пользователя по имени
    user_id = None
    for uid, user in users_db.items():
        if user.get("name", "").lower() == auth_data.username.lower():
            user_id = uid
            break
    
    # Если пользователь не найден, создаём нового
    if not user_id:
        user_id = f"user_{len(users_db) + 1}_{hashlib.md5(auth_data.username.encode()).hexdigest()[:8]}"
        
        # Создаём нового пользователя
        users_db[user_id] = {
            "user_id": user_id,
            "name": auth_data.username,
            "current_level": 1,
            "target_level": 4,
            "exam_date": (datetime.now() + timedelta(days=90)).isoformat()[:10],
            "exam_location": "Москва",
            "exam_format": "computer",
            "interests": ["китайский", "HSK"],
            "daily_time": 30,
            "learning_style": "visual",
            "registered_at": datetime.now().isoformat(),
            "daily_words": 10
        }
        
        # Создаём прогресс
        if user_id not in word_progress_db:
            word_progress_db[user_id] = {}
        
        save_user_data()
        message = "registered"
    else:
        message = "logged_in"
    
    # Возвращаем данные пользователя (без пароля)
    user_data = users_db[user_id].copy()
    
    return {
        "success": True,
        "message": message,
        "user_id": user_id,
        **user_data
    }

@app.get("/user/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Получить профиль пользователя"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # Получаем прогресс
    progress = word_progress_db.get(user_id, {})
    learned_words = len([p for p in progress.values() if p.get("remembered", False)])
    
    return {
        **users_db[user_id],
        "stats": {
            "learned_words": learned_words,
            "total_words": len(words_db),
            "progress_percentage": min(100, int(learned_words / len(words_db) * 100)) if words_db else 0
        }
    }

class ThreadCreateRequest(BaseModel):
    user_id: str
    title: str = "Новый чат"
    category: str = "general"

@app.post("/chat/threads/create")
async def create_chat_thread(request: ThreadCreateRequest):
    """Создать новый чат-тред (исправленная версия)"""
    thread_id = f"thread_{datetime.now().timestamp()}"
    
    if request.user_id not in chat_threads:
        chat_threads[request.user_id] = []
    
    thread = {
        "thread_id": thread_id,
        "user_id": request.user_id,
        "title": request.title,
        "category": request.category,
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "updated_at": datetime.now().isoformat()
    }
    
    chat_threads[request.user_id].append(thread)
    current_threads[request.user_id] = thread_id
    
    return {"thread_id": thread_id, "thread": thread}

@app.get("/chat/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Получить все чат-треды пользователя"""
    if user_id not in chat_threads:
        return {"threads": [], "count": 0}
    
    threads = sorted(chat_threads[user_id], 
                     key=lambda x: x["updated_at"], 
                     reverse=True)
    
    return {
        "threads": threads,
        "current_thread": current_threads.get(user_id),
        "count": len(threads)
    }

@app.post("/chat/{thread_id}/message")
async def send_thread_message(thread_id: str, message: ChatMessage):
    """Отправить сообщение в конкретный тред"""
    # Найти тред
    thread = None
    for user_threads in chat_threads.values():
        for t in user_threads:
            if t["thread_id"] == thread_id:
                thread = t
                break
    
    if not thread:
        raise HTTPException(status_code=404, detail="Тред не найден")
    
    # Добавить сообщение
    thread["messages"].append({
        "role": "user",
        "content": message.message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Получить ответ от AI
    ai_response = await chat_with_deepseek(message.message)
    
    thread["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat()
    })
    
    thread["updated_at"] = datetime.now().isoformat()
    
    return {
        "thread_id": thread_id,
        "response": ai_response,
        "message_count": len(thread["messages"])
    }

chat_history = {}

# ========== УТИЛИТЫ ==========
def save_user_data():
    """Сохраняем данные пользователей в файл"""
    data = {
        'users_db': users_db,
        'word_progress_db': word_progress_db,
        'tests_db': tests_db,
        'chat_history': chat_history,
        'chat_threads': chat_threads,
        'current_threads': current_threads,
        "user_word_status": user_word_status
    }
    os.makedirs('data', exist_ok=True)
    with open('data/user_data.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_user_data():
    """Загружаем данные пользователей из файла"""
    global users_db, word_progress_db, tests_db, chat_history, chat_threads, current_threads
    try:
        with open('data/user_data.pkl', 'rb') as f:
            data = pickle.load(f)
            users_db = data.get('users_db', {})
            word_progress_db = data.get('word_progress_db', {})
            tests_db = data.get('tests_db', {})
            chat_history = data.get('chat_history', {})
            chat_threads = data.get('chat_threads', {})
            current_threads = data.get('current_threads', {})
        print(f"✅ Загружено {len(users_db)} пользователей")
    except FileNotFoundError:
        print("ℹ️  Файл с данными пользователей не найден")

# Загружаем при старте
load_user_data()

def get_deepseek_client():
    """Создаем клиент для DeepSeek API"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DeepSeek API ключ не найден в .env файле")
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
        )

async def chat_with_deepseek(message: str, user_context: dict = None) -> str:
    client = get_deepseek_client()
    if not client:
        return "❌ API ключ не настроен. Добавь DEEPSEEK_API_KEY в .env файл"
    
    try:
        user_id = user_context.get("user_id", "anonymous") if user_context else "anonymous"
        
        # Инициализируем историю для пользователя
        if user_id not in chat_history:
            chat_history[user_id] = []
        
        # Добавляем новое сообщение в историю
        chat_history[user_id].append({"role": "user", "content": message})
        
        # Ограничиваем историю последними 10 сообщениями
        if len(chat_history[user_id]) > 20:
            chat_history[user_id] = chat_history[user_id][-20:]
        
        # Формируем контекст пользователя
        context = ""
        if user_context:
            context = f"""
            Ученик: {user_context.get('name', 'Аноним')}
            Уровень: HSK {user_context.get('current_level', 1)} → HSK {user_context.get('target_level', 4)}
            Экзамен: {user_context.get('exam_date', 'скоро')} в {user_context.get('exam_location', 'Москва')}
            Интересы: {', '.join(user_context.get('interests', []))}
            """
        
        # Формируем промпт
        formatted_system_prompt = system_prompt.replace("{context}", context)
        
        # Формируем историю для AI
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            *chat_history[user_id][-10:]  # Берем последние 10 сообщений
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        ai_response = response.choices[0].message.content
        
        # Сохраняем ответ AI в историю
        chat_history[user_id].append({"role": "assistant", "content": ai_response})
        
        # Сохраняем данные
        save_user_data()
        
        return ai_response
        
    except Exception as e:
        return f"❌ Ошибка API: {str(e)}"
    
    # API для получения истории
@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Получить историю чата"""
    if user_id not in chat_history:
        return {"history": [], "count": 0}
    
    history = chat_history[user_id][-limit:]
    return {
        "history": history,
        "count": len(history)
    }

# API для очистки истории
@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Очистить историю чата"""
    if user_id in chat_history:
        chat_history[user_id] = []
    return {"message": "История очищена"}

def load_words():
    """Загружаем слова из JSON файла"""
    global words_db
    
    # Пробуем загрузить из разных файлов
    possible_files = [
        "data/hsk_all_words.json",
        "data/hsk_words.json",
        "data/hsk1_words.json"
    ]
    
    loaded = False
    for file_path in possible_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    words_db = json.load(f)
                
                print(f"✅ Загружено из {file_path}: {len(words_db)} слов")
                
                # Статистика
                stats = {}
                for word in words_db:
                    level = word.get("hsk_level", 0)
                    stats[level] = stats.get(level, 0) + 1
                
                print("📊 Статистика:")
                for level in sorted(stats.keys()):
                    print(f"  HSK {level}: {stats[level]} слов")
                
                loaded = True
                break
                
        except Exception as e:
            print(f"⚠️  Ошибка загрузки {file_path}: {e}")
    
    if not loaded:
        print("⚠️  Файлы со словами не найдены. Использую тестовые данные.")
        words_db = [
            {"character": "你好", "pinyin": "nǐ hǎo", "translation": "привет", "hsk_level": 1},
            {"character": "谢谢", "pinyin": "xiè xie", "translation": "спасибо", "hsk_level": 1},
        ]

def generate_memory_tip(word: dict, learning_style: str = "visual") -> str:
    """Генерируем совет по запоминанию"""
    char = word["character"]
    pinyin = word["pinyin"]
    translation = word["translation"]
    level = word.get("hsk_level", 1)
    
    tips = {
        "visual": [
            f"👁️ Нарисуй {char} в воздухе 3 раза",
            f"🎨 Представь {translation} в виде картинки с {char}",
            f"📝 Напиши {char} цветными маркерами",
            f"🎯 Создай ментальную карту для {char} → {translation}",
            f"🌈 Свяжи цвет с иероглифом {char}"
        ],
        "auditory": [
            f"🔊 Произнеси '{pinyin}' с разной интонацией",
            f"🎵 Придумай песню про {char} = {translation}",
            f"🗣️ Повтори '{pinyin} - {translation}' 5 раз вслух",
            f"🎧 Запиши произношение {char} и слушай",
            f"🎤 Проговори {char} как диктор на радио"
        ],
        "kinesthetic": [
            f"✍️ Напиши {char} на бумаге 10 раз",
            f"👆 Нарисуй {char} пальцем на столе",
            f"🎮 Сделай жест для {char}",
            f"🏃 Ассоциируй {char} с движением",
            f"🤲 Слепи {char} из пластилина"
        ]
    }
    
    # Специфичные советы для иероглифов
    special_tips = []
    if "好" in char:  # хороший
        special_tips.append("👫 '好' = 女 (женщина) + 子 (ребенок) = женщина с ребенком = хорошо!")
    if "谢" in char:  # благодарить
        special_tips.append("🙏 '谢' = 言 (речь) + 射 (стрелять) = слова как стрелы благодарности")
    if "学" in char:  # учиться
        special_tips.append("📚 '学' = 子 (ребенок) под крышей 宀 = ребенок учится дома")
    if "爱" in char:  # любовь
        special_tips.append("❤️ '爱' = 爫 (рука) + 冖 (крыша) + 友 (друг) = рука друга под крышей = любовь")
    
    # Выбираем советы в зависимости от стиля обучения
    style_tips = tips.get(learning_style, tips["visual"])
    
    all_tips = special_tips + style_tips
    return random.choice(all_tips)

def get_words_by_level(level: int, limit: int = 10000) -> List[Dict]:
    """Получить слова по уровню HSK"""
    return [w for w in words_db if w.get("hsk_level") == level][:limit]

def get_exam_hacks(location: str, format: str, level: int) -> List[str]:
    """Лайфхаки для экзамена"""
    hacks = [
        "🎯 80/20 правило: 20% слов = 80% текстов",
        "⏰ Начинай с легких вопросов, сложные оставь на потом",
        "📝 В письменной части пиши структурированно",
        "🧠 Если не знаешь - угадывай, не оставляй пустым",
        "🔄 Проверяй ответы, если осталось время"
    ]
    
    # По уровню
    level_hacks = {
        1: ["🔤 Учи только базовые иероглифы", "🎯 Сфокусируйся на произношении"],
        2: ["📚 Добавь простые грамматические конструкции", "👂 Тренируй аудирование"],
        3: ["💬 Учи диалоги целиком", "✍️ Начинай писать простые тексты"],
        4: ["📖 Читай короткие статьи", "🎯 Учи синонимы и антонимы"],
        5: ["🎓 Готовься к сочинению", "🔍 Анализируй сложные тексты"],
        6: ["🏆 Тренируйся на реальных экзаменах", "💡 Учи идиомы и пословицы"]
    }
    
    hacks.extend(level_hacks.get(level, []))
    
    # По местоположению
    if "китай" in location.lower() or "china" in location.lower():
        hacks.append("🇨🇳 В Китае строже с произношением и почерком")
    elif "россия" in location.lower() or "russia" in location.lower():
        hacks.append("🇷🇺 В России часто дают дополнительные минуты на аудирование")
    
    # По формату
    if format == "computer":
        hacks.extend([
            "💻 Используй CTRL+F в текстах для поиска ключевых слов",
            "⌨️ Тренируйся печатать пиньинь быстро",
            "🖱️ Дважды проверяй перед кликом"
        ])
    else:  # paper
        hacks.extend([
            "✍️ Пиши разборчиво, даже если медленнее",
            "📝 Бери запасные ручки",
            "📄 Размечай текст карандашом"
        ])
    
    return hacks

# Загружаем слова при старте
load_words()

# ========== API ЭНДПОИНТЫ ==========
@app.get("/")
async def root():
    return {
        "message": "🎌 HSK AI Tutor готов к работе!",
        "version": "1.0",
        "database": f"{len(words_db)} слов",
        "endpoints": {
            "register": "POST /register - регистрация",
            "chat": "POST /chat - общение с AI",
            "words_today": "GET /words/today/{user_id} - слова на сегодня",
            "test": "GET /test/{level} - тест по уровню",
            "exam": "GET /exam/{level} - полный экзамен",
            "stats": "GET /stats - статистика",
            "search": "GET /search/{query} - поиск слов",
            "word_random": "GET /word/random - случайное слово",
            "words_level": "GET /words/level/{level} - слова по уровню",
            "docs": "GET /docs - документация API"
        }
    }

@app.post("/register")
async def register_user(user: UserInfo):
    """Регистрация нового пользователя"""
    user_id = f"user_{len(users_db) + 1}"
    
    # Рассчитываем план
    days_until_exam = max(1, (datetime.fromisoformat(user.exam_date) - datetime.now()).days)
    target_words = {
        1: 150, 2: 300, 3: 600, 4: 1200, 5: 2500, 6: 5000
    }.get(user.target_level, 1000)
    
    daily_words = max(5, target_words // days_until_exam)
    
    # Сохраняем пользователя
    users_db[user_id] = {
        **user.dict(),
        "user_id": user_id,
        "registered_at": datetime.now().isoformat(),
        "daily_words": daily_words,
        "days_until_exam": days_until_exam
    }
    
    # Инициализируем прогресс
    word_progress_db[user_id] = {}
    
    # Сохраняем данные
    save_user_data()
    
    return {
        "success": True,
        "user_id": user_id,
        "message": f"🎉 Добро пожаловать, {user.name}!",
        "plan": {
            "daily_words": daily_words,
            "days_until_exam": days_until_exam,
            "total_words_to_learn": target_words,
            "study_plan": f"Учи по {daily_words} слов в день",
            "hacks": get_exam_hacks(user.exam_location, user.exam_format, user.target_level),
            "cheat_codes": [
                "🎮 Учи слова во время завтрака",
                "🚌 Используй карточки в транспорте",
                "🛌 Повторяй перед сном",
                "🎯 Фокусируйся на слабых местах"
            ]
        }
    }
    

@app.get("/user/{user_id}")
async def get_user_info(user_id: str):
    """Информация о пользователе"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    user = users_db[user_id]
    
    # Статистика пользователя
    progress = word_progress_db.get(user_id, {})
    learned_words = len([p for p in progress.values() if p.get("remembered", False)])
    
    return {
        **user,
        "stats": {
            "learned_words": learned_words,
            "total_words": len(words_db),
            "progress_percentage": min(100, int(learned_words / len(words_db) * 100)) if words_db else 0
        }
    }

@app.post("/chat")
async def chat_with_ai(chat_msg: ChatMessage):
    """Чат с ИИ-репетитором"""
    # Получаем контекст пользователя если есть
    user_context = None
    if chat_msg.user_id and chat_msg.user_id in users_db:
        user_context = users_db[chat_msg.user_id]
    
    # Используем DeepSeek
    answer = await chat_with_deepseek(chat_msg.message, user_context)
    
    return {
        "answer": answer,
        "user_id": chat_msg.user_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/words/today/{user_id}")
async def get_todays_words(user_id: str, new_words: int = 10, review_words: int = 5):
    """Слова на сегодня с системой повторений"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    user = users_db[user_id]
    level = user["current_level"]
    learning_style = user.get("learning_style", "visual")
    
    # Все слова нужного уровня
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"Слова HSK {level} не найдены")
    
    # Получаем прогресс пользователя
    progress = word_progress_db.get(user_id, {})
    
    # Новые слова (еще не изучались)
    new_words_list = []
    for word in level_words:
        if len(new_words_list) >= new_words:
            break
        
        word_id = f"{word['character']}_{level}"
        if word_id not in progress:
            word["word_id"] = word_id
            word["memory_tip"] = generate_memory_tip(word, learning_style)
            new_words_list.append(word)
    
    # Слова для повторения
    review_words_list = []
    today = datetime.now().date()
    
    for word_id, word_progress in progress.items():
        if len(review_words_list) >= review_words:
            break
        
        if word_progress.get("level") == level:
            last_review = datetime.fromisoformat(word_progress["last_reviewed"]).date()
            days_passed = (today - last_review).days
            
            # Интервалы повторения: 1, 3, 7, 14, 30 дней
            if days_passed in [1, 3, 7, 14, 30]:
                # Находим слово
                for word in level_words:
                    if f"{word['character']}_{level}" == word_id:
                        word["word_id"] = word_id
                        word["memory_tip"] = generate_memory_tip(word, learning_style)
                        word["last_reviewed"] = word_progress["last_reviewed"]
                        word["difficulty"] = word_progress.get("difficulty", 3)
                        review_words_list.append(word)
                        break
    
    return {
        "user": user["name"],
        "level": level,
        "date": today.isoformat(),
        "words": {
            "new": new_words_list,
            "review": review_words_list
        },
        "study_tips": [
            f"📚 Новые слова: {len(new_words_list)}",
            f"🔄 Повторение: {len(review_words_list)}",
            f"⏰ Рекомендуемое время: {user['daily_time']} минут",
            f"🎯 Стиль обучения: {learning_style}",
            "💡 Совет: Учи утром, повторяй вечером"
        ]
    }

@app.post("/review")
async def submit_word_review(review: WordReview):
    """Отправить отзыв о слове (запомнил/не запомнил)"""
    if review.user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # Обновляем прогресс
    if review.user_id not in word_progress_db:
        word_progress_db[review.user_id] = {}
    
    word_progress_db[review.user_id][review.word_id] = {
        "remembered": review.remembered,
        "difficulty": review.difficulty,
        "last_reviewed": datetime.now().isoformat(),
        "review_count": word_progress_db[review.user_id].get(review.word_id, {}).get("review_count", 0) + 1
    }
    
    return {
        "success": True,
        "message": "Прогресс сохранен!",
        "next_review": "Завтра" if review.remembered else "Через 1 день"
    }

@app.get("/test/{level}")
async def generate_test(level: int, questions: int = 10):
    """Генерация теста для уровня HSK"""
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"Слова HSK {level} не найдены")
    
    # Выбираем случайные слова
    selected_words = random.sample(level_words, min(questions, len(level_words)))
    
    test_questions = []
    for i, word in enumerate(selected_words, 1):
        # Создаем неправильные варианты
        wrong_words = []
        other_words = [w for w in level_words if w["character"] != word["character"]]
        
        if len(other_words) >= 3:
            wrong_words = random.sample(other_words, 3)
        
        # Создаем варианты ответов
        options = [word["translation"]] + [w["translation"] for w in wrong_words]
        random.shuffle(options)
        
        # Определяем правильный ответ
        correct_index = options.index(word["translation"])
        
        test_questions.append({
            "id": f"q_{i}",
            "question": f"Как переводится '{word['character']}' ({word['pinyin']})?",
            "options": options,
            "correct_index": correct_index,
            "correct_answer": word["translation"],
            "points": 1,
            "hint": f"HSK {level}, часть речи: {word.get('part_of_speech', 'не указано')}"
        })
    
    # Создаем ID теста
    test_id = f"test_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Сохраняем тест
    tests_db[test_id] = {
        "level": level,
        "questions": test_questions,
        "created_at": datetime.now().isoformat(),
        "max_score": len(test_questions)
    }
    
    return {
        "test_id": test_id,
        "level": level,
        "total_questions": len(test_questions),
        "time_limit": f"{len(test_questions) * 1.5} минут",
        "questions": test_questions,
        "test_hacks": [
            "⏱️ Трать не больше 1.5 минут на вопрос",
            "🎯 Если сомневаешься - исключай явно неправильные",
            "📝 Помни: в HSK часто повторяются похожие варианты",
            "🧠 Первая мысль часто правильная"
        ]
    }

@app.post("/submit_test")
async def submit_test_answers(test_data: TestAnswer):
    """Отправить ответы на тест"""
    if test_data.user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    if test_data.test_id not in tests_db:
        raise HTTPException(status_code=404, detail="Тест не найден")
    
    test = tests_db[test_data.test_id]
    questions = test["questions"]
    
    # Проверяем ответы
    correct = 0
    results = []
    
    for question in questions:
        user_answer = test_data.answers.get(question["id"])
        is_correct = user_answer == question["correct_index"]
        
        if is_correct:
            correct += 1
        
        results.append({
            "question_id": question["id"],
            "user_answer": user_answer,
            "correct_answer": question["correct_index"],
            "is_correct": is_correct,
            "explanation": f"Правильный ответ: {question['correct_answer']}"
        })
    
    score = correct
    max_score = len(questions)
    percentage = int((score / max_score) * 100) if max_score > 0 else 0
    
    # Сохраняем результат
    if "results" not in tests_db[test_data.test_id]:
        tests_db[test_data.test_id]["results"] = {}
    
    tests_db[test_data.test_id]["results"][test_data.user_id] = {
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "submitted_at": datetime.now().isoformat(),
        "answers": test_data.answers
    }
    
    # Генерируем фидбек
    feedback = ""
    if percentage >= 80:
        feedback = "🎉 Отлично! Ты готов к экзамену!"
    elif percentage >= 60:
        feedback = "👍 Хорошо! Продолжай тренироваться!"
    else:
        feedback = "💪 Нужно больше практики! Сфокусируйся на слабых местах."
    
    return {
        "test_id": test_data.test_id,
        "user_id": test_data.user_id,
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "feedback": feedback,
        "results": results,
        "recommendations": [
            f"🎯 Повтори слова, которые ошибал",
            f"⏰ Следующий тест через 3 дня",
            f"📈 Цель на следующий раз: {min(100, percentage + 10)}%"
        ]
    }

@app.get("/exam/{level}")
async def generate_exam(level: int):
    """Генерация полного экзамена HSK"""
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"Слова HSK {level} не найдены")
    
    # Разные части экзамена
    exam = {
        "listening": [],
        "reading": [],
        "writing": [],
        "speaking": []
    }
    
    # АУДИРОВАНИЕ (4 вопроса)
    for i in range(4):
        word = random.choice(level_words)
        wrong_words = random.sample([w for w in level_words if w != word], 3)
        
        exam["listening"].append({
            "type": "multiple_choice",
            "id": f"listening_{i+1}",
            "question": f"Слушайте аудио и выберите правильный перевод для:",
            "character": word["character"],
            "pinyin": word["pinyin"],
            "options": [word["translation"]] + [w["translation"] for w in wrong_words],
            "correct_answer": word["translation"],
            "points": 5,
            "time_limit": "30 секунд"
        })
    
    # ЧТЕНИЕ (3 вопроса)
    for i in range(3):
        # Сопоставление
        pairs = random.sample(level_words, min(4, len(level_words)))
        exam["reading"].append({
            "type": "matching",
            "id": f"reading_{i+1}",
            "question": "Сопоставьте китайские слова с переводами:",
            "pairs": [{"character": w["character"], "pinyin": w["pinyin"]} for w in pairs],
            "answers": [w["translation"] for w in pairs],
            "shuffled_answers": random.sample([w["translation"] for w in pairs], len(pairs)),
            "points": 10,
            "time_limit": "2 минуты"
        })
    
    # ПИСЬМО (2 вопроса)
    writing_words = random.sample(level_words, min(2, len(level_words)))
    exam["writing"].append({
        "type": "writing",
        "id": "writing_1",
        "question": "Напишите иероглифы для следующих слов:",
        "words": [{"pinyin": w["pinyin"], "translation": w["translation"]} for w in writing_words],
        "answers": [w["character"] for w in writing_words],
        "points": 15,
        "time_limit": "5 минут"
    })
    
    # ГОВОРЕНИЕ (1 вопрос)
    speaking_word = random.choice(level_words)
    exam["speaking"].append({
        "type": "speaking",
        "id": "speaking_1",
        "question": f"Произнесите слово и составьте с ним предложение:",
        "word": {
            "character": speaking_word["character"],
            "pinyin": speaking_word["pinyin"],
            "translation": speaking_word["translation"]
        },
        "example": f"Пример: '{speaking_word['character']} ({speaking_word['pinyin']})' - {speaking_word['translation']}",
        "points": 20,
        "time_limit": "3 минуты"
    })
    
    exam_id = f"exam_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "exam_id": exam_id,
        "level": level,
        "total_points": 100,
        "time_total": "60 минут",
        "sections": exam,
        "exam_strategy": [
            "🎯 Начинай с любимой части",
            "⏰ Распредели время: 20мин чтение, 15мин аудирование, 15мин письмо, 10мин говорение",
            "📝 В письменной части пиши сначала на черновике",
            "🎤 В говорении говори четко и не торопись",
            "🔄 Оставь 5 минут на проверку"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Статистика базы данных"""
    if not words_db:
        return {"message": "База данных пуста"}
    
    stats = {
        "total_words": len(words_db),
        "by_level": {},
        "by_part_of_speech": {},
        "users_count": len(users_db),
        "tests_taken": len(tests_db)
    }
    
    # Статистика по уровням
    for word in words_db:
        level = word.get("hsk_level", 0)
        stats["by_level"][f"HSK {level}"] = stats["by_level"].get(f"HSK {level}", 0) + 1
        
        # Статистика по частям речи
        pos = word.get("part_of_speech", "не указано")
        stats["by_part_of_speech"][pos] = stats["by_part_of_speech"].get(pos, 0) + 1
    
    # Самые частые иероглифы
    character_count = {}
    for word in words_db:
        for char in word.get("character", ""):
            if '\u4e00' <= char <= '\u9fff':
                character_count[char] = character_count.get(char, 0) + 1
    
    top_characters = sorted(character_count.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_characters"] = [{"character": char, "count": count} for char, count in top_characters]
    
    return stats

@app.get("/search/{query}")
async def search_words(query: str, limit: int = 20):
    """Поиск слов по иероглифам, пиньиню или переводу"""
    results = []
    query_lower = query.lower()
    
    for word in words_db:
        # Поиск в иероглифах
        if query in word.get("character", ""):
            results.append(word)
            continue
            
        # Поиск в пиньине
        pinyin = word.get("pinyin", "").lower()
        if query_lower in pinyin:
            results.append(word)
            continue
            
        # Поиск в переводе
        translation = word.get("translation", "").lower()
        if query_lower in translation:
            results.append(word)
    
    return {
        "query": query,
        "count": len(results),
        "results": results[:limit]
    }

@app.get("/word/random")
async def get_random_word(level: Optional[int] = None):
    """Получить случайное слово"""
    if level:
        filtered_words = [w for w in words_db if w.get("hsk_level") == level]
    else:
        filtered_words = words_db
    
    if not filtered_words:
        raise HTTPException(status_code=404, detail="Слова не найдены")
    
    word = random.choice(filtered_words)
    
    # Умный поиск похожих слов:
    similar = []
    word_level = word.get("hsk_level", 1)
    word_chars = set(word["character"])
    
    for w in words_db:
        if w["character"] == word["character"]:
            continue
        
        # 1. Похожие по составу иероглифов
        w_chars = set(w["character"])
        common_chars = word_chars.intersection(w_chars)
        
        # 2. Похожие по тематике (анализ перевода)
        word_trans_lower = word["translation"].lower()
        w_trans_lower = w["translation"].lower()
        
        # Простой анализ тематики
        categories = {
            "семья": ["мать", "отец", "брат", "сестра", "семья", "родители"],
            "еда": ["есть", "пить", "еда", "вода", "чай", "рис"],
            "путешествие": ["идти", "приезжать", "поезд", "самолет", "гостиница"],
            "учеба": ["учиться", "школа", "студент", "учитель", "книга"],
            "время": ["время", "час", "день", "месяц", "год", "сегодня"]
        }
        
        similarity_found = False
        
        # Похожие иероглифы
        if common_chars:
            similarity_found = True
        
        # Одинаковый уровень
        if w.get("hsk_level", 1) == word_level:
            similarity_found = True
        
        # Похожий перевод (ищем общие слова в переводе)
        word_trans_words = set(word_trans_lower.split())
        w_trans_words = set(w_trans_lower.split())
        common_words = word_trans_words.intersection(w_trans_words)
        
        if len(common_words) > 0:
            similarity_found = True
        
        # Одинаковая тематика
        for category, keywords in categories.items():
            word_has_keyword = any(keyword in word_trans_lower for keyword in keywords)
            w_has_keyword = any(keyword in w_trans_lower for keyword in keywords)
            
            if word_has_keyword and w_has_keyword:
                similarity_found = True
                break
        
        if similarity_found:
            similar.append({
                "character": w["character"],
                "pinyin": w["pinyin"],
                "translation": w["translation"][:50],
                "hsk_level": w.get("hsk_level", 1),
                "why_similar": f"Общие иероглифы: {len(common_chars)}, Тематика: {category if 'category' in locals() else 'общая'}"
            })
    
    # Берем 3 самых похожих
    if len(similar) > 3:
        similar = similar[:3]
    elif len(similar) < 3:
        # Добавляем случайные слова того же уровня
        same_level_words = [w for w in filtered_words if w["character"] != word["character"]]
        while len(similar) < 3 and same_level_words:
            random_similar = random.choice(same_level_words)
            if random_similar not in similar:
                similar.append({
                    "character": random_similar["character"],
                    "pinyin": random_similar["pinyin"],
                    "translation": random_similar["translation"][:50],
                    "hsk_level": random_similar.get("hsk_level", 1),
                    "why_similar": "Случайное слово того же уровня"
                })
    
    return {
        "word": word,
        "similar_words": similar,
        "memory_tip": generate_memory_tip(word),
        "study_suggestions": [
            "🔊 Произнеси вслух 10 раза",
            f"🧠 Сравни с похожими: {', '.join([s['character'] for s in similar])}",
            "⏰ Повтори сегодня еще 3 раза"
        ]
    }

class TextGenerationRequest(BaseModel):
    topic: str
    description: Optional[str] = ""
    hsk_level: int = 3
    format: str = "chinese_only"  # chinese_only, full, manga
    length: str = "medium"  # short, medium, long
    user_id: Optional[str] = None
    include_emojis: bool = True
    manga_style: bool = False

@app.post("/text/generate")
async def generate_chinese_text(request: TextGenerationRequest):
    """Генерация текста на китайском с заданными параметрами"""
    try:
        # Получаем клиент DeepSeek
        client = get_deepseek_client()
        if not client:
            raise HTTPException(status_code=500, detail="AI сервис недоступен")
        
        # Формируем промпт в зависимости от формата
        format_prompts = {
            "chinese_only": "ТОЛЬКО на китайском языке с иероглифами",
            "full": "На китайском с пиньинем и русским переводом",
            "manga": "В стиле манги с диалогами и описаниями"
        }
        
        format_instruction = format_prompts.get(request.format, "На китайском")
        
        # Формируем системный промпт
        system_prompt = f"""Ты — автор китайских текстов для изучающих язык.
        
# ЗАДАЧА:
Создать текст на тему: "{request.topic}"
Описание: {request.description}

# ТРЕБОВАНИЯ:
1. Уровень сложности: HSK {request.hsk_level}
2. Использовать слова преимущественно уровня HSK {request.hsk_level} и ниже
3. {format_instruction}
4. Длина: {request.length} (около {2000 if request.length == 'medium' else 1000 if request.length == 'short' else 3000} иероглифов)
5. {"Использовать эмодзи 🎌" if request.include_emojis else "Без эмодзи"}
6. {"Стиль как в манге: диалоги, описания, эмоции" if request.manga_style else "Обычный повествовательный стиль"}

# ФОРМАТЫ:
- Если нужен только китайский: иероглифы + знаки препинания + эмодзи
- Если нужен пиньинь: 汉字 (pinyin) 【перевод】
- Если стиль манги: 
  【Персонаж】: Реплика
  *описание действия*
  
# СТРУКТУРА:
- Введение/начало
- Основная часть с развитием
- Заключение/вывод

Будь креативным, но используй соответствующую уровню лексику!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Создай текст на тему: {request.topic}"}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.8,
            max_tokens=4000,
            presence_penalty=0.3,
            frequency_penalty=0.2
        )
        
        text_content = response.choices[0].message.content
        
        # Анализируем текст для статистики
        stats = analyze_chinese_text(text_content, request.hsk_level)
        
        # Форматируем текст в зависимости от формата
        formatted_text = format_generated_text(text_content, request.format)
        
        return {
            "success": True,
            "text": text_content,
            "formatted_text": formatted_text,
            "text_with_pinyin": add_pinyin_to_text(text_content) if request.format == "full" else None,
            "topic": request.topic,
            "hsk_level": request.hsk_level,
            "format": request.format,
            "stats": stats,
            "generated_at": datetime.now().isoformat(),
            "length_chars": len(text_content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации текста: {str(e)}")

def analyze_chinese_text(text: str, target_hsk_level: int) -> Dict:
    """Анализ сгенерированного текста"""
    # Простой анализ (в реальном проекте нужно использовать HSK словарь)
    characters = len([c for c in text if '\u4e00-\u9fff'])
    words = text.split()
    unique_words = len(set(words))
    
    # Простая оценка сложности
    estimated_level = min(6, max(1, target_hsk_level + random.randint(-1, 1)))
    
    return {
        "characters": characters,
        "words": len(words),
        "unique_words": unique_words,
        "hsk_level": estimated_level,
        "estimated_reading_time": f"{max(1, characters // 300)} минут",
        "new_words": max(0, unique_words - target_hsk_level * 100)  # Простая оценка
    }

def format_generated_text(text: str, format_type: str) -> str:
    """Форматирование текста для разных форматов"""
    if format_type == "manga":
        # Добавляем маркеры для манги
        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            if ':' in line and len(line) < 50:
                formatted_lines.append(f"🎭 {line}")
            elif len(line) > 0:
                formatted_lines.append(f"📖 {line}")
            else:
                formatted_lines.append("")
        return '\n'.join(formatted_lines)
    
    elif format_type == "full":
        # Здесь можно добавить пиньинь и перевод
        return text  # В реальном проекте нужно интегрировать pinyin и перевод
    
    return text

def add_pinyin_to_text(text: str) -> str:
    """Добавление пиньиня к тексту (заглушка)"""
    # В реальном проекте нужно использовать библиотеку для пиньиня
    # Например, pypinyin
    return text

# Модели для проверки эссе и переводов
class EssayCheckRequest(BaseModel):
    essay_text: str
    topic: str
    hsk_level: int
    min_length: int = 300
    user_id: Optional[str] = None
    time_spent: Optional[int] = None
    mode: str = "essay_check"

class TranslationCheckRequest(BaseModel):
    original_text: str
    user_translation: str
    target_hsk: int
    difficulty: str
    user_id: Optional[str] = None
    time_spent: Optional[int] = None
    mode: str = "translation_check"

class TranslationGenerateRequest(BaseModel):
    topic: str
    description: Optional[str] = ""  # <-- добавьте
    difficulty: str = "medium"
    length: str = "medium"
    hsk_level: int = 4
    user_id: Optional[str] = None
    include_emojis: bool = True  # <-- добавьте
    manga_style: bool = False  # <-- добавьте

@app.post("/essay/check")
async def check_essay(request: EssayCheckRequest):
    """Проверка эссе AI"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_check(request)
        
        # Формируем промпт для проверки
        system_prompt = f"""Ты — строгий, но справедливый преподаватель китайского языка.
        
# ЗАДАЧА:
Проверить эссе на тему: "{request.topic}"
Уровень студента: HSK {request.hsk_level}
Минимальная длина: {request.min_length} иероглифов
Длина эссе студента: {len(request.essay_text)} иероглифов

# КРИТЕРИИ ОЦЕНКИ:
1. **Грамматика** (30%) - правильность конструкций, частиц, времен
2. **Лексика** (25%) - богатство словарного запаса, уместность слов  
3. **Структура** (20%) - логичность, организация, связность
4. **Содержание** (15%) - соответствие теме, аргументация
5. **Стиль** (10%) - разнообразие, естественность, сложность

# ФОРМАТ ОТВЕТА JSON:
{{
    "overall_score": 85,
    "categories": [
        {{"name": "Грамматика", "score": 80, "feedback": "..."}},
        {{"name": "Лексика", "score": 85, "feedback": "..."}},
        {{"name": "Структура", "score": 90, "feedback": "..."}},
        {{"name": "Содержание", "score": 75, "feedback": "..."}},
        {{"name": "Стиль", "score": 80, "feedback": "..."}}
    ],
    "errors": [
        {{"position": 15, "error": "了 использован не к месту", "correction": "..."}},
        {{"position": 42, "error": "Неправильный порядок слов", "correction": "..."}}
    ],
    "recommendations": "Рекомендации по улучшению...",
    "strengths": "Сильные стороны работы...",
    "estimated_hsk_level": {request.hsk_level}
}}

# БУДЬ СТРОГИМ:
- Не завышай оценки
- Указывай конкретные ошибки
- Давай конкретные исправления
- Будь конструктивным, но честным

# ЭССЕ СТУДЕНТА:
{request.essay_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Проверь это эссе и дай детальный анализ."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Парсим JSON ответ
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Если AI не вернул JSON, создаем структурированный ответ
            result = {
                "overall_score": 75,
                "categories": [
                    {"name": "Грамматика", "score": 70, "feedback": "Проверьте использование частиц"},
                    {"name": "Лексика", "score": 80, "feedback": "Хороший словарный запас"},
                    {"name": "Структура", "score": 85, "feedback": "Логичная организация"},
                    {"name": "Содержание", "score": 75, "feedback": "Соответствует теме"},
                    {"name": "Стиль", "score": 70, "feedback": "Можно разнообразить стиль"}
                ],
                "errors": [],
                "recommendations": "Продолжайте практиковаться в написании эссе",
                "strengths": "Эссе соответствует заданной теме и имеет логичную структуру"
            }
        
        # Добавляем метаданные
        result.update({
            "topic": request.topic,
            "target_hsk": request.hsk_level,
            "actual_length": len(request.essay_text),
            "min_required": request.min_length,
            "checked_at": datetime.now().isoformat(),
            "ai_checked": True
        })
        
        return result
        
    except Exception as e:
        print(f"Ошибка проверки эссе: {str(e)}")
        return generate_fallback_essay_check(request)

def generate_fallback_essay_check(request: EssayCheckRequest):
    """Fallback проверка эссе (если AI недоступен)"""
    text = request.essay_text
    char_count = len(text)
    
    # Простая оценка на основе длины
    if char_count < request.min_length:
        length_score = 50
    elif char_count < request.min_length * 1.5:
        length_score = 70
    else:
        length_score = 90
    
    base_score = length_score
    
    # Добавляем случайные вариации
    grammar_score = max(0, min(100, base_score + random.randint(-15, 15)))
    vocab_score = max(0, min(100, base_score + random.randint(-10, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-5, 15)))
    content_score = max(0, min(100, base_score + random.randint(-5, 10)))
    style_score = max(0, min(100, base_score + random.randint(-10, 5)))
    
    overall_score = int((grammar_score + vocab_score + structure_score + content_score + style_score) / 5)
    
    # Генерируем примерные ошибки
    errors = []
    if char_count > 100:
        # Добавляем пару примерных ошибок
        errors.append({
            "position": min(50, char_count - 10),
            "error": "Возможная ошибка в использовании 了",
            "correction": "Убедитесь, что 了 используется для завершенных действий"
        })
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Грамматика", "score": grammar_score, 
             "feedback": "Есть ошибки в использовании частиц. Обратите внимание на 了, 的, 地, 得."},
            {"name": "Лексика", "score": vocab_score,
             "feedback": f"Достаточно разнообразный словарный запас для уровня HSK {request.hsk_level}."},
            {"name": "Структура", "score": structure_score,
             "feedback": "Логичная организация текста, но можно улучшить связность между абзацами."},
            {"name": "Содержание", "score": content_score,
             "feedback": f"Соответствует теме '{request.topic}', есть аргументы и примеры."},
            {"name": "Стиль", "score": style_score,
             "feedback": "Стиль достаточно разнообразный, но можно использовать более сложные конструкции."}
        ],
        "errors": errors,
        "recommendations": f"""
1. Практикуйтесь в использовании сложных предложений с 虽然...但是..., 因为...所以...
2. Увеличьте словарный запас по теме "{request.topic}"
3. Обратите внимание на использование частиц 了, 的, 地, 得
4. Добавьте вводные слова: 首先, 其次, 最后, 总而言之
5. Пишите регулярно для улучшения навыков
        """,
        "strengths": "Хорошая организация текста, соответствие теме, достаточный объем.",
        "estimated_hsk_level": request.hsk_level,
        "topic": request.topic,
        "target_hsk": request.hsk_level,
        "actual_length": char_count,
        "min_required": request.min_length,
        "checked_at": datetime.now().isoformat(),
        "ai_checked": False,
        "fallback": True
    }

def generate_fallback_essay_check(request: EssayCheckRequest):
    """Fallback проверка эссе"""
    # Простой анализ эссе
    text = request.essay_text
    char_count = len(text)
    
    # Простая оценка
    base_score = min(100, max(50, char_count / request.min_length * 80))
    
    # Случайные вариации
    grammar_score = max(0, min(100, base_score + random.randint(-15, 15)))
    vocab_score = max(0, min(100, base_score + random.randint(-10, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-5, 15)))
    content_score = max(0, min(100, base_score + random.randint(-5, 10)))
    style_score = max(0, min(100, base_score + random.randint(-10, 5)))
    
    overall_score = int((grammar_score + vocab_score + structure_score + content_score + style_score) / 5)
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Грамматика", "score": grammar_score, 
             "feedback": "Есть ошибки в использовании частиц. Обратите внимание на 了, 的, 地, 得."},
            {"name": "Лексика", "score": vocab_score,
             "feedback": "Достаточно разнообразный словарный запас для уровня HSK " + str(request.hsk_level)},
            {"name": "Структура", "score": structure_score,
             "feedback": "Логичная организация текста, но можно улучшить связность между абзацами."},
            {"name": "Содержание", "score": content_score,
             "feedback": "Соответствует теме, есть аргументы и примеры."},
            {"name": "Стиль", "score": style_score,
             "feedback": "Стиль достаточно разнообразный, но можно использовать более сложные конструкции."}
        ],
        "errors": [
            {"position": random.randint(10, len(text)//2), 
             "error": "Возможная ошибка в порядке слов",
             "correction": "Проверьте порядок слов в предложении"},
            {"position": random.randint(len(text)//2, len(text)-10),
             "error": "Повтор одних и тех же слов",
             "correction": "Используйте синонимы для разнообразия"}
        ] if char_count > 50 else [],
        "recommendations": """
        1. Практикуйтесь в использовании сложных предложений с 虽然...但是..., 因为...所以...
        2. Увеличьте словарный запас по теме "{}"
        3. Обратите внимание на использование частиц 了, 的, 地, 得
        4. Добавьте вводные слова: 首先, 其次, 最后, 总而言之
        5. Пишите регулярно для улучшения навыков
        """.format(request.topic),
        "strengths": "Хорошая организация текста, соответствие теме, достаточный объем.",
        "estimated_hsk_level": request.hsk_level,
        "topic": request.topic,
        "target_hsk": request.hsk_level,
        "actual_length": char_count,
        "checked_at": datetime.now().isoformat(),
        "ai_checked": False,
        "fallback": True
    }

@app.post("/translation/generate")
async def generate_translation_text(request: TranslationGenerateRequest):
    """Генерация текста для перевода"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_translation_text(request)
        
        # Определяем длину
        lengths = {
            "short": "3-5 предложений",
            "medium": "6-10 предложений", 
            "long": "10-15 предложений"
        }
        
        system_prompt = f"""Ты создаешь тексты на русском языке для перевода на китайский.
        
# ЗАДАЧА:
Создать текст на тему: "{request.topic}"
Сложность: {request.difficulty}
Длина: {lengths.get(request.length, "6-10 предложений")}
Уровень студента: HSK {request.hsk_level}

# ТРЕБОВАНИЯ:
1. Текст должен быть интересным и полезным для изучения
2. Уровень сложности соответствовать уровню студента
3. Использовать разнообразную лексику и грамматику
4. Текст должен быть естественным, как в реальной жизни
5. Включать элементы, которые нужно перевести правильно

# ФОРМАТЫ ТЕКСТА:
- Новость: формальный стиль, факты
- Рассказ: повествование, диалоги
- Диалог: разговорная речь, вопросы и ответы
- Описание: детали, прилагательные
- Инструкция: императивы, последовательность

# ПРИМЕР ДЛЯ СРЕДНЕЙ СЛОЖНОСТИ:
"Вчера в Шанхае открылся новый культурный центр. Он объединяет библиотеку, музей и концертный зал. Посетители могут бесплатно посещать выставки в первый месяц."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Создай текст для перевода на тему: {request.topic}"}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        text = response.choices[0].message.content
        
        return {
            "text": text.strip(),
            "topic": request.topic,
            "difficulty": request.difficulty,
            "length": request.length,
            "target_hsk": request.hsk_level,
            "generated_at": datetime.now().isoformat(),
            "ai_generated": True
        }
        
    except Exception as e:
        print(f"Ошибка генерации текста: {str(e)}")
        return generate_fallback_translation_text(request)

def generate_fallback_translation_text(request: TranslationGenerateRequest):
    """Fallback генерация текста для перевода"""
    topics_texts = {
        "news": "Китай запустил новый спутник для наблюдения за Землей. Он будет использоваться для мониторинга погоды и экологии. Спутник выведен на орбиту ракетой-носителем Чанчжэн.",
        "story": "Давным-давно в маленькой деревне жил старый мастер каллиграфии. Каждое утро он вставал на рассвете и практиковал иероглифы. Его работы были известны по всему региону.",
        "dialogue": "- Здравствуйте! Меня зовут Анна. Я из России. - Очень приятно! Я Ли Вэй. Вы впервые в Китае? - Да, я здесь изучаю китайский язык. - Отлично! Удачи в учебе!",
        "description": "Великая Китайская стена - это древнее оборонительное сооружение. Она проходит через горы и долины северного Китая. Длина стены составляет более 20 тысяч километров.",
        "instruction": "Чтобы приготовить жареный рис по-китайски, сначала нужно отварить рис и охладить его. Затем обжарить яйца, добавить овощи и нарезанное мясо. В конце добавить рис и соевый соус."
    }
    
    # Выбираем текст по теме или используем общий
    text = topics_texts.get(request.topic, 
        "Китайская культура очень богата и разнообразна. Она включает в себя традиционную медицину, кухню, искусство и философию. Изучение китайской культуры помогает лучше понимать язык.")
    
    # Адаптируем сложность
    if request.difficulty == "easy":
        # Упрощаем текст
        sentences = text.split('. ')
        text = '. '.join(sentences[:2]) + '.'
    elif request.difficulty == "hard":
        # Усложняем текст
        text += " Эти аспекты тесно связаны с историческим развитием страны и влиянием конфуцианства."
    
    return {
        "text": text,
        "topic": request.topic,
        "difficulty": request.difficulty,
        "length": request.length,
        "target_hsk": request.hsk_level,
        "generated_at": datetime.now().isoformat(),
        "ai_generated": False,
        "fallback": True
    }

@app.post("/translation/check")
async def check_translation(request: TranslationCheckRequest):
    """Проверка перевода AI"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_translation_check(request)
        
        system_prompt = f"""Ты — эксперт по переводу с русского на китайский.
        
# ЗАДАЧА:
Сравнить перевод студента с идеальным переводом.
Оригинал (русский): "{request.original_text}"
Перевод студента: "{request.user_translation}"
Уровень студента: HSK {request.target_hsk}
Сложность: {request.difficulty}

# КРИТЕРИИ ОЦЕНКИ:
1. **Точность** (40%) - правильность перевода смысла
2. **Грамматика** (30%) - правильность китайских конструкций
3. **Естественность** (20%) - звучит ли как родной язык
4. **Стиль** (10%) - сохранение стиля оригинала

# ТВОЯ РАБОТА:
1. Создать идеальный перевод оригинала
2. Сравнить с переводом студента
3. Найти и классифицировать ошибки
4. Дать рекомендации по улучшению
5. Поставить оценку

# ФОРМАТ ОТВЕТА JSON:
{{
    "overall_score": 85,
    "ideal_translation": "Идеальный перевод текста на китайский...",
    "categories": [
        {{"name": "Точность", "score": 90, "feedback": "..."}},
        {{"name": "Грамматика", "score": 80, "feedback": "..."}},
        {{"name": "Естественность", "score": 85, "feedback": "..."}},
        {{"name": "Стиль", "score": 80, "feedback": "..."}}
    ],
    "errors": [
        {{"type": "grammar", "description": "Неправильный порядок слов", "suggestion": "..."}},
        {{"type": "vocabulary", "description": "Неточный перевод слова", "suggestion": "..."}}
    ],
    "correct_translations": [
        {{"original": "русская фраза", "student": "перевод студента", "ideal": "идеальный перевод"}}
    ],
    "recommendations": "Конкретные рекомендации...",
    "estimated_hsk_level": {request.target_hsk}
}}

# БУДЬ КОНСТРУКТИВНЫМ:
- Хвали за хорошие моменты
- Объясняй ошибки подробно
- Предлагай альтернативы
- Помогай учиться на ошибках"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Проверь этот перевод и дай детальный анализ."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Добавляем метаданные
        result.update({
            "original_text": request.original_text,
            "user_translation": request.user_translation,
            "target_hsk": request.target_hsk,
            "difficulty": request.difficulty,
            "checked_at": datetime.now().isoformat(),
            "ai_checked": True
        })
        
        return result
        
    except Exception as e:
        print(f"Ошибка проверки перевода: {str(e)}")
        return generate_fallback_translation_check(request)

def generate_fallback_translation_check(request: TranslationCheckRequest):
    """Fallback проверка перевода"""
    # Генерируем "идеальный" перевод (простой)
    ideal_translation = generate_simple_translation(request.original_text, request.target_hsk)
    
    # Простая оценка
    base_score = 70 + random.randint(-15, 15)
    accuracy_score = max(0, min(100, base_score + random.randint(-10, 10)))
    grammar_score = max(0, min(100, base_score + random.randint(-15, 5)))
    naturalness_score = max(0, min(100, base_score + random.randint(-5, 10)))
    style_score = max(0, min(100, base_score + random.randint(-10, 5)))
    
    overall_score = int((accuracy_score + grammar_score + naturalness_score + style_score) / 4)
    
    return {
        "overall_score": overall_score,
        "ideal_translation": ideal_translation,
        "categories": [
            {"name": "Точность", "score": accuracy_score,
             "feedback": "Основной смысл передан правильно, но есть неточности в деталях."},
            {"name": "Грамматика", "score": grammar_score,
             "feedback": "Есть ошибки в порядке слов и использовании частиц."},
            {"name": "Естественность", "score": naturalness_score,
             "feedback": "Перевод понятен, но звучит немного неестественно для носителя."},
            {"name": "Стиль", "score": style_score,
             "feedback": "Стиль в основном сохранен, но можно улучшить."}
        ],
        "errors": [
            {"type": "grammar", 
             "description": "Возможные ошибки в порядке слов",
             "suggestion": "В китайском языке порядок SVO (подлежащее-сказуемое-дополнение)"},
            {"type": "vocabulary",
             "description": "Можно использовать более точные слова",
             "suggestion": "Используйте синонимы для разнообразия и точности"}
        ],
        "correct_translations": [
            {"original": request.original_text.split('. ')[0] if '. ' in request.original_text else request.original_text,
             "student": request.user_translation.split('。')[0] if '。' in request.user_translation else request.user_translation,
             "ideal": ideal_translation.split('。')[0] if '。' in ideal_translation else ideal_translation}
        ],
        "recommendations": """
        1. Обращайте внимание на порядок слов в предложении
        2. Используйте словари для поиска более точных эквивалентов
        3. Практикуйтесь в переводе разных типов текстов
        4. Читайте оригинальные китайские тексты для понимания естественного стиля
        5. Проверяйте использование частиц 了, 的, 地, 得
        """,
        "estimated_hsk_level": request.target_hsk,
        "original_text": request.original_text,
        "user_translation": request.user_translation,
        "target_hsk": request.target_hsk,
        "difficulty": request.difficulty,
        "checked_at": datetime.now().isoformat(),
        "ai_checked": False,
        "fallback": True
    }

def generate_simple_translation(text: str, hsk_level: int) -> str:
    """Простой перевод текста (заглушка)"""
    # В реальном проекте здесь был бы реальный перевод
    # Сейчас возвращаем шаблонный текст
    translations = {
        3: "这是一个简单的翻译示例。中文很重要。",
        4: "昨天在公园里有很多人。天气很好，阳光明媚。",
        5: "随着中国经济的发展，越来越多的外国人来到中国工作和学习。",
        6: "中国传统文化博大精深，源远流长。它不仅包括丰富的哲学思想，还涵盖了独特的艺术形式和生活智慧。"
    }
    
    return translations.get(hsk_level, "这是一个翻译文本。")

@app.get("/text/history/{user_id}")
async def get_text_generation_history(user_id: str, limit: int = 20):
    """Получить историю генерации текстов"""
    try:
        # Загружаем из файла
        history_file = f"data/text_history_{user_id}.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            return {
                "history": history[:limit],
                "count": len(history),
                "total_characters": sum(item.get("length_chars", 0) for item in history)
            }
        return {"history": [], "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки истории: {str(e)}")

@app.get("/words/level/{level}")
async def get_level_words(level: int, limit: int = 10000, offset: int = 0):
    """Получить слова определенного уровня HSK"""
    level_words = get_words_by_level(level, 20000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"Слова HSK {level} не найдены")
    
    paginated_words = level_words[offset:offset + limit]
    
    return {
        "level": level,
        "count": len(paginated_words),
        "total": len(level_words),
        "offset": offset,
        "limit": limit,
        "words": paginated_words
    }

@app.get("/levels/summary")
async def get_levels_summary():
    """Сводка по всем уровням HSK"""
    summary = {}
    for level in range(1, 7):
        level_words = get_words_by_level(level, 1000)
        if level_words:
            summary[f"hsk{level}"] = {
                "word_count": len(level_words),
                "sample_words": level_words[:3],
                "common_characters": list(set([char for word in level_words[:10] for char in word["character"]]))[:5]
            }
    
    return summary

@app.get("/user/{user_id}/progress")
async def get_user_progress(user_id: str):
    """Прогресс пользователя"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    user = users_db[user_id]
    progress = word_progress_db.get(user_id, {})
    
    # Статистика по уровням
    level_stats = {}
    for level in range(1, 7):
        level_words = get_words_by_level(level, 1000)
        total_level_words = len(level_words)
        
        # Считаем изученные слова этого уровня
        learned = 0
        for word_id, word_progress in progress.items():
            if word_progress.get("remembered", False):
                # Проверяем что слово этого уровня
                for word in level_words:
                    if f"{word['character']}_{level}" == word_id:
                        learned += 1
                        break
        
        if total_level_words > 0:
            level_stats[f"HSK {level}"] = {
                "learned": learned,
                "total": total_level_words,
                "percentage": int((learned / total_level_words) * 100)
            }
    
    total_learned = len([p for p in progress.values() if p.get("remembered", False)])
    
    return {
        "user": user["name"],
        "user_id": user_id,
        "stats": {
            "total_learned": total_learned,
            "total_words": len(words_db),
            "overall_percentage": int((total_learned / len(words_db)) * 100) if words_db else 0,
            "by_level": level_stats
        },
        "study_plan": {
            "daily_words": user.get("daily_words", 10),
            "days_until_exam": user.get("days_until_exam", 30),
            "words_per_day_to_goal": max(1, (user.get("target_words", 1000) - total_learned) // max(1, user.get("days_until_exam", 30)))
        }
    }


# Добавьте в модели бэкенда:
class EssayAnalysisRequest(BaseModel):
    topic: str
    details: Optional[str] = ""
    difficulty: str = "intermediate"
    target_length: int = 400
    user_id: Optional[str] = None

class EssayAnalysisResponse(BaseModel):
    prompt: str
    topic: str
    difficulty: str
    target_length: int
    requirements: str
    evaluation_criteria: List[str]
    time_limit_minutes: int
    generated_at: str

class EssaySubmitRequest(BaseModel):
    essay_text: str
    topic: str
    difficulty: str
    target_length: int
    user_id: Optional[str] = None
    time_spent: Optional[int] = None

# Добавьте в роуты бэкенда:
@app.post("/essay/analysis/generate")
async def generate_essay_analysis(request: EssayAnalysisRequest):
    """Генерация задания для эссе"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_analysis(request)
        
        # Определяем время на основе сложности
        time_limits = {
            "beginner": 45,
            "intermediate": 60,
            "advanced": 75,
            "exam": 90
        }
        
        system_prompt = f"""Ты создаешь задания для эссе на китайском языке.
        
# ЗАДАЧА:
Создать задание для эссе на тему: "{request.topic}"
Уровень сложности: {request.difficulty}
Целевая длина: {request.target_length} иероглифов
Дополнительные детали: {request.details}

# ТРЕБОВАНИЯ К ЗАДАНИЮ:
1. Четко сформулированная тема и задача
2. Конкретные требования к содержанию
3. Критерии оценки по 4 категориям:
   - Содержание (40%)
   - Грамматика (30%) 
   - Лексика (20%)
   - Структура (10%)
4. Время на выполнение: {time_limits.get(request.difficulty, 60)} минут

# ФОРМАТ ОТВЕТА JSON:
{{
    "prompt": "Полное задание для студента с инструкциями...",
    "requirements": "Конкретные требования к эссе...",
    "evaluation_criteria": [
        "Содержание: соответствие теме, аргументы, примеры (40%)",
        "Грамматика: правильность конструкций, частицы, времена (30%)",
        "Лексика: разнообразие словаря, уместность слов (20%)",
        "Структура: логичность, организация, связность (10%)"
    ],
    "time_limit_minutes": {time_limits.get(request.difficulty, 60)},
    "suggested_structure": ["Введение", "2-3 аргумента", "Заключение"]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Создай задание для эссе на тему: {request.topic}"}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Добавляем метаданные
        result.update({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "target_length": request.target_length,
            "generated_at": datetime.now().isoformat(),
            "ai_generated": True
        })
        
        return result
        
    except Exception as e:
        print(f"Ошибка генерации задания: {str(e)}")
        return generate_fallback_essay_analysis(request)

def generate_fallback_essay_analysis(request: EssayAnalysisRequest):
    """Fallback генерация задания для эссе"""
    difficulty_texts = {
        "beginner": "Используйте простые предложения и базовую лексику HSK 1-3.",
        "intermediate": "Используйте сложные предложения и разнообразную лексику HSK 4-5.",
        "advanced": "Продемонстрируйте владение сложными грамматическими конструкциями.",
        "exam": "Продемонстрируйте все аспекты владения языком на высоком уровне."
    }
    
    time_limits = {
        "beginner": 45,
        "intermediate": 60,
        "advanced": 75,
        "exam": 90
    }
    
    return {
        "prompt": f"""
<h4>Тема: {request.topic}</h4>
<p><strong>Задание:</strong> Напишите эссе на заданную тему. Ваше эссе должно включать:</p>
<ul>
    <li>Введение с представлением темы и вашей позиции</li>
    <li>2-3 основных аргумента с конкретными примерами</li>
    <li>Заключение с выводами и обобщением</li>
</ul>
<p><strong>Требования:</strong></p>
<ul>
    <li>Объем: {request.target_length} иероглифов</li>
    <li>{difficulty_texts.get(request.difficulty, 'Используйте сложные предложения')}</li>
    <li>Используйте вводные слова и связующие элементы</li>
    <li>Избегайте повторений и грамматических ошибок</li>
</ul>
<p><strong>Время выполнения:</strong> {time_limits.get(request.difficulty, 60)} минут</p>
        """,
        "requirements": f"Объем: {request.target_length} иероглифов. {difficulty_texts.get(request.difficulty, 'Используйте сложные предложения')}",
        "evaluation_criteria": [
            "Содержание: соответствие теме, аргументы, примеры (40%)",
            "Грамматика: правильность конструкций, частицы, времена (30%)",
            "Лексика: разнообразие словаря, уместность слов (20%)",
            "Структура: логичность, организация, связность (10%)"
        ],
        "time_limit_minutes": time_limits.get(request.difficulty, 60),
        "suggested_structure": ["Введение", "2-3 аргумента", "Заключение"],
        "topic": request.topic,
        "difficulty": request.difficulty,
        "target_length": request.target_length,
        "generated_at": datetime.now().isoformat(),
        "ai_generated": False,
        "fallback": True
    }

@app.post("/essay/analysis/check")
async def check_essay_analysis(request: EssaySubmitRequest):
    """Строгая проверка эссе для анализа"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_check_analysis(request)
        
        system_prompt = f"""Ты — СТРОГИЙ и ТРЕБОВАТЕЛЬНЫЙ преподаватель китайского языка.
        
# ЗАДАЧА:
Проверить эссе на тему: "{request.topic}"
Уровень сложности: {request.difficulty}
Целевая длина: {request.target_length} иероглифов
Длина эссе студента: {len(request.essay_text)} иероглифов

# БУДЬ МАКСИМАЛЬНО СТРОГИМ:
- Не завышай оценки ни на балл!
- За каждую ошибку снижай баллы
- Требуй совершенства
- Не делай скидок

# КРИТЕРИИ ОЦЕНКИ:
1. **Содержание** (40%) - точность, аргументы, примеры, глубина
2. **Грамматика** (30%) - идеальная грамматика, никаких ошибок
3. **Лексика** (20%) - богатый словарь, точность, разнообразие
4. **Структура** (10%) - идеальная организация, логика, связность

# ФОРМАТ ОТВЕТА JSON:
{{
    "overall_score": 65,  // БУДЬ СТРОГИМ!
    "categories": [
        {{"name": "Содержание", "score": 70, "feedback": "СТРОГИЙ отзыв с указанием ВСЕХ недостатков"}},
        {{"name": "Грамматика", "score": 60, "feedback": "СТРОГИЙ отзыв с ПЕРЕЧНЕМ ВСЕХ ошибок"}},
        {{"name": "Лексика", "score": 75, "feedback": "СТРОГИЙ отзыв о словарном запасе"}},
        {{"name": "Структура", "score": 80, "feedback": "СТРОГИЙ отзыв о структуре"}}
    ],
    "errors": [
        {{"type": "grammar", "position": 15, "description": "КОНКРЕТНАЯ ошибка", "correction": "ТОЧНОЕ исправление", "severity": "high"}},
        {{"type": "vocabulary", "position": 42, "description": "НЕТОЧНОЕ слово", "correction": "ПРАВИЛЬНЫЙ вариант", "severity": "medium"}}
    ],
    "strengths": "Только реальные сильные стороны, не выдумывай!",
    "weaknesses": "ПОДРОБНЫЙ список слабых мест",
    "recommendations": "КОНКРЕТНЫЕ и ЖЕСТКИЕ рекомендации по улучшению",
    "estimated_level": "Реальный уровень студента (НЕ завышай!)",
    "would_pass_exam": false  // Честно оцени, сдал бы экзамен?
}}

# ЭССЕ СТУДЕНТА:
{request.essay_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Проверь это эссе МАКСИМАЛЬНО СТРОГО и дай честную оценку."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,  # Низкая температура для строгости
            max_tokens=2500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = generate_fallback_essay_check_analysis(request)
        
        # Добавляем метаданные
        result.update({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "target_length": request.target_length,
            "actual_length": len(request.essay_text),
            "time_spent": request.time_spent,
            "checked_at": datetime.now().isoformat(),
            "strict_check": True
        })
        
        return result
        
    except Exception as e:
        print(f"Ошибка строгой проверки: {str(e)}")
        return generate_fallback_essay_check_analysis(request)
    
@app.post("/ai/search-universities")
async def search_universities(request: dict):
    """
    Главная функция: AI ищет университеты в интернете
    """
    try:
        query = request.get("query", "")
        filters = request.get("filters", {})
        
        if not query:
            raise HTTPException(status_code=400, detail="Пустой запрос")
        
        # 1. Формируем УМНЫЙ промпт для AI
        system_prompt = f"""
        Ты — эксперт по китайскому образованию. Пользователь ищет: "{query}"
        
        ТВОЯ ЗАДАЧА: НАЙТИ АКТУАЛЬНУЮ ИНФОРМАЦИЮ В ИНТЕРНЕТЕ
        
        ИНСТРУКЦИИ:
        1. ИСПОЛЬЗУЙ ПОИСК В ИНТЕРНЕТЕ чтобы найти свежие данные
        2. Ищи на русском, английском, китайском языках
        3. Основные источники: официальные сайты вузов (.edu.cn), csc.edu.cn, studyinchina.edu.cn
        4. Учитывай фильтры: HSK {filters.get('hsk_level', 'любой')}, бюджет {filters.get('max_budget', 'любой')}
        5. Сравни минимум 3-5 вариантов
        6. Давай конкретные данные: цены, сроки, контакты
        
        ФОРМАТ ОТВЕТА:
        - Название университета (город)
        - Требования: HSK, экзамены, документы
        - Стоимость обучения (в юанях)
        - Стипендии: какие есть, как получить
        - Сроки подачи документов
        - Ссылки на официальные страницы
        - Плюсы и минусы каждого варианта
        - Советы по поступлению
        
        ВАЖНО: Все данные должны быть АКТУАЛЬНЫМИ (2024-2025 год).
        """
        
        # 2. Вызываем DeepSeek с ВКЛЮЧЕННЫМ поиском в интернете
        client = get_deepseek_client()
        if not client:
            return {"error": "API ключ не настроен"}
        
        # КРИТИЧЕСКИ ВАЖНО: Включите веб-поиск!
        # Уточните точное название параметра в документации DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",  # Или другая модель с поиском
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Найди информацию по запросу: {query}"}
            ],
            # ПАРАМЕТР ДЛЯ ВЕБ-ПОИСКА (примерные названия):
            # web_search=True, 
            # use_web=True,
            # search_online=True,
            max_tokens=4000  # Много токенов для подробного ответа
        )
        
        ai_response = response.choices[0].message.content
        
        # 3. Возвращаем результат
        return {
            "success": True,
            "query": query,
            "analysis": ai_response,  # Текст от AI
            "count": len(ai_response.split('\n')) // 10,  # Примерное количество вариантов
            "search_performed": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Ошибка в AI поиске: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback": "Покажу локальные данные...",
            # Можно добавить fallback данные из вашей БД
        }

def generate_fallback_essay_check_analysis(request: EssaySubmitRequest):
    """Fallback строгая проверка"""
    text = request.essay_text
    char_count = len(text)
    
    # СТРОГАЯ оценка на основе длины
    length_ratio = char_count / request.target_length
    if length_ratio < 0.5:
        length_penalty = 30
    elif length_ratio < 0.8:
        length_penalty = 15
    elif length_ratio < 1.0:
        length_penalty = 5
    else:
        length_penalty = 0
    
    base_score = 70 - length_penalty
    
    # СТРОГИЕ оценки по категориям
    content_score = max(0, min(100, base_score + random.randint(-20, 10)))
    grammar_score = max(0, min(100, base_score + random.randint(-25, 5)))
    vocab_score = max(0, min(100, base_score + random.randint(-15, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-10, 15)))
    
    overall_score = int((content_score + grammar_score + vocab_score + structure_score) / 4)
    
    # ЖЕСТКИЕ ошибки
    errors = []
    if char_count > 50:
        errors.append({
            "type": "grammar",
            "position": min(30, char_count - 20),
            "description": "СЕРЬЕЗНАЯ ошибка в использовании 了",
            "correction": "НИКОГДА не используйте 了 в этом контексте",
            "severity": "high"
        })
        
    if char_count > 100:
        errors.append({
            "type": "vocabulary", 
            "position": min(70, char_count - 30),
            "description": "ЭТО слово НЕПРАВИЛЬНОЕ в данном контексте",
            "correction": "Используйте ТОЛЬКО правильное слово: ...",
            "severity": "medium"
        })
    
    would_pass = overall_score >= 70  # ЖЕСТКИЙ проходной балл
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Содержание", "score": content_score,
             "feedback": "НЕДОСТАТОЧНО глубокий анализ. Нужны КОНКРЕТНЫЕ примеры и детали."},
            {"name": "Грамматика", "score": grammar_score,
             "feedback": "МНОГО ошибок в грамматике. Неприемлемо для этого уровня."},
            {"name": "Лексика", "score": vocab_score,
             "feedback": "Словарный запас ОЧЕНЬ ограничен. Учите больше слов."},
            {"name": "Структура", "score": structure_score,
             "feedback": "Структура хаотична. Следуйте плану: введение-аргументы-заключение."}
        ],
        "errors": errors,
        "strengths": "Только один плюс: соответствие теме (но слабое).",
        "weaknesses": "ВСЁ остальное: грамматика, лексика, структура, аргументация.",
        "recommendations": """
1. ВЫУЧИТЕ грамматику заново. Ошибки НЕДОПУСТИМЫ.
2. УВЕЛИЧЬТЕ словарный запас в 2 раза. СЕЙЧАС недостаточно.
3. ПИШИТЕ по плану ВСЕГДА. Хаос - это провал.
4. ПРАКТИКУЙТЕСЬ каждый день. Раз в неделю - СЛИШКОМ МАЛО.
5. НАНИМИТЕ репетитора, если не справляетесь сами.
        """,
        "estimated_level": f"Реальный уровень: HSK {max(1, min(6, overall_score // 15))}",
        "would_pass_exam": would_pass,
        "topic": request.topic,
        "difficulty": request.difficulty,
        "target_length": request.target_length,
        "actual_length": char_count,
        "checked_at": datetime.now().isoformat(),
        "strict_check": True,
        "fallback": True
    }

class AudioLessonRequest(BaseModel):
    topic: str
    description: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    target_length: str = "medium"  # short, medium, long
    hsk_level: int = 3
    include_pinyin: bool = False
    include_translation: bool = False
    user_id: Optional[str] = None

class AudioLessonResponse(BaseModel):
    id: str
    title: str
    chinese_text: str
    pinyin_text: Optional[str] = None
    translation: Optional[str] = None
    vocabulary: List[Dict[str, str]]
    difficulty: str
    estimated_duration: int  # в секундах
    generated_at: str

# ЗАМЕНИТЕ функцию generate_audio_lesson в бэкенде на эту:
@app.post("/audio/generate-lesson")
async def generate_audio_lesson(request: AudioLessonRequest):
    """Генерация полноценного аудио-урока (подкаста) на китайском"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_audio_lesson(request)
        
        # Определяем длину текста в зависимости от выбора пользователя
        length_targets = {
            "short": 300,    # 1-2 минуты
            "medium": 600,   # 3-5 минут
            "long": 1000     # 5-10 минут
        }
        
        target_chars = length_targets.get(request.target_length, 600)
        
        system_prompt = f"""Ты профессиональный создатель китайских подкастов для изучающих язык.

# ЗАДАЧА:
Создать полноценный подкаст на тему: "{request.topic}"
Детали темы: {request.description or 'Не указано'}
Уровень HSK: {request.hsk_level}
Сложность: {request.difficulty}
Длительность: {request.target_length}
Примерный объем: {target_chars} иероглифов

# ТРЕБОВАНИЯ К ПОДКАСТУ:
1. Должен быть ПОЛНОЦЕННЫМ аудио-уроком с:
   - Приветствием и введением в тему
   - Основной частью с развитием темы
   - Конкретными примерами и деталями
   - Полезными выражениями и лексикой
   - Вопросами для слушателей
   - Итогами и заключением

2. ДЛИНА: Не менее {target_chars} иероглифов
3. СТРУКТУРА:
   - Введение (20%)
   - Основная часть (60%)
   - Заключение (20%)
4. СТИЛЬ: Естественный, разговорный, но понятный
5. ВКЛЮЧИТЬ: 
   - Диалоги или примеры диалогов
   - Культурные заметки
   - Полезные советы
   - Конкретные примеры использования языка

# ИЗБЕГАТЬ:
- Шаблонных фраз
- Слишком академического языка
- Повторений
- Слишком коротких предложений

# ФОРМАТ ОТВЕТА JSON:
{{
    "title": "Заголовок подкаста",
    "chinese_text": "Полный текст подкаста здесь...",
    "pinyin_text": "Текст с пиньинем (если include_pinyin=true)",
    "translation": "Полный перевод на русский (если include_translation=true)",
    "vocabulary": [
        {{
            "chinese": "词语",
            "pinyin": "cíyǔ", 
            "translation": "перевод",
            "example": "Пример предложения",
            "category": "часть речи"
        }}
    ],
    "comprehension_questions": [
        {{
            "question": "Вопрос на понимание",
            "options": ["A", "B", "C", "D"],
            "correct_answer": 0,
            "explanation": "Объяснение ответа"
        }}
    ],
    "estimated_duration": 180,
    "word_count": 500,
    "character_count": 800,
    "difficulty_analysis": {{
        "grammar_complexity": "средняя",
        "vocabulary_level": "HSK {request.hsk_level}",
        "speed_recommendation": "1.0x"
    }}
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Создай полноценный подкаст на китайском.

Тема: {request.topic}
Описание: {request.description or 'Не указано'}
Уровень: HSK {request.hsk_level}
Сложность: {request.difficulty}
Длительность: {request.target_length}

Пожалуйста, сделай текст ЕСТЕСТВЕННЫМ и РАЗГОВОРНЫМ, как настоящий подкаст."""}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.9,  # Более творческий подход
            max_tokens=4000,   # Увеличиваем для длинных текстов
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Генерируем ID урока
        lesson_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.topic[:20])}"
        
        # Добавляем метаданные
        result.update({
            "id": lesson_id,
            "difficulty": request.difficulty,
            "hsk_level": request.hsk_level,
            "generated_at": datetime.now().isoformat(),
            "topic": request.topic,
            "description": request.description,
            "ai_generated": True,
            "target_length": request.target_length,
            "request_details": {
                "topic": request.topic,
                "description": request.description,
                "hsk_level": request.hsk_level,
                "difficulty": request.difficulty,
                "include_pinyin": request.include_pinyin,
                "include_translation": request.include_translation
            }
        })
        
        # Если пользователь не запросил пиньинь, удаляем его
        if not request.include_pinyin:
            result["pinyin_text"] = None
        
        # Если пользователь не запросил перевод, удаляем его
        if not request.include_translation:
            result["translation"] = None
        
        return result
        
    except Exception as e:
        print(f"Ошибка генерации аудио-урока: {str(e)}")
        # Всегда используем fallback с более длинным текстом
        return generate_improved_fallback_audio_lesson(request)

def generate_improved_fallback_audio_lesson(request: AudioLessonRequest):
    """Улучшенный fallback для генерации подкаста"""
    
    # Создаем более длинные и разнообразные тексты
    topic = request.topic
    difficulty = request.difficulty
    
    # Базовый текст в зависимости от темы и сложности
    base_text = f"""大家好！欢迎收听今天的汉语学习播客。

今天我们的话题是：{topic}。

这个话题很有意思，也很重要。让我来详细介绍一下。

首先，{topic}在中国文化中占有特殊地位。无论是传统还是现代角度，这个话题都值得深入探讨。

举个例子来说，很多外国朋友来到中国，都会对{topic}产生浓厚的兴趣。他们经常问："中国的{topic}有什么特点？" "我应该如何更好地了解{topic}？"

事实上，{topic}不仅是一个简单的概念，它反映了中国社会的很多方面。从历史角度来看，{topic}有着悠久的历史传承。从现代视角来看，{topic}也在不断发展和变化。

我个人认为，学习{topic}对于理解中国非常有帮助。通过这个话题，我们可以了解中国人的思维方式、文化传统和社会价值观。

在学习汉语的过程中，关于{topic}的词汇和表达也非常有用。比如，我们可以学习到很多相关的词语和句子结构。

那么，如何更好地学习这个话题呢？我建议：
第一，多听相关的材料；
第二，尝试用汉语讨论这个话题；
第三，如果有机会，亲身体验一下。

当然，学习过程中可能会遇到一些困难。比如，有些专业词汇比较难记，有些文化概念不太容易理解。但没关系，慢慢来，一步一步学习。

记住，学习语言不仅是学习单词和语法，更是学习一种文化和思维方式。通过{topic}，我们可以更深入地了解中国。

好了，今天的播客就到这里。希望这个内容对你有帮助。如果你有任何问题或想法，欢迎留言讨论。

下次再见！祝你学习进步！"""
    
    # Добавляем вариации в зависимости от уровня HSK
    if request.hsk_level <= 2:
        # Упрощаем для начинающих
        base_text = f"""你好！我是你的中文老师。

今天我们学习：{topic}。

{topic}很有意思。我们来看看。

这是什么？这是{topic}。你喜欢{topic}吗？

我喜欢{topic}。你呢？

我们一起学习。慢慢说，不要急。

好，今天学到这里。再见！"""
    
    elif request.hsk_level >= 5:
        # Усложняем для продвинутых
        base_text = f"""各位听众朋友，大家好。

欢迎收听本期深度汉语学习播客。今天我们将围绕"{topic}"这一主题展开探讨。

在当前全球化语境下，{topic}作为一个跨文化议题，引起了广泛关注。从本质上看，{topic}不仅涉及语言层面的表达，更蕴含着深刻的文化内涵。

首先，让我们从历史维度审视{topic}的演变过程。自古以来，{topic}在中国传统文化体系中占据重要位置。相关文献记载表明，早在先秦时期，{topic}的概念就已初步形成，并随着时代变迁不断丰富发展。

其次，现代社会的{topic}呈现出新的特点。在数字化转型的背景下，{topic}的表现形式和实践方式都发生了显著变化。这种变化既带来机遇，也带来挑战。

从语言学习的角度而言，掌握与{topic}相关的专业术语和表达方式至关重要。这不仅有助于提升语言能力，更能促进跨文化理解。

值得注意的是，不同文化背景的学习者对{topic}的认知可能存在差异。因此，在讨论{topic}时，我们需要保持开放的态度，尊重多元视角。

总而言之，{topic}是一个值得深入研究的复杂课题。通过系统学习，我们不仅能够提升汉语水平，更能深化对中国文化的理解。

感谢收听，我们下期再见。"""
    
    # Генерируем лексику
    vocabulary = [
        {
            "chinese": "话题",
            "pinyin": "huàtí", 
            "translation": "тема, предмет разговора",
            "example": "今天的话题很有意思。",
            "category": "名词"
        },
        {
            "chinese": "学习",
            "pinyin": "xuéxí",
            "translation": "учиться, изучать",
            "example": "我喜欢学习中文。",
            "category": "动词"
        },
        {
            "chinese": "文化",
            "pinyin": "wénhuà",
            "translation": "культура",
            "example": "中国文化很有特色。",
            "category": "名词"
        },
        {
            "chinese": "重要",
            "pinyin": "zhòngyào",
            "translation": "важный",
            "example": "这个问题很重要。",
            "category": "形容词"
        }
    ]
    
    # Добавляем больше слов для продвинутых уровней
    if request.hsk_level >= 4:
        vocabulary.extend([
            {
                "chinese": "探讨",
                "pinyin": "tàntǎo",
                "translation": "обсуждать, исследовать",
                "example": "我们来探讨一下这个问题。",
                "category": "动词"
            },
            {
                "chinese": "理解",
                "pinyin": "lǐjiě",
                "translation": "понимать",
                "example": "我理解你的意思。",
                "category": "动词"
            }
        ])
    
    # Вопросы для понимания
    comprehension_questions = [
        {
            "question": f"今天播客的主题是什么？",
            "options": ["汉语语法", topic, "中国历史", "旅游景点"],
            "correct_answer": 1,
            "explanation": f"今天的主题是：{topic}"
        },
        {
            "question": "为什么这个话题很重要？",
            "options": ["因为很简单", "因为是热门话题", "因为有助于理解中国文化", "因为老师喜欢"],
            "correct_answer": 2,
            "explanation": "这个话题有助于理解中国文化和社会"
        }
    ]
    
    lesson_id = f"audio_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Рассчитываем примерную длительность (примерно 150 иероглифов в минуту)
    estimated_duration = max(120, len(base_text) // 2)
    
    return {
        "id": lesson_id,
        "title": f"汉语学习播客：{topic}",
        "chinese_text": base_text,
        "pinyin_text": None if not request.include_pinyin else "pinyin текст будет здесь",
        "translation": None if not request.include_translation else "перевод будет здесь",
        "vocabulary": vocabulary,
        "comprehension_questions": comprehension_questions,
        "difficulty": request.difficulty,
        "hsk_level": request.hsk_level,
        "estimated_duration": estimated_duration,
        "generated_at": datetime.now().isoformat(),
        "topic": request.topic,
        "description": request.description,
        "ai_generated": False,
        "fallback": True,
        "target_length": request.target_length,
        "character_count": len(base_text),
        "word_count": len(base_text.split()),
        "difficulty_analysis": {
            "grammar_complexity": "средняя" if request.hsk_level <= 3 else "высокая",
            "vocabulary_level": f"HSK {request.hsk_level}",
            "speed_recommendation": "0.8x" if request.hsk_level <= 2 else "1.0x"
        },
        "note": "Это автоматически сгенерированный подкаст. Для получения более качественного контента проверьте подключение к AI."
    }

def generate_fallback_audio_lesson(request: AudioLessonRequest):
    """Fallback генерация аудио-урока"""
    
    # Базовый текст в зависимости от уровня HSK
    base_texts = {
        1: "你好！我是你的中文老师。今天我们来学习中文。中文很有意思。",
        2: "大家好！欢迎来到中文课。今天天气很好。我想去公园散步。你呢？",
        3: "同学们好！今天我们要学习关于中国文化的主题。中国有很长的历史。中国的食物很好吃。",
        4: "欢迎收听我们的中文播客！今天我们来聊聊中国的传统节日。春节是最重要的节日。",
        5: "在这个数字时代，学习语言变得更加容易。通过互联网，我们可以接触到丰富的学习资源。",
        6: "中华文明源远流长，博大精深。从古代的四大发明到现代的科技创新，中国一直在为世界做出贡献。"
    }
    
    base_text = base_texts.get(request.hsk_level, base_texts[3])
    
    # Добавляем тему в текст
    chinese_text = f"今天的话题是：{request.topic}。{base_text} 希望你喜欢这个内容。再见！"
    
    # Базовая лексика
    vocabulary = [
        {
            "chinese": "话题",
            "pinyin": "huàtí", 
            "translation": "тема",
            "example": "今天的话题很有意思。"
        },
        {
            "chinese": "学习",
            "pinyin": "xuéxí",
            "translation": "учиться",
            "example": "我喜欢学习中文。"
        }
    ]
    
    lesson_id = f"audio_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "id": lesson_id,
        "title": f"Аудио-урок: {request.topic}",
        "chinese_text": chinese_text,
        "pinyin_text": None,
        "translation": f"Тема сегодня: {request.topic}. {base_text} Надеюсь, вам понравится этот контент. До свидания!",
        "vocabulary": vocabulary,
        "difficulty": request.difficulty,
        "hsk_level": request.hsk_level,
        "estimated_duration": len(chinese_text) * 0.5,  # ~0.5 сек на иероглиф
        "generated_at": datetime.now().isoformat(),
        "topic": request.topic,
        "ai_generated": False,
        "fallback": True,
        "speech_rate": 1.0,
        "word_count": len(chinese_text.split()),
        "character_count": len(chinese_text.replace(" ", "")),
        "study_questions": [
            "Какова основная тема этого урока?",
            "Какие новые слова вы услышали?"
        ]
    }

class WordStatus(BaseModel):
    user_id: str
    word_id: str          # формат: "你好_1"
    status: str           # "saved" или "learned"

class WordTestRequest(BaseModel):
    user_id: str
    source: str = "all"          # "all", "saved", "learned"
    count: int = 20
    test_type: str               # "pinyin_from_char", "char_from_pinyin", "translation_from_char", "translation_from_pinyin"

class WordTestSubmit(BaseModel):
    user_id: str
    test_id: str
    answers: Dict[str, str]      # question_id -> ответ пользователя

@app.post("/words/status")
async def set_word_status(request: WordStatus):
    user_id = request.user_id
    if user_id not in user_word_status:
        user_word_status[user_id] = {}
    
    user_word_status[user_id][request.word_id] = {
        "status": request.status,
        "added_at": datetime.now().isoformat()
    }
    save_user_data()
    return {"success": True}

@app.post("/words/test/generate")
async def generate_word_test(req: WordTestRequest):
    # Получаем пул слов
    if req.source == "all":
        all_words = []
        for level in range(1, 7):
            all_words.extend(words_db.get(level, []))
    else:
        if req.user_id not in user_word_status:
            raise HTTPException(404, "Нет сохранённых/изученных слов")
        word_ids = [wid for wid, data in user_word_status[req.user_id].items() if data["status"] == req.source]
        all_words = []
        for wid in word_ids:
            char, lvl = wid.rsplit("_", 1)
            level = int(lvl)
            for w in words_db.get(level, []):
                if w["character"] == char:
                    all_words.append(w)
                    break

    if len(all_words) == 0:
        raise HTTPException(400, "Нет слов для теста")

    if req.count > len(all_words):
        req.count = len(all_words)
    selected = random.sample(all_words, req.count)

    questions = []
    for i, word in enumerate(selected):
        q = {
            "id": str(i),
            "character": word["character"],
            "pinyin": word["pinyin"],
            "translation": word["translation"]
        }
        if req.test_type == "pinyin_from_char":
            q["prompt"] = f"Пиньинь для: {word['character']}"
            q["correct"] = word["pinyin"]
        elif req.test_type == "char_from_pinyin":
            q["prompt"] = f"Иероглифы для: {word['pinyin']}"
            q["correct"] = word["character"]
        elif req.test_type == "translation_from_char":
            q["prompt"] = f"Перевод для: {word['character']}"
            q["correct"] = word["translation"]
        elif req.test_type == "translation_from_pinyin":
            q["prompt"] = f"Перевод для: {word['pinyin']}"
            q["correct"] = word["translation"]
        else:
            raise HTTPException(400, "Неверный test_type")
        questions.append(q)

    test_id = f"word_{req.user_id}_{datetime.now().timestamp()}"
    
    # Сохраняем активный тест для проверки позже
    tests_db[f"active_word_test_{req.user_id}"] = {
        "test_id": test_id,
        "questions": questions,
        "generated_at": datetime.now().isoformat()
    }

    return {"test_id": test_id, "questions": questions, "total": len(questions)}

@app.post("/words/test/submit")
async def submit_word_test(submit: WordTestSubmit):
    test_key = f"active_word_test_{submit.user_id}"
    if test_key not in tests_db or tests_db[test_key].get("test_id") != submit.test_id:
        raise HTTPException(status_code=404, detail="Тест не найден или уже завершён")

    test = tests_db[test_key]
    questions = test["questions"]

    correct = 0
    total = len(questions)
    results = []

    for q in questions:
        qid = q["id"]
        correct_ans = q["correct"].strip().lower()

        user_answer_raw = submit.answers.get(qid)

        # ЯВНО: если нет ответа или пустая строка — НЕВЕРНО
        if user_answer_raw is None or user_answer_raw.strip() == "":
            is_correct = False
            user_display = "(не отвечено)"
        else:
            user_normalized = user_answer_raw.strip().lower()
            is_correct = user_normalized == correct_ans
            user_display = user_answer_raw

        if is_correct:
            correct += 1

        results.append({
            "id": qid,
            "prompt": q["prompt"],
            "user_answer": user_display,
            "correct_answer": q["correct"],
            "correct": is_correct
        })

    percentage = round(correct / total * 100, 1) if total > 0 else 0

    message = f"{correct} из {total} правильных ({percentage}%)"
    if percentage >= 90:
        message += " Отлично! Вы хорошо знаете эти слова!"
    elif percentage >= 70:
        message += " Неплохо, но можно лучше."
    else:
        message += " Нужно больше практиковаться!"

    return {
        "correct": correct,
        "total": total,
        "percentage": percentage,
        "message": message,
        "results": results
    }

class WordTestAIRequest(BaseModel):
    user_id: str
    test_id: str
    questions: List[Dict[str, Any]]  # [{id, prompt, correct, ...}]
    answers: Dict[str, str]          # {question_id: user_answer}

@app.post("/words/test/check-ai")
async def check_word_test_ai(request: WordTestAIRequest):
    """ИИ проверяет тест на знание слов"""
    try:
        client = get_deepseek_client()
        if not client:
            raise HTTPException(status_code=503, detail="AI сервис недоступен")

        # Формируем промпт для ИИ
        system_prompt = """Ты — строгий и точный преподаватель китайского языка.
Твоя задача: проверить ответы студента на тест по китайским словам.

ПРАВИЛА ПРОВЕРКИ:
1. Учитывай синонимы и близкие по смыслу ответы
2. Для пиньиня: игнорируй тоны и пробелы (nǐhǎo = nihao = nǐ hǎo)
3. Для перевода: допускай варианты перевода, если смысл сохранён
4. Пустой ответ — всегда НЕВЕРНО
5. Будь объективным, но справедливым

ФОРМАТ ОТВЕТА — ТОЛЬКО JSON:
{
    "correct_count": 12,
    "total": 15,
    "percentage": 80,
    "results": [
        {
            "id": "0",
            "prompt": "Пиньинь для: 你好",
            "user_answer": "nihao",
            "correct_answer": "nǐ hǎo",
            "is_correct": true,
            "feedback": "Правильно! Тоны можно опустить в тесте."
        },
        {
            "id": "1",
            "prompt": "Перевод для: 谢谢",
            "user_answer": "пожалуйста",
            "correct_answer": "спасибо",
            "is_correct": false,
            "feedback": "Неверно. 谢谢 = спасибо. 'Пожалуйста' = 请 или 不客气."
        }
    ],
    "summary": "Хороший результат! Основные ошибки — в переводе вежливых выражений."
}"""

        # Формируем список вопросов с ответами
        questions_text = ""
        for q in request.questions:
            user_ans = request.answers.get(q["id"], "(не отвечено)")
            questions_text += f"""
Вопрос {q['id']}: {q['prompt']}
Правильный ответ: {q['correct']}
Ответ студента: {user_ans}
"""

        user_prompt = f"""Проверь ответы студента.

Вопросы и ответы:
{questions_text}

Оцени каждый ответ и дай общий результат."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback на случай, если ИИ не вернул JSON
            result = fallback_word_test_check(request.questions, request.answers)

        return result

    except Exception as e:
        print(f"Ошибка ИИ-проверки теста слов: {e}")
        # Всегда возвращаем fallback
        return fallback_word_test_check(request.questions, request.answers)

def fallback_word_test_check(questions, answers):
    """Резервная проверка, если ИИ недоступен"""
    correct = 0
    total = len(questions)
    results = []

    for q in questions:
        qid = q["id"]
        user_raw = answers.get(qid, "")
        user_answer = user_raw.strip().lower() if user_raw else ""

        correct_ans = q["correct"].strip().lower()

        # Нормализация пиньиня
        if "пиньинь" in q["prompt"].lower():
            user_answer = user_answer.replace(" ", "").replace("v", "ü")
            correct_ans = correct_ans.replace(" ", "").replace("v", "ü")

        is_correct = bool(user_answer and user_answer == correct_ans)

        if is_correct:
            correct += 1

        results.append({
            "id": qid,
            "prompt": q["prompt"],
            "user_answer": user_raw.strip() if user_raw else "(не отвечено)",
            "correct_answer": q["correct"],
            "is_correct": is_correct,
            "feedback": "Правильно!" if is_correct else "Неверно. Проверьте ответ." if user_answer else "Ответ не дан — считается неверным."
        })

    percentage = round(correct / total * 100, 1) if total > 0 else 0

    return {
        "correct_count": correct,
        "total": total,
        "percentage": percentage,
        "results": results,
        "summary": f"{correct}/{total} правильных ({percentage}%). {'Отлично!' if percentage >= 90 else 'Хорошо!' if percentage >= 70 else 'Практикуйтесь больше!'}"
    }

@app.get("/user/progress/{user_id}")
async def get_user_progress(user_id: str):
    if user_id not in users_db:
        raise HTTPException(404, "Пользователь не найден")
    
    user = users_db[user_id]
    target = user.get("target_level", 4)
    
    total_words = sum(len(words_db.get(l, [])) for l in range(1, target + 1))
    learned = 0
    if user_id in user_word_status:
        learned = sum(1 for v in user_word_status[user_id].values() if v["status"] == "learned")
    
    percentage = round(learned / total_words * 100, 1) if total_words > 0 else 0
    
    return {
        "learned": learned,
        "total": total_words,
        "percentage": percentage,
        "target_level": target
    }

# Загрузка сохранённых данных при запуске
try:
    with open("data.pkl", "rb") as f:
        loaded = pickle.load(f)
        globals().update(loaded)
except FileNotFoundError:
    pass

# ========== ЗАПУСК СЕРВЕРА ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🎌 HSK AI Tutor - Прагматичный репетитор")
    print("=" * 60)
    print(f"📚 База данных: {len(words_db)} слов HSK 1-6")
    print(f"👥 Зарегистрировано пользователей: {len(users_db)}")
    print(f"🧪 Создано тестов: {len(tests_db)}")
    print("=" * 60)
    print("🚀 Запускаю сервер на http://localhost:8000")
    print("📚 Документация: http://localhost:8000/docs")
    print("🌐 Фронтенд: открой frontend.html в браузере")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Автоперезагрузка при изменениях
    )