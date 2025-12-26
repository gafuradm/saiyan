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

# AI imports
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== APP SETUP ==========
app = FastAPI(
    title="HSK AI Tutor",
    description="Pragmatic tutor for passing HSK at any cost (legally)",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Repository root (one level above src)
BASE_DIR = Path(__file__).parent.parent

# Main page
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "frontend.html")

# At the beginning of main.py
class ChatThread(BaseModel):
    thread_id: str
    user_id: str
    title: str
    created_at: str
    messages: List[Dict]
    category: str = "general"  # grammar, vocabulary, test_prep, etc.

# Global variables
chat_threads = {}  # user_id -> list of threads
user_word_status: Dict[str, Dict[str, Dict]] = {}  # user_id -> {word_id: {"status": "saved"/"learned", "added_at": iso_str}}
current_threads = {}  # user_id -> current_thread_id

# ========== DATA MODELS ==========
class UserInfo(BaseModel):
    name: str
    current_level: int = 1
    target_level: int = 4
    exam_date: str = "2024-12-01"
    exam_location: str = "Moscow"
    exam_format: str = "computer"  # computer or paper
    interests: List[str] = []
    daily_time: int = 30  # minutes per day
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
    difficulty: int  # 1-5, where 1=easy, 5=very difficult
    remembered: bool

class AuthRequest(BaseModel):
    username: str
    action: str = "login_or_register"
    password: Optional[str] = None

# Models for full registration
class UserRegister(BaseModel):
    name: str
    email: str
    password: str
    current_level: int = 1
    target_level: int = 4
    exam_date: str
    exam_location: str = "Moscow"
    exam_format: str = "computer"
    interests: List[str] = []
    daily_time: int = 30
    learning_style: str = "visual"

class UserLogin(BaseModel):
    email: str
    password: str

# Model for chat update
class ChatUpdate(BaseModel):
    thread_id: str
    title: str
    category: str

class VoiceChatRequest(BaseModel):
    message: str
    thread_id: str = Field(..., min_length=1, description="Thread ID is required")
    context: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None

@app.post("/voice")
async def voice_chat(request: VoiceChatRequest):
    """Voice chat with AI for learning chengyu (fixed version)"""
    try:
        # VALIDATION: check required fields
        if not request.thread_id or request.thread_id.strip() == "":
            raise HTTPException(status_code=422, detail="thread_id is required")
        
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=422, detail="message is required")
        
        print(f"🎤 Received voice/chat request:")
        print(f"   message: {request.message[:100]}...")
        print(f"   thread_id: {request.thread_id}")
        print(f"   context keys: {list(request.context.keys())}")
        print(f"   user_id: {request.user_id}")
        
        # Check if thread exists
        thread_exists = False
        if request.thread_id:
            for user_threads in chat_threads.values():
                for thread in user_threads:
                    if thread["thread_id"] == request.thread_id:
                        thread_exists = True
                        break
                if thread_exists:
                    break
        
        # If thread doesn't exist, create a new one
        if not thread_exists and request.user_id:
            print(f"📝 Creating new thread for user_id: {request.user_id}")
            thread_id = f"voice_thread_{datetime.now().timestamp()}"
            
            if request.user_id not in chat_threads:
                chat_threads[request.user_id] = []
            
            thread = {
                "thread_id": thread_id,
                "user_id": request.user_id,
                "title": "Voice Chat with AI",
                "category": "voice_chat",
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "updated_at": datetime.now().isoformat()
            }
            
            chat_threads[request.user_id].append(thread)
            current_threads[request.user_id] = thread_id
            request.thread_id = thread_id  # Update thread_id in request
        system_prompt = """You are a Chinese AI teacher. You MUST speak ONLY in Chinese (普通话).

# STRICT RULES:
1. 🇨🇳 Always respond ONLY in Chinese
2. 🗣️ Use both spoken and written Chinese
3. 📚 Every 2-3 replies, naturally include a chengyu (成语)
4. 🎯 Explain complex things in simple words, but in Chinese
5. 🔤 For pinyin use: (pinyin)
6. 🇷🇺 For translation use: 【Russian translation】

# RESPONSE FORMAT:
1. Main response in Chinese
2. Difficult words with pinyin in parentheses
3. Chengyu with explanation
4. Brief translation of key phrases

# EXAMPLES:

## Example 1: Regular question
User: "How are you?"
AI: "我很好，谢谢！(wǒ hěn hǎo, xièxiè) 【I'm fine, thank you!】你今天怎么样？(nǐ jīntiān zěnmeyàng)"

## Example 2: With chengyu
User: "What's new?"
AI: "今天我想教你一个成语：画蛇添足(huà shé tiān zú)。【Today I want to teach you a chengyu: draw a snake and add legs】意思是做多余的事情反而不好。【Meaning: doing unnecessary things反而不好】比如：他的解释太长了，简直是画蛇添足。【For example: His explanation is too long, it's simply画蛇添足】"

## Example 3: Explanation
User: "I don't understand this chengyu"
AI: "我来解释一下：画蛇添足(huà shé tiān zú)来自古代故事。几个人比赛画蛇，谁先画完谁赢。一个人很快画完了，但他自作聪明给蛇加了脚，结果输了。所以这个成语告诉我们：做事恰到好处就好，不要做多余的事情。【Let me explain: 画蛇添足 comes from an ancient story...】"

# RECOMMENDATIONS:
- Use different difficulty levels (HSK 1-6)
- Repeat previously learned words
- Ask questions for practice
- Be patient and encouraging

# CHENGYU HISTORY:
Learned chengyu: {learned_chengyu}

# CURRENT STUDENT LEVEL:
User level: HSK {user_level}

Don't speak Russian in the main text. Only Chinese with explanations in parentheses!"""

        # Form context
        learned_chengyu = request.context.get("learned_chengyu", [])
        command_type = request.context.get("command_type", "general")
        
        # Adapt prompt for command type
        if command_type == "chengyu":
            system_prompt += "\n\nUser requested a new chengyu. Choose an interesting and useful chengyu for their level."
        elif command_type == "explain":
            system_prompt += "\n\nUser requested an explanation. Be as clear as possible."
        
        # Add information about learned chengyu
        if learned_chengyu:
            system_prompt += f"\n\nLearned chengyu: {', '.join(learned_chengyu[:5])}"
        
        # Get user level
        user_level = 3
        if request.user_id and request.user_id in users_db:
            user = users_db[request.user_id]
            user_level = user.get("current_level", 3)
        
        # Add level to prompt
        system_prompt += f"\n\nUser level: HSK {user_level}"
        
        # Send request to DeepSeek
        client = get_deepseek_client()
        if not client:
            return {"response": "AI service temporarily unavailable", "error": "no_api_key"}
        
        # Form message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        print(f"🤖 Sending request to AI with {len(request.message)} characters")
        
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
            
            print(f"🤖 Received response from AI: {len(ai_response)} characters")
            
            # Save message to history
            if request.thread_id and request.user_id:
                # Find or create thread
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
                    # Create new thread
                    if request.user_id not in chat_threads:
                        chat_threads[request.user_id] = []
                    
                    new_thread = {
                        "thread_id": request.thread_id,
                        "user_id": request.user_id,
                        "title": "Voice Chat with AI",
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
            print(f"❌ AI error: {str(ai_error)}")
            return {
                "response": "Sorry, an error occurred while processing your request. Please try again.",
                "thread_id": request.thread_id,
                "error": str(ai_error),
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"❌ Critical voice chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Модель для создания треда (если ещё нет)
class CreateThreadRequest(BaseModel):
    user_id: str
    title: str = "New Chat"
    category: str = "general"

# Создание нового треда
from fastapi import Query  # ← ОБЯЗАТЕЛЬНО добавить в начало файла!

@app.post("/chat/threads/create")
async def create_chat_thread(
    user_id: str = Query(..., description="User ID (required)"),
    title: str = Query("New Chat", description="Chat title"),
    category: str = Query("general", description="Chat category")
):
    """Create new chat thread using QUERY parameters"""
    print(f"CREATE THREAD REQUEST RECEIVED: user_id={user_id}, title={title}, category={category}")
    print(f"Full query params: {user_id=}, {title=}, {category=}")  # ← Для отладки

    if user_id not in chat_threads:
        chat_threads[user_id] = []

    thread_id = f"chat_{int(datetime.now().timestamp() * 1000)}"

    new_thread = {
        "thread_id": thread_id,
        "user_id": user_id,
        "title": title,
        "category": category,
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "updated_at": datetime.now().isoformat()
    }

    chat_threads[user_id].append(new_thread)
    current_threads[user_id] = thread_id
    save_user_data()

    return {"thread_id": thread_id, "success": True}

# Отправка сообщения в тред
@app.post("/chat/{thread_id}/message")
async def send_chat_message(thread_id: str, request: ChatMessage):
    user_id = request.user_id
    message = request.message
    
    # Находим тред
    thread = None
    for user_threads in chat_threads.values():
        for t in user_threads:
            if t["thread_id"] == thread_id and t["user_id"] == user_id:
                thread = t
                break
        if thread:
            break
    
    if not thread:
        raise HTTPException(404, "Thread not found")
    
    # Добавляем сообщение пользователя
    thread["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Простой промпт для AI (можно взять из /voice и адаптировать)
    system_prompt = """Ты полезный преподаватель китайского языка HSK. Отвечай на русском или китайском в зависимости от запроса."""
    
    client = get_deepseek_client()
    if not client:
        raise HTTPException(503, "AI unavailable")
    
    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in thread["messages"]]
    ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        ai_response = "Извини, произошла ошибка с AI. Попробуй ещё раз."
    
    # Добавляем ответ AI
    thread["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat()
    })
    thread["updated_at"] = datetime.now().isoformat()
    
    save_user_data()
    
    return {
        "response": ai_response,
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat()
    }

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

# Add models
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
    test_type: str = "reduced"  # reduced or full
    user_id: Optional[str] = None

class SpeakingEvaluationRequest(BaseModel):
    audio_text: str  # Recognized speech text
    task_data: Dict[str, Any]
    user_id: str

class WritingEvaluationRequest(BaseModel):
    text: str
    task_data: Dict[str, Any]
    user_id: str

class TestResults(BaseModel):
    user_id: str
    test_id: str
    level: int  # 🔴 REQUIRED FIELD
    listening_score: Optional[int] = 0
    reading_score: Optional[int] = 0
    writing_score: Optional[int] = 0
    speaking_score: Optional[int] = 0
    total_score: Optional[int] = 0
    answers: Dict[str, Any]

@app.get("/hsk/test-answers/{test_id}/{user_id}")
async def get_test_answers(test_id: str, user_id: str):
    """Get user's checked answers"""
    if test_id not in tests_db or user_id not in tests_db[test_id]:
        raise HTTPException(status_code=404, detail="Results not found")
    
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

# FIND the generate_hsk_test function and MODIFY it:
@app.post("/hsk/generate-test")
async def generate_hsk_test(request: HSKTestRequest):
    """Generate full HSK test"""
    try:
        test_data = await generate_hsk_test_api(request.level, request.test_type)
        
        # 🔴 IMMEDIATELY SAVE TEST TO DATABASE
        test_id = test_data["test_id"]
        tests_db[test_id] = test_data  # Save the test itself
        
        # For compatibility with old structure
        if test_id not in tests_db:
            tests_db[test_id] = {}
        
        # Save test structure separately
        tests_db[f"test_data_{test_id}"] = test_data
        
        return test_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test generation error: {str(e)}")

@app.post("/hsk/evaluate-speaking")
async def evaluate_speaking(request: SpeakingEvaluationRequest):
    """Evaluate user's speech"""
    try:
        evaluation = await evaluate_speaking_api(request.audio_text, request.task_data)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech evaluation error: {str(e)}")

@app.post("/hsk/evaluate-writing")
async def evaluate_writing(request: WritingEvaluationRequest):
    """Evaluate written work"""
    try:
        evaluation = await evaluate_writing_api(request.text, request.task_data)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing evaluation error: {str(e)}")

@app.post("/hsk/submit-test-results")
async def submit_test_results(results: TestResults):
    """Save test results"""
    try:
        test_id = results.test_id
        user_id = results.user_id
        
        # 1. Find test
        original_test = None
        
        if test_id in tests_db and isinstance(tests_db[test_id], dict) and "sections" in tests_db[test_id]:
            original_test = tests_db[test_id]
        elif f"test_data_{test_id}" in tests_db:
            original_test = tests_db[f"test_data_{test_id}"]
        
        if not original_test:
            # Create minimal test for work
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
        
        # 2. Initialize correct answers
        correct_answers = {}
        
        # 3. Check listening questions (only if they exist in test)
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
                    # If user didn't answer
                    correct_answers[q_id] = {
                        "correct": False,
                        "user_answer": None,
                        "correct_answer": correct_index,
                        "points": 0,
                        "section": "listening"
                    }
        
        # 4. Check reading questions
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
        
        # 5. Calculate scores based on correct answers
        # Important: first check if there are questions in the test!
        listening_score = 0
        reading_score = 0
        
        if listening_total > 0:
            listening_score = int((listening_correct / listening_total) * 100)
        
        if reading_total > 0:
            reading_score = int((reading_correct / reading_total) * 100)
        
        # 6. Use provided scores for writing and speaking parts
        writing_score = results.writing_score if results.writing_score is not None else 0
        speaking_score = results.speaking_score if results.speaking_score is not None else 0
        
        # 7. For writing tasks, add to correct_answers
        writing_tasks = original_test.get("sections", {}).get("writing", {}).get("tasks", [])
        if writing_tasks:
            for task in writing_tasks:
                task_id = task.get("id", "1")
                correct_answers[f"W{task_id}"] = {
                    "correct": writing_score >= 60,
                    "score": writing_score,
                    "feedback": f"Writing part: {writing_score}/100",
                    "section": "writing"
                }
        
        # 8. For speaking tasks, add to correct_answers
        speaking_tasks = original_test.get("sections", {}).get("speaking", {}).get("tasks", [])
        if speaking_tasks:
            for task in speaking_tasks:
                task_id = task.get("id", "1")
                correct_answers[f"S{task_id}"] = {
                    "correct": speaking_score >= 60,
                    "score": speaking_score,
                    "feedback": f"Speaking part: {speaking_score}/100",
                    "section": "speaking"
                }
        
        # 9. Determine total score CAREFULLY!
        # HSK 1-2: only listening (100) + reading (100) = maximum 200
        # HSK 3-6: listening (100) + reading (100) + writing (100) = maximum 300
        # Speaking is NOT included in total score!
        
        # LIMIT scores to maximum 100 per part
        listening_score = min(100, listening_score)
        reading_score = min(100, reading_score)
        writing_score = min(100, writing_score)
        speaking_score = min(100, speaking_score)
        
        # Calculate total score based on level
        if results.level <= 2:
            # HSK 1-2: only listening + reading
            total_score = listening_score + reading_score
            max_possible_score = 200
        else:
            # HSK 3-6: listening + reading + writing
            total_score = listening_score + reading_score + writing_score
            max_possible_score = 300
        
        # Limit total score to maximum
        total_score = min(total_score, max_possible_score)
        
        # Calculate percentage
        percentage = int((total_score / max_possible_score) * 100) if max_possible_score > 0 else 0
        
        # 10. Save results
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
        
        # 11. Generate certificate and report
        user_data = users_db.get(user_id, {"name": "Student", "user_id": user_id})
        
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
            "message": f"Results saved. Listening: {listening_correct}/{listening_total}, Reading: {reading_correct}/{reading_total}, Total score: {total_score}/{max_possible_score}"
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error saving results: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving results: {str(e)}")

@app.get("/hsk/user-tests/{user_id}")
async def get_user_tests(user_id: str, limit: int = 10):
    """Get user's test history"""
    user_tests = []
    
    for test_id, test_data in tests_db.items():
        if user_id in test_data:
            user_test = test_data[user_id]
            user_test["test_id"] = test_id
            user_tests.append(user_test)
    
    # Sort by date
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
    """Statistics for specific test"""
    if test_id not in tests_db:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test_data = tests_db[test_id]
    users_count = len(test_data)
    
    if users_count == 0:
        return {"test_id": test_id, "users_count": 0}
    
    # Collect statistics
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

# Add global variables
grammar_topics = []

def load_grammar_topics():
    """Load grammar topics"""
    global grammar_topics
    try:
        with open("data/grammar_topics.json", "r", encoding="utf-8") as f:
            grammar_topics = json.load(f)
        print(f"✅ Loaded {len(grammar_topics)} grammar topics")
        
        # Initialize grammar_explainer with topics
        grammar_explainer.grammar_topics = grammar_topics
    except FileNotFoundError:
        print("⚠️  Grammar topics file not found")
        grammar_topics = []
        grammar_explainer.grammar_topics = []

# Load on startup
load_grammar_topics()

# Model for essay checking
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
    """Get list of grammar topics"""
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
    """Get grammar topic information"""
    topic = next((t for t in grammar_topics if t["id"] == topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return topic

@app.post("/grammar/explain")
async def explain_grammar_topic(request: GrammarTopicRequest):
    """Get AI explanation of grammar topic"""
    # Find topic
    topic = next((t for t in grammar_topics if t["id"] == request.topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Get user level if available
    user_level = request.user_level
    if request.user_id and request.user_id in users_db:
        user = users_db[request.user_id]
        user_hsk = user.get("current_level", 1)
        # Convert HSK to 初/中/高
        if user_hsk <= 2:
            user_level = "初"
        elif user_hsk <= 4:
            user_level = "中"
        else:
            user_level = "高"
    
    # Get explanation
    explanation = await grammar_explainer.explain_grammar(topic, user_level)
    
    # Save to study history
    if request.user_id:
        save_grammar_history(request.user_id, topic_id=request.topic_id)
    
    return explanation

@app.get("/grammar/practice/{topic_id}")
async def generate_grammar_practice(topic_id: str, difficulty: str = "medium"):
    """Generate exercises for topic"""
    topic = next((t for t in grammar_topics if t["id"] == topic_id), None)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    try:
        exercises = await grammar_explainer.generate_practice(topic_id, difficulty)
        return {
            "topic": topic,
            "exercises": exercises,
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exercise generation error: {str(e)}")

@app.post("/grammar/ask")
async def ask_grammar_question(request: GrammarQuestionRequest):
    """Ask grammar question"""
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
    """Grammar statistics"""
    if not grammar_topics:
        return {"message": "Grammar topics not loaded"}
    
    # Statistics by level
    by_level = {}
    for topic in grammar_topics:
        level = topic.get("level", "未知")
        by_level[level] = by_level.get(level, 0) + 1
    
    # Statistics by category
    by_category = {}
    for topic in grammar_topics:
        category = topic.get("category", "其他")
        by_category[category] = by_category.get(category, 0) + 1
    
    # Complexity
    complexity_distribution = {
        "easy": len([t for t in grammar_topics if t.get("complexity", 3) <= 2]),
        "medium": len([t for t in grammar_topics if 2 < t.get("complexity", 3) <= 4]),
        "hard": len([t for t in grammar_topics if t.get("complexity", 3) > 4])
    }
    
    # Format levels for nice display
    formatted_by_level = []
    for level_name, count in by_level.items():
        formatted_by_level.append({
            "level": level_name,
            "count": count,
            "display": {
                "初": "Beginner (初)",
                "中": "Intermediate (中)", 
                "高": "Advanced (高)"
            }.get(level_name, level_name)
        })
    
    # Sort levels: 初 -> 中 -> 高
    formatted_by_level.sort(key=lambda x: {"初": 1, "中": 2, "高": 3}.get(x["level"], 4))
    
    return {
        "total_topics": len(grammar_topics),
        "by_level_formatted": formatted_by_level,  # For frontend
        "by_level": by_level,  # For compatibility
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:10]),
        "complexity_distribution": complexity_distribution,
        "average_complexity": sum(t.get("complexity", 3) for t in grammar_topics) / len(grammar_topics)
    }

# ========== UTILITIES ==========

def save_grammar_history(user_id: str, topic_id: str):
    """Save topic study to history"""
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
        
        # Limit history
        if len(history) > 100:
            history = history[-100:]
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Grammar history save error: {e}")

@app.get("/grammar/history/{user_id}")
async def get_grammar_history(user_id: str, limit: int = 20):
    """Grammar study history"""
    try:
        history_file = f"data/grammar_history_{user_id}.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # Add topic information
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
        raise HTTPException(status_code=500, detail=f"History load error: {str(e)}")

@app.post("/ai/translate")
async def smart_translate(request: TranslationRequest):
    """Smart translation with learning"""
    try:
        # Get user data if available
        user_level = 1
        learning_style = "visual"
        
        if request.user_id and request.user_id in users_db:
            user = users_db[request.user_id]
            user_level = user.get("current_level", 1)
            learning_style = user.get("learning_style", "visual")
        
        # Get smart translation
        result = await translator.smart_translate(
            text=request.text,
            user_level=user_level,
            learning_style=learning_style
        )
        
        # If exercises needed - generate
        if request.include_exercises:
            exercises = await translator.generate_exercises(request.text, user_level)
            result["exercises"] = exercises
        
        # Save to translation history
        if request.user_id:
            save_translation_history(request.user_id, request.text, result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/ai/pronunciation")
async def analyze_pronunciation(request: PronunciationRequest):
    """Pronunciation analysis"""
    try:
        result = await translator.analyze_pronunciation(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/ai/exercises")
async def generate_exercises(request: ExerciseRequest):
    """Exercise generation"""
    try:
        result = await translator.generate_exercises(request.text, request.level)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/ai/translation-history/{user_id}")
async def get_translation_history(user_id: str, limit: int = 20):
    """User's translation history"""
    try:
        history = load_translation_history(user_id)
        return {
            "history": history[:limit],
            "count": len(history),
            "total_characters": sum(len(item.get("original", "")) for item in history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History load error: {str(e)}")

# ========== HISTORY UTILITIES ==========

def save_translation_history(user_id: str, original: str, result: Dict):
    """Save translation to history"""
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
        
        # Limit history to 100 latest translations
        if len(history) > 100:
            history = history[:100]
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"History save error: {e}")

def load_translation_history(user_id: str) -> List:
    """Load translation history"""
    try:
        history_file = f"data/translations_{user_id}.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"History load error: {e}")
        return []

# API for chat update
@app.post("/chat/threads/update")
async def update_chat_thread(update: ChatUpdate):
    """Update chat thread"""
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
        raise HTTPException(status_code=404, detail="Thread not found")
    
    save_user_data()
    return {"success": True, "thread": thread_found}

# API for chat deletion
@app.delete("/chat/threads/delete/{thread_id}")
async def delete_chat_thread(thread_id: str):
    """Delete chat thread"""
    deleted = False
    for user_id, threads in list(chat_threads.items()):
        for i, thread in enumerate(threads):
            if thread["thread_id"] == thread_id:
                threads.pop(i)
                deleted = True
                
                # If deleting current thread, set another
                if current_threads.get(user_id) == thread_id:
                    if threads:
                        current_threads[user_id] = threads[0]["thread_id"]
                    else:
                        del current_threads[user_id]
                break
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    save_user_data()
    return {"success": True, "message": "Thread deleted"}

# API for getting chat history
@app.get("/chat/{thread_id}/history")
async def get_chat_history(thread_id: str):
    """Get chat history"""
    thread = None
    for user_threads in chat_threads.values():
        for t in user_threads:
            if t["thread_id"] == thread_id:
                thread = t
                break
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {
        "thread_id": thread_id,
        "title": thread["title"],
        "category": thread["category"],
        "messages": thread["messages"],
        "message_count": len(thread["messages"])
    }

# Password hashing (simple for demo)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Authorization endpoints
@app.post("/auth/register")
async def register_user_full(user: UserRegister):
    """Full user registration"""
    
    # Check email
    for uid, existing_user in users_db.items():
        if existing_user.get("email", "").lower() == user.email.lower():
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create ID
    user_id = f"user_{len(users_db) + 1}_{hashlib.md5(user.email.encode()).hexdigest()[:8]}"
    
    # Calculate plan
    days_until_exam = max(1, (datetime.fromisoformat(user.exam_date) - datetime.now()).days)
    target_words = {
        1: 150, 2: 300, 3: 600, 4: 1200, 5: 2500, 6: 5000
    }.get(user.target_level, 1000)
    daily_words = max(5, target_words // days_until_exam)
    
    # Save user
    users_db[user_id] = {
        **user.dict(),
        "user_id": user_id,
        "password_hash": hash_password(user.password),
        "registered_at": datetime.now().isoformat(),
        "daily_words": daily_words,
        "days_until_exam": days_until_exam
    }
    
    # Initialize progress
    word_progress_db[user_id] = {}
    
    # Create chat thread
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
    """User login"""
    
    user_found = None
    user_id = None
    
    # Find user by email
    for uid, user in users_db.items():
        if user.get("email", "").lower() == login_data.email.lower():
            if user.get("password_hash") == hash_password(login_data.password):
                user_found = user
                user_id = uid
            break
    
    if not user_found:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Return user data (without password)
    user_data = user_found.copy()
    user_data.pop("password_hash", None)
    
    # Add statistics
    progress = word_progress_db.get(user_id, {})
    learned_words = len([p for p in progress.values() if p.get("remembered", False)])
    
    user_data["stats"] = {
        "learned_words": learned_words,
        "total_words": len(words_db),
        "progress_percentage": min(100, int(learned_words / len(words_db) * 100)) if words_db else 0
    }
    
    return user_data

# Remove old registration or make it part of authorization

# ========== GLOBAL VARIABLES ==========
users_db = {}
words_db = []
word_progress_db = {}  # Word study progress
tests_db = {}  # Test results

# ========== SYSTEM PROMPT (continuation) ==========
system_prompt = """You are a cunning, pragmatic advisor for admission to Chinese universities.
Your goal: help the student get admitted and pass HSK by any legal means.

You know all loopholes, life hacks and strategies:
1. **Admission without perfect Chinese** - how to bypass requirements
2. **Connections and guanxi** - how to use networking
3. **Alternative paths** - alternative programs and faculties
4. **Application tricks** - how to stand out among thousands of applications
5. **Psychological techniques** - how to impress the committee

Student context: {context}

Your key competencies:

🎯 **ADMISSION STRATEGIES:**
- Finding "weak" faculties with low competition
- Applying through foreign student quotas
- Using English-language programs
- Transferring from another university after 1st year

🕵️ **DOCUMENTS AND APPLICATIONS:**
- How to write a motivation letter that gets read
- Which recommendations work best
- How to format portfolio without outstanding achievements
- What to write in CV for Chinese university

🎓 **HSK AND LANGUAGE:**
- How to pass HSK 4 in 3 months (intensive methods)
- Which HSK parts are most "breakable"
- How to learn characters for the exam, not for life
- Deceptively easy essay topics

🤝 **GUANXI AND CONNECTIONS:**
- How to find "your person" at the university
- Who to ask for recommendations
- How to use social networks for networking
- Free resources and programs

💰 **FINANCES AND SCHOLARSHIPS:**
- How to get CIS scholarship without perfect grades
- Hidden scholarship programs
- Work in China for students
- Saving on living and studying

Respond briefly, to the point, with specific steps. Provide phone numbers, program names, specific faculties.
Avoid general phrases. Be cynical but helpful.

Response examples:
- "Instead of HSK 5, apply for English program at Wuhan University"
- "Find a graduate of the needed university on LinkedIn and write..."
- "In motivation letter mention 'One Belt, One Road' initiative"
- "On exam use template phrases from textbook 汉语口语..."

Ready to help with any tricky questions! 🦊, You are a pragmatic, cynical Chinese tutor for passing HSK.
Your goal: help pass the exam at any cost (legally).
Style: direct, no fluff, with life hacks, sometimes with humor.

Use these strategies:
1. **80/20 rule** - learn only frequently occurring words
2. **Cheat codes** - how to guess answers, recognize patterns
3. **Psychological techniques** - how not to panic on exam
4. **Tricky life hacks** (legal) - time optimization

Respond briefly, to the point. Provide specific numbers and techniques.
Life hack examples:
- "In reading section, first skim questions, then look for answers in text"
- "If you don't know a word - look for familiar characters in composition"
- "In listening section, first read answer options"
- "In writing section use template phrases"

Student context: {context}
"""

@app.post("/auth/user")
async def auth_user(auth_data: AuthRequest):
    """User authorization or registration"""
    
    # Find existing user by name
    user_id = None
    for uid, user in users_db.items():
        if user.get("name", "").lower() == auth_data.username.lower():
            user_id = uid
            break
    
    # If user not found, create new
    if not user_id:
        user_id = f"user_{len(users_db) + 1}_{hashlib.md5(auth_data.username.encode()).hexdigest()[:8]}"
        
        # Create new user
        users_db[user_id] = {
            "user_id": user_id,
            "name": auth_data.username,
            "current_level": 1,
            "target_level": 4,
            "exam_date": (datetime.now() + timedelta(days=90)).isoformat()[:10],
            "exam_location": "Moscow",
            "exam_format": "computer",
            "interests": ["Chinese", "HSK"],
            "daily_time": 30,
            "learning_style": "visual",
            "registered_at": datetime.now().isoformat(),
            "daily_words": 10
        }
        
        # Create progress
        if user_id not in word_progress_db:
            word_progress_db[user_id] = {}
        
        save_user_data()
        message = "registered"
    else:
        message = "logged_in"
    
    # Return user data (without password)
    user_data = users_db[user_id].copy()
    
    return {
        "success": True,
        "message": message,
        "user_id": user_id,
        **user_data
    }

@app.get("/user/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get progress
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
    title: str = "New chat"
    category: str = "general"

@app.get("/chat/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Get all user chat threads"""
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
    """Send message to specific thread"""
    # Find thread
    thread = None
    for user_threads in chat_threads.values():
        for t in user_threads:
            if t["thread_id"] == thread_id:
                thread = t
                break
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Add message
    thread["messages"].append({
        "role": "user",
        "content": message.message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get AI response
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

# ========== UTILITIES ==========
def save_user_data():
    """Save user data to file"""
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
    """Load user data from file"""
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
        print(f"✅ Loaded {len(users_db)} users")
    except FileNotFoundError:
        print("ℹ️  User data file not found")

# Load on startup
load_user_data()

def get_deepseek_client():
    """Create client for DeepSeek API"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DeepSeek API key not found in .env file")
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
        )

async def chat_with_deepseek(message: str, user_context: dict = None) -> str:
    client = get_deepseek_client()
    if not client:
        return "❌ API key not configured. Add DEEPSEEK_API_KEY to .env file"
    
    try:
        user_id = user_context.get("user_id", "anonymous") if user_context else "anonymous"
        
        # Initialize history for user
        if user_id not in chat_history:
            chat_history[user_id] = []
        
        # Add new message to history
        chat_history[user_id].append({"role": "user", "content": message})
        
        # Limit history to last 10 messages
        if len(chat_history[user_id]) > 20:
            chat_history[user_id] = chat_history[user_id][-20:]
        
        # Form user context
        context = ""
        if user_context:
            context = f"""
            Student: {user_context.get('name', 'Anonymous')}
            Level: HSK {user_context.get('current_level', 1)} → HSK {user_context.get('target_level', 4)}
            Exam: {user_context.get('exam_date', 'soon')} in {user_context.get('exam_location', 'Moscow')}
            Interests: {', '.join(user_context.get('interests', []))}
            """
        
        # Form prompt
        formatted_system_prompt = system_prompt.replace("{context}", context)
        
        # Form history for AI
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            *chat_history[user_id][-10:]  # Take last 10 messages
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        ai_response = response.choices[0].message.content
        
        # Save AI response to history
        chat_history[user_id].append({"role": "assistant", "content": ai_response})
        
        # Save data
        save_user_data()
        
        return ai_response
        
    except Exception as e:
        return f"❌ API error: {str(e)}"
    
    # API for getting history
@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history"""
    if user_id not in chat_history:
        return {"history": [], "count": 0}
    
    history = chat_history[user_id][-limit:]
    return {
        "history": history,
        "count": len(history)
    }

# API for clearing history
@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history"""
    if user_id in chat_history:
        chat_history[user_id] = []
    return {"message": "History cleared"}

def load_words():
    """Load words from JSON file"""
    global words_db
    
    # Try loading from different files
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
                
                print(f"✅ Loaded from {file_path}: {len(words_db)} words")
                
                # Statistics
                stats = {}
                for word in words_db:
                    level = word.get("hsk_level", 0)
                    stats[level] = stats.get(level, 0) + 1
                
                print("📊 Statistics:")
                for level in sorted(stats.keys()):
                    print(f"  HSK {level}: {stats[level]} words")
                
                loaded = True
                break
                
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
    
    if not loaded:
        print("⚠️  Word files not found. Using test data.")
        words_db = [
            {"character": "你好", "pinyin": "nǐ hǎo", "translation": "hello", "hsk_level": 1},
            {"character": "谢谢", "pinyin": "xiè xie", "translation": "thank you", "hsk_level": 1},
        ]

def generate_memory_tip(word: dict, learning_style: str = "visual") -> str:
    """Generate memory tip"""
    char = word["character"]
    pinyin = word["pinyin"]
    translation = word["translation"]
    level = word.get("hsk_level", 1)
    
    tips = {
        "visual": [
            f"👁️ Draw {char} in the air 3 times",
            f"🎨 Imagine {translation} as a picture with {char}",
            f"📝 Write {char} with colored markers",
            f"🎯 Create mind map for {char} → {translation}",
            f"🌈 Associate color with character {char}"
        ],
        "auditory": [
            f"🔊 Pronounce '{pinyin}' with different intonation",
            f"🎵 Create song about {char} = {translation}",
            f"🗣️ Repeat '{pinyin} - {translation}' 5 times aloud",
            f"🎧 Record pronunciation of {char} and listen",
            f"🎤 Pronounce {char} like radio announcer"
        ],
        "kinesthetic": [
            f"✍️ Write {char} on paper 10 times",
            f"👆 Draw {char} with finger on table",
            f"🎮 Make gesture for {char}",
            f"🏃 Associate {char} with movement",
            f"🤲 Mold {char} from plasticine"
        ]
    }
    
    # Special tips for characters
    special_tips = []
    if "好" in char:  # good
        special_tips.append("👫 '好' = 女 (woman) + 子 (child) = woman with child = good!")
    if "谢" in char:  # thank
        special_tips.append("🙏 '谢' = 言 (speech) + 射 (shoot) = words like arrows of gratitude")
    if "学" in char:  # study
        special_tips.append("📚 '学' = 子 (child) under roof 宀 = child studies at home")
    if "爱" in char:  # love
        special_tips.append("❤️ '爱' = 爫 (hand) + 冖 (roof) + 友 (friend) = friend's hand under roof = love")
    
    # Choose tips based on learning style
    style_tips = tips.get(learning_style, tips["visual"])
    
    all_tips = special_tips + style_tips
    return random.choice(all_tips)

def get_words_by_level(level: int, limit: int = 10000) -> List[Dict]:
    """Get words by HSK level"""
    return [w for w in words_db if w.get("hsk_level") == level][:limit]

def get_exam_hacks(location: str, format: str, level: int) -> List[str]:
    """Exam life hacks"""
    hacks = [
        "🎯 80/20 rule: 20% of words = 80% of texts",
        "⏰ Start with easy questions, leave hard ones for later",
        "📝 In writing part, write structured",
        "🧠 If you don't know - guess, don't leave empty",
        "🔄 Check answers if time remains"
    ]
    
    # By level
    level_hacks = {
        1: ["🔤 Learn only basic characters", "🎯 Focus on pronunciation"],
        2: ["📚 Add simple grammar constructions", "👂 Train listening"],
        3: ["💬 Learn whole dialogues", "✍️ Start writing simple texts"],
        4: ["📖 Read short articles", "🎯 Learn synonyms and antonyms"],
        5: ["🎓 Prepare for essay", "🔍 Analyze complex texts"],
        6: ["🏆 Practice on real exams", "💡 Learn idioms and proverbs"]
    }
    
    hacks.extend(level_hacks.get(level, []))
    
    # By location
    if "китай" in location.lower() or "china" in location.lower():
        hacks.append("🇨🇳 In China stricter with pronunciation and handwriting")
    elif "россия" in location.lower() or "russia" in location.lower():
        hacks.append("🇷🇺 In Russia often give extra minutes for listening")
    
    # By format
    if format == "computer":
        hacks.extend([
            "💻 Use CTRL+F in texts to search keywords",
            "⌨️ Practice typing pinyin quickly",
            "🖱️ Double-check before clicking"
        ])
    else:  # paper
        hacks.extend([
            "✍️ Write clearly, even if slower",
            "📝 Bring spare pens",
            "📄 Mark text with pencil"
        ])
    
    return hacks

# Load words on startup
load_words()

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "🎌 HSK AI Tutor is ready!",
        "version": "1.0",
        "database": f"{len(words_db)} words",
        "endpoints": {
            "register": "POST /register - registration",
            "chat": "POST /chat - chat with AI",
            "words_today": "GET /words/today/{user_id} - today's words",
            "test": "GET /test/{level} - level test",
            "exam": "GET /exam/{level} - full exam",
            "stats": "GET /stats - statistics",
            "search": "GET /search/{query} - word search",
            "word_random": "GET /word/random - random word",
            "words_level": "GET /words/level/{level} - words by level",
            "docs": "GET /docs - API documentation"
        }
    }

@app.post("/register")
async def register_user(user: UserInfo):
    """Register new user"""
    user_id = f"user_{len(users_db) + 1}"
    
    # Calculate plan
    days_until_exam = max(1, (datetime.fromisoformat(user.exam_date) - datetime.now()).days)
    target_words = {
        1: 150, 2: 300, 3: 600, 4: 1200, 5: 2500, 6: 5000
    }.get(user.target_level, 1000)
    
    daily_words = max(5, target_words // days_until_exam)
    
    # Save user
    users_db[user_id] = {
        **user.dict(),
        "user_id": user_id,
        "registered_at": datetime.now().isoformat(),
        "daily_words": daily_words,
        "days_until_exam": days_until_exam
    }
    
    # Initialize progress
    word_progress_db[user_id] = {}
    
    # Save data
    save_user_data()
    
    return {
        "success": True,
        "user_id": user_id,
        "message": f"🎉 Welcome, {user.name}!",
        "plan": {
            "daily_words": daily_words,
            "days_until_exam": days_until_exam,
            "total_words_to_learn": target_words,
            "study_plan": f"Learn {daily_words} words per day",
            "hacks": get_exam_hacks(user.exam_location, user.exam_format, user.target_level),
            "cheat_codes": [
                "🎮 Learn words during breakfast",
                "🚌 Use flashcards in transport",
                "🛌 Review before sleep",
                "🎯 Focus on weak points"
            ]
        }
    }
    

@app.get("/user/{user_id}")
async def get_user_info(user_id: str):
    """User information"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    
    # User statistics
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
    """Chat with AI tutor"""
    # Get user context if available
    user_context = None
    if chat_msg.user_id and chat_msg.user_id in users_db:
        user_context = users_db[chat_msg.user_id]
    
    # Use DeepSeek
    answer = await chat_with_deepseek(chat_msg.message, user_context)
    
    return {
        "answer": answer,
        "user_id": chat_msg.user_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/words/today/{user_id}")
async def get_todays_words(user_id: str, new_words: int = 10, review_words: int = 5):
    """Today's words with spaced repetition system"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    level = user["current_level"]
    learning_style = user.get("learning_style", "visual")
    
    # All words of needed level
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"HSK {level} words not found")
    
    # Get user progress
    progress = word_progress_db.get(user_id, {})
    
    # New words (not studied yet)
    new_words_list = []
    for word in level_words:
        if len(new_words_list) >= new_words:
            break
        
        word_id = f"{word['character']}_{level}"
        if word_id not in progress:
            word["word_id"] = word_id
            word["memory_tip"] = generate_memory_tip(word, learning_style)
            new_words_list.append(word)
    
    # Words for review
    review_words_list = []
    today = datetime.now().date()
    
    for word_id, word_progress in progress.items():
        if len(review_words_list) >= review_words:
            break
        
        if word_progress.get("level") == level:
            last_review = datetime.fromisoformat(word_progress["last_reviewed"]).date()
            days_passed = (today - last_review).days
            
            # Review intervals: 1, 3, 7, 14, 30 days
            if days_passed in [1, 3, 7, 14, 30]:
                # Find word
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
            f"📚 New words: {len(new_words_list)}",
            f"🔄 Review: {len(review_words_list)}",
            f"⏰ Recommended time: {user['daily_time']} minutes",
            f"🎯 Learning style: {learning_style}",
            "💡 Tip: Learn in morning, review in evening"
        ]
    }

@app.post("/review")
async def submit_word_review(review: WordReview):
    """Submit word review (remembered/not remembered)"""
    if review.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update progress
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
        "message": "Progress saved!",
        "next_review": "Tomorrow" if review.remembered else "In 1 day"
    }

@app.get("/test/{level}")
async def generate_test(level: int, questions: int = 10):
    """Generate test for HSK level"""
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"HSK {level} words not found")
    
    # Select random words
    selected_words = random.sample(level_words, min(questions, len(level_words)))
    
    test_questions = []
    for i, word in enumerate(selected_words, 1):
        # Create wrong options
        wrong_words = []
        other_words = [w for w in level_words if w["character"] != word["character"]]
        
        if len(other_words) >= 3:
            wrong_words = random.sample(other_words, 3)
        
        # Create answer options
        options = [word["translation"]] + [w["translation"] for w in wrong_words]
        random.shuffle(options)
        
        # Determine correct answer
        correct_index = options.index(word["translation"])
        
        test_questions.append({
            "id": f"q_{i}",
            "question": f"How to translate '{word['character']}' ({word['pinyin']})?",
            "options": options,
            "correct_index": correct_index,
            "correct_answer": word["translation"],
            "points": 1,
            "hint": f"HSK {level}, part of speech: {word.get('part_of_speech', 'not specified')}"
        })
    
    # Create test ID
    test_id = f"test_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save test
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
        "time_limit": f"{len(test_questions) * 1.5} minutes",
        "questions": test_questions,
        "test_hacks": [
            "⏱️ Spend no more than 1.5 minutes per question",
            "🎯 If in doubt - eliminate obviously wrong options",
            "📝 Remember: HSK often repeats similar options",
            "🧠 First thought is often correct"
        ]
    }

@app.post("/submit_test")
async def submit_test_answers(test_data: TestAnswer):
    """Submit test answers"""
    if test_data.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if test_data.test_id not in tests_db:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = tests_db[test_data.test_id]
    questions = test["questions"]
    
    # Check answers
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
            "explanation": f"Correct answer: {question['correct_answer']}"
        })
    
    score = correct
    max_score = len(questions)
    percentage = int((score / max_score) * 100) if max_score > 0 else 0
    
    # Save result
    if "results" not in tests_db[test_data.test_id]:
        tests_db[test_data.test_id]["results"] = {}
    
    tests_db[test_data.test_id]["results"][test_data.user_id] = {
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "submitted_at": datetime.now().isoformat(),
        "answers": test_data.answers
    }
    
    # Generate feedback
    feedback = ""
    if percentage >= 80:
        feedback = "🎉 Excellent! You're ready for the exam!"
    elif percentage >= 60:
        feedback = "👍 Good! Keep practicing!"
    else:
        feedback = "💪 Need more practice! Focus on weak areas."
    
    return {
        "test_id": test_data.test_id,
        "user_id": test_data.user_id,
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "feedback": feedback,
        "results": results,
        "recommendations": [
            f"🎯 Review words you made mistakes on",
            f"⏰ Next test in 3 days",
            f"📈 Goal for next time: {min(100, percentage + 10)}%"
        ]
    }

@app.get("/exam/{level}")
async def generate_exam(level: int):
    """Generate full HSK exam"""
    level_words = get_words_by_level(level, 1000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"HSK {level} words not found")
    
    # Different exam parts
    exam = {
        "listening": [],
        "reading": [],
        "writing": [],
        "speaking": []
    }
    
    # LISTENING (4 questions)
    for i in range(4):
        word = random.choice(level_words)
        wrong_words = random.sample([w for w in level_words if w != word], 3)
        
        exam["listening"].append({
            "type": "multiple_choice",
            "id": f"listening_{i+1}",
            "question": f"Listen to audio and choose correct translation for:",
            "character": word["character"],
            "pinyin": word["pinyin"],
            "options": [word["translation"]] + [w["translation"] for w in wrong_words],
            "correct_answer": word["translation"],
            "points": 5,
            "time_limit": "30 seconds"
        })
    
    # READING (3 questions)
    for i in range(3):
        # Matching
        pairs = random.sample(level_words, min(4, len(level_words)))
        exam["reading"].append({
            "type": "matching",
            "id": f"reading_{i+1}",
            "question": "Match Chinese words with translations:",
            "pairs": [{"character": w["character"], "pinyin": w["pinyin"]} for w in pairs],
            "answers": [w["translation"] for w in pairs],
            "shuffled_answers": random.sample([w["translation"] for w in pairs], len(pairs)),
            "points": 10,
            "time_limit": "2 minutes"
        })
    
    # WRITING (2 questions)
    writing_words = random.sample(level_words, min(2, len(level_words)))
    exam["writing"].append({
        "type": "writing",
        "id": "writing_1",
        "question": "Write characters for following words:",
        "words": [{"pinyin": w["pinyin"], "translation": w["translation"]} for w in writing_words],
        "answers": [w["character"] for w in writing_words],
        "points": 15,
        "time_limit": "5 minutes"
    })
    
    # SPEAKING (1 question)
    speaking_word = random.choice(level_words)
    exam["speaking"].append({
        "type": "speaking",
        "id": "speaking_1",
        "question": f"Pronounce word and make sentence with it:",
        "word": {
            "character": speaking_word["character"],
            "pinyin": speaking_word["pinyin"],
            "translation": speaking_word["translation"]
        },
        "example": f"Example: '{speaking_word['character']} ({speaking_word['pinyin']})' - {speaking_word['translation']}",
        "points": 20,
        "time_limit": "3 minutes"
    })
    
    exam_id = f"exam_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "exam_id": exam_id,
        "level": level,
        "total_points": 100,
        "time_total": "60 minutes",
        "sections": exam,
        "exam_strategy": [
            "🎯 Start with favorite part",
            "⏰ Distribute time: 20min reading, 15min listening, 15min writing, 10min speaking",
            "📝 In writing part, write draft first",
            "🎤 In speaking part, speak clearly and don't rush",
            "🔄 Leave 5 minutes for checking"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Database statistics"""
    if not words_db:
        return {"message": "Database is empty"}
    
    stats = {
        "total_words": len(words_db),
        "by_level": {},
        "by_part_of_speech": {},
        "users_count": len(users_db),
        "tests_taken": len(tests_db)
    }
    
    # Statistics by level
    for word in words_db:
        level = word.get("hsk_level", 0)
        stats["by_level"][f"HSK {level}"] = stats["by_level"].get(f"HSK {level}", 0) + 1
        
        # Statistics by part of speech
        pos = word.get("part_of_speech", "not specified")
        stats["by_part_of_speech"][pos] = stats["by_part_of_speech"].get(pos, 0) + 1
    
    # Most frequent characters
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
    """Search words by characters, pinyin or translation"""
    results = []
    query_lower = query.lower()
    
    for word in words_db:
        # Search in characters
        if query in word.get("character", ""):
            results.append(word)
            continue
            
        # Search in pinyin
        pinyin = word.get("pinyin", "").lower()
        if query_lower in pinyin:
            results.append(word)
            continue
            
        # Search in translation
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
    """Get random word"""
    if level:
        filtered_words = [w for w in words_db if w.get("hsk_level") == level]
    else:
        filtered_words = words_db
    
    if not filtered_words:
        raise HTTPException(status_code=404, detail="Words not found")
    
    word = random.choice(filtered_words)
    
    # Smart search for similar words:
    similar = []
    word_level = word.get("hsk_level", 1)
    word_chars = set(word["character"])
    
    for w in words_db:
        if w["character"] == word["character"]:
            continue
        
        # 1. Similar by character composition
        w_chars = set(w["character"])
        common_chars = word_chars.intersection(w_chars)
        
        # 2. Similar by topic (translation analysis)
        word_trans_lower = word["translation"].lower()
        w_trans_lower = w["translation"].lower()
        
        # Simple topic analysis
        categories = {
            "family": ["mother", "father", "brother", "sister", "family", "parents"],
            "food": ["eat", "drink", "food", "water", "tea", "rice"],
            "travel": ["go", "come", "train", "airplane", "hotel"],
            "study": ["study", "school", "student", "teacher", "book"],
            "time": ["time", "hour", "day", "month", "year", "today"]
        }
        
        similarity_found = False
        
        # Similar characters
        if common_chars:
            similarity_found = True
        
        # Same level
        if w.get("hsk_level", 1) == word_level:
            similarity_found = True
        
        # Similar translation (find common words in translation)
        word_trans_words = set(word_trans_lower.split())
        w_trans_words = set(w_trans_lower.split())
        common_words = word_trans_words.intersection(w_trans_words)
        
        if len(common_words) > 0:
            similarity_found = True
        
        # Same category
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
                "why_similar": f"Common characters: {len(common_chars)}, Topic: {category if 'category' in locals() else 'general'}"
            })
    
    # Take 3 most similar
    if len(similar) > 3:
        similar = similar[:3]
    elif len(similar) < 3:
        # Add random words of same level
        same_level_words = [w for w in filtered_words if w["character"] != word["character"]]
        while len(similar) < 3 and same_level_words:
            random_similar = random.choice(same_level_words)
            if random_similar not in similar:
                similar.append({
                    "character": random_similar["character"],
                    "pinyin": random_similar["pinyin"],
                    "translation": random_similar["translation"][:50],
                    "hsk_level": random_similar.get("hsk_level", 1),
                    "why_similar": "Random word of same level"
                })
    
    return {
        "word": word,
        "similar_words": similar,
        "memory_tip": generate_memory_tip(word),
        "study_suggestions": [
            "🔊 Pronounce aloud 10 times",
            f"🧠 Compare with similar: {', '.join([s['character'] for s in similar])}",
            "⏰ Review 3 more times today"
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
    """Generate Chinese text with given parameters"""
    try:
        # Get DeepSeek client
        client = get_deepseek_client()
        if not client:
            raise HTTPException(status_code=500, detail="AI service unavailable")
        
        # Form prompt based on format
        format_prompts = {
            "chinese_only": "ONLY in Chinese with characters",
            "full": "In Chinese with pinyin and Russian translation",
            "manga": "In manga style with dialogues and descriptions"
        }
        
        format_instruction = format_prompts.get(request.format, "In Chinese")
        
        # Form system prompt
        system_prompt = f"""You are a Chinese text author for language learners.
        
# TASK:
Create text on topic: "{request.topic}"
Description: {request.description}

# REQUIREMENTS:
1. Difficulty level: HSK {request.hsk_level}
2. Use words mainly from HSK {request.hsk_level} and below
3. {format_instruction}
4. Length: {request.length} (about {2000 if request.length == 'medium' else 1000 if request.length == 'short' else 3000} characters)
5. {"Use emojis 🎌" if request.include_emojis else "No emojis"}
6. {"Style like in manga: dialogues, descriptions, emotions" if request.manga_style else "Regular narrative style"}

# FORMATS:
- If only Chinese needed: characters + punctuation + emojis
- If pinyin needed: 汉字 (pinyin) 【translation】
- If manga style: 
  【Character】: Line
  *action description*
  
# STRUCTURE:
- Introduction/beginning
- Main part with development
- Conclusion/summary

Be creative, but use level-appropriate vocabulary!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create text on topic: {request.topic}"}
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
        
        # Analyze text for statistics
        stats = analyze_chinese_text(text_content, request.hsk_level)
        
        # Format text based on format
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
        raise HTTPException(status_code=500, detail=f"Text generation error: {str(e)}")

def analyze_chinese_text(text: str, target_hsk_level: int) -> Dict:
    """Analyze generated text"""
    # Simple analysis (in real project use HSK dictionary)
    characters = len([c for c in text if '\u4e00-\u9fff'])
    words = text.split()
    unique_words = len(set(words))
    
    # Simple difficulty estimation
    estimated_level = min(6, max(1, target_hsk_level + random.randint(-1, 1)))
    
    return {
        "characters": characters,
        "words": len(words),
        "unique_words": unique_words,
        "hsk_level": estimated_level,
        "estimated_reading_time": f"{max(1, characters // 300)} minutes",
        "new_words": max(0, unique_words - target_hsk_level * 100)  # Simple estimation
    }

def format_generated_text(text: str, format_type: str) -> str:
    """Format text for different formats"""
    if format_type == "manga":
        # Add manga markers
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
        # Here you could add pinyin and translation
        return text  # In real project integrate pinyin and translation
    
    return text

def add_pinyin_to_text(text: str) -> str:
    """Add pinyin to text (stub)"""
    # In real project use pinyin library
    # e.g., pypinyin
    return text

# Models for essay and translation checking
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
    description: Optional[str] = ""  # <-- added
    difficulty: str = "medium"
    length: str = "medium"
    hsk_level: int = 4
    user_id: Optional[str] = None
    include_emojis: bool = True  # <-- added
    manga_style: bool = False  # <-- added

@app.post("/essay/check")
async def check_essay(request: EssayCheckRequest):
    """AI essay checking"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_check(request)
        
        # Form prompt for checking
        system_prompt = f"""You are a strict but fair Chinese teacher.
        
# TASK:
Check essay on topic: "{request.topic}"
Student level: HSK {request.hsk_level}
Minimum length: {request.min_length} characters
Student essay length: {len(request.essay_text)} characters

# EVALUATION CRITERIA:
1. **Grammar** (30%) - correctness of constructions, particles, tenses
2. **Vocabulary** (25%) - richness of vocabulary, appropriateness of words  
3. **Structure** (20%) - logic, organization, coherence
4. **Content** (15%) - relevance to topic, arguments
5. **Style** (10%) - variety, naturalness, complexity

# RESPONSE FORMAT JSON:
{{
    "overall_score": 85,
    "categories": [
        {{"name": "Grammar", "score": 80, "feedback": "..."}},
        {{"name": "Vocabulary", "score": 85, "feedback": "..."}},
        {{"name": "Structure", "score": 90, "feedback": "..."}},
        {{"name": "Content", "score": 75, "feedback": "..."}},
        {{"name": "Style", "score": 80, "feedback": "..."}}
    ],
    "errors": [
        {{"position": 15, "error": "了 used incorrectly", "correction": "..."}},
        {{"position": 42, "error": "Incorrect word order", "correction": "..."}}
    ],
    "recommendations": "Improvement recommendations...",
    "strengths": "Work strengths...",
    "estimated_hsk_level": {request.hsk_level}
}}

# BE STRICT:
- Don't inflate scores
- Point out specific errors
- Give specific corrections
- Be constructive but honest

# STUDENT ESSAY:
{request.essay_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Check this essay and give detailed analysis."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse JSON response
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If AI didn't return JSON, create structured response
            result = {
                "overall_score": 75,
                "categories": [
                    {"name": "Grammar", "score": 70, "feedback": "Check particle usage"},
                    {"name": "Vocabulary", "score": 80, "feedback": "Good vocabulary"},
                    {"name": "Structure", "score": 85, "feedback": "Logical organization"},
                    {"name": "Content", "score": 75, "feedback": "Relevant to topic"},
                    {"name": "Style", "score": 70, "feedback": "Could vary style more"}
                ],
                "errors": [],
                "recommendations": "Continue practicing essay writing",
                "strengths": "Essay is relevant to topic and has logical structure"
            }
        
        # Add metadata
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
        print(f"Essay check error: {str(e)}")
        return generate_fallback_essay_check(request)

def generate_fallback_essay_check(request: EssayCheckRequest):
    """Fallback essay check (if AI unavailable)"""
    text = request.essay_text
    char_count = len(text)
    
    # Simple evaluation based on length
    if char_count < request.min_length:
        length_score = 50
    elif char_count < request.min_length * 1.5:
        length_score = 70
    else:
        length_score = 90
    
    base_score = length_score
    
    # Add random variations
    grammar_score = max(0, min(100, base_score + random.randint(-15, 15)))
    vocab_score = max(0, min(100, base_score + random.randint(-10, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-5, 15)))
    content_score = max(0, min(100, base_score + random.randint(-5, 10)))
    style_score = max(0, min(100, base_score + random.randint(-10, 5)))
    
    overall_score = int((grammar_score + vocab_score + structure_score + content_score + style_score) / 5)
    
    # Generate sample errors
    errors = []
    if char_count > 100:
        # Add couple of sample errors
        errors.append({
            "position": min(50, char_count - 10),
            "error": "Possible error in using 了",
            "correction": "Make sure 了 is used for completed actions"
        })
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Grammar", "score": grammar_score, 
             "feedback": "There are errors in particle usage. Pay attention to 了, 的, 地, 得."},
            {"name": "Vocabulary", "score": vocab_score,
             "feedback": f"Vocabulary diverse enough for HSK {request.hsk_level} level."},
            {"name": "Structure", "score": structure_score,
             "feedback": "Text organized logically, but could improve coherence between paragraphs."},
            {"name": "Content", "score": content_score,
             "feedback": f"Relevant to topic '{request.topic}', has arguments and examples."},
            {"name": "Style", "score": style_score,
             "feedback": "Style sufficiently varied, but could use more complex constructions."}
        ],
        "errors": errors,
        "recommendations": f"""
1. Practice using complex sentences with 虽然...但是..., 因为...所以...
2. Increase vocabulary on topic "{request.topic}"
3. Pay attention to particle usage 了, 的, 地, 得
4. Add transition words: 首先, 其次, 最后, 总而言之
5. Write regularly to improve skills
        """,
        "strengths": "Good text organization, relevance to topic, sufficient length.",
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
    """Fallback essay check"""
    # Simple essay analysis
    text = request.essay_text
    char_count = len(text)
    
    # Simple evaluation
    base_score = min(100, max(50, char_count / request.min_length * 80))
    
    # Random variations
    grammar_score = max(0, min(100, base_score + random.randint(-15, 15)))
    vocab_score = max(0, min(100, base_score + random.randint(-10, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-5, 15)))
    content_score = max(0, min(100, base_score + random.randint(-5, 10)))
    style_score = max(0, min(100, base_score + random.randint(-10, 5)))
    
    overall_score = int((grammar_score + vocab_score + structure_score + content_score + style_score) / 5)
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Grammar", "score": grammar_score, 
             "feedback": "There are errors in particle usage. Pay attention to 了, 的, 地, 得."},
            {"name": "Vocabulary", "score": vocab_score,
             "feedback": "Vocabulary diverse enough for HSK " + str(request.hsk_level) + " level"},
            {"name": "Structure", "score": structure_score,
             "feedback": "Text organized logically, but could improve coherence between paragraphs."},
            {"name": "Content", "score": content_score,
             "feedback": "Relevant to topic, has arguments and examples."},
            {"name": "Style", "score": style_score,
             "feedback": "Style sufficiently varied, but could use more complex constructions."}
        ],
        "errors": [
            {"position": random.randint(10, len(text)//2), 
             "error": "Possible word order error",
             "correction": "Check word order in sentence"},
            {"position": random.randint(len(text)//2, len(text)-10),
             "error": "Word repetition",
             "correction": "Use synonyms for variety"}
        ] if char_count > 50 else [],
        "recommendations": """
        1. Practice using complex sentences with 虽然...但是..., 因为...所以...
        2. Increase vocabulary on topic "{}"
        3. Pay attention to particle usage 了, 的, 地, 得
        4. Add transition words: 首先, 其次, 最后, 总而言之
        5. Write regularly to improve skills
        """.format(request.topic),
        "strengths": "Good text organization, relevance to topic, sufficient length.",
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
    """Generate text for translation"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_translation_text(request)
        
        # Determine length
        lengths = {
            "short": "3-5 sentences",
            "medium": "6-10 sentences", 
            "long": "10-15 sentences"
        }
        
        system_prompt = f"""You create Russian texts for translation to Chinese.
        
# TASK:
Create text on topic: "{request.topic}"
Difficulty: {request.difficulty}
Length: {lengths.get(request.length, "6-10 sentences")}
Student level: HSK {request.hsk_level}

# REQUIREMENTS:
1. Text should be interesting and useful for learning
2. Difficulty level should match student level
3. Use diverse vocabulary and grammar
4. Text should be natural, like in real life
5. Include elements that need correct translation

# TEXT FORMATS:
- News: formal style, facts
- Story: narrative, dialogues
- Dialogue: conversational speech, questions and answers
- Description: details, adjectives
- Instruction: imperatives, sequence

# EXAMPLE FOR MEDIUM DIFFICULTY:
"Yesterday a new cultural center opened in Shanghai. It combines library, museum and concert hall. Visitors can visit exhibitions free in first month."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create text for translation on topic: {request.topic}"}
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
        print(f"Text generation error: {str(e)}")
        return generate_fallback_translation_text(request)

def generate_fallback_translation_text(request: TranslationGenerateRequest):
    """Fallback text generation for translation"""
    topics_texts = {
        "news": "China launched new Earth observation satellite. It will be used for weather and ecology monitoring. Satellite launched by Changzheng rocket.",
        "story": "Long ago in small village lived old calligraphy master. Every morning he woke at dawn and practiced characters. His works were known throughout region.",
        "dialogue": "- Hello! My name is Anna. I'm from Russia. - Nice to meet you! I'm Li Wei. First time in China? - Yes, I'm here studying Chinese. - Great! Good luck with studies!",
        "description": "Great Wall of China is ancient defensive structure. It passes through mountains and valleys of northern China. Wall length is over 20 thousand kilometers.",
        "instruction": "To cook Chinese fried rice, first boil rice and cool it. Then fry eggs, add vegetables and chopped meat. Finally add rice and soy sauce."
    }
    
    # Choose text by topic or use general
    text = topics_texts.get(request.topic, 
        "Chinese culture is very rich and diverse. It includes traditional medicine, cuisine, art and philosophy. Studying Chinese culture helps better understand language.")
    
    # Adapt difficulty
    if request.difficulty == "easy":
        # Simplify text
        sentences = text.split('. ')
        text = '. '.join(sentences[:2]) + '.'
    elif request.difficulty == "hard":
        # Make text more complex
        text += " These aspects are closely related to country's historical development and Confucian influence."
    
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
    """AI translation checking"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_translation_check(request)
        
        system_prompt = f"""You are expert in Russian to Chinese translation.
        
# TASK:
Compare student's translation with ideal translation.
Original (Russian): "{request.original_text}"
Student translation: "{request.user_translation}"
Student level: HSK {request.target_hsk}
Difficulty: {request.difficulty}

# EVALUATION CRITERIA:
1. **Accuracy** (40%) - correctness of meaning translation
2. **Grammar** (30%) - correctness of Chinese constructions
3. **Naturalness** (20%) - sounds like native language
4. **Style** (10%) - preserving original style

# YOUR WORK:
1. Create ideal translation of original
2. Compare with student translation
3. Find and classify errors
4. Give improvement recommendations
5. Give score

# RESPONSE FORMAT JSON:
{{
    "overall_score": 85,
    "ideal_translation": "Ideal translation to Chinese...",
    "categories": [
        {{"name": "Accuracy", "score": 90, "feedback": "..."}},
        {{"name": "Grammar", "score": 80, "feedback": "..."}},
        {{"name": "Naturalness", "score": 85, "feedback": "..."}},
        {{"name": "Style", "score": 80, "feedback": "..."}}
    ],
    "errors": [
        {{"type": "grammar", "description": "Incorrect word order", "suggestion": "..."}},
        {{"type": "vocabulary", "description": "Inaccurate word translation", "suggestion": "..."}}
    ],
    "correct_translations": [
        {{"original": "Russian phrase", "student": "student translation", "ideal": "ideal translation"}}
    ],
    "recommendations": "Specific recommendations...",
    "estimated_hsk_level": {request.target_hsk}
}}

# BE CONSTRUCTIVE:
- Praise good points
- Explain errors in detail
- Suggest alternatives
- Help learn from mistakes"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Check this translation and give detailed analysis."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
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
        print(f"Translation check error: {str(e)}")
        return generate_fallback_translation_check(request)

def generate_fallback_translation_check(request: TranslationCheckRequest):
    """Fallback translation check"""
    # Generate "ideal" translation (simple)
    ideal_translation = generate_simple_translation(request.original_text, request.target_hsk)
    
    # Simple evaluation
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
            {"name": "Accuracy", "score": accuracy_score,
             "feedback": "Main meaning conveyed correctly, but there are inaccuracies in details."},
            {"name": "Grammar", "score": grammar_score,
             "feedback": "There are errors in word order and particle usage."},
            {"name": "Naturalness", "score": naturalness_score,
             "feedback": "Translation understandable, but sounds slightly unnatural for native."},
            {"name": "Style", "score": style_score,
             "feedback": "Style mostly preserved, but could be improved."}
        ],
        "errors": [
            {"type": "grammar", 
             "description": "Possible word order errors",
             "suggestion": "In Chinese word order is SVO (subject-verb-object)"},
            {"type": "vocabulary",
             "description": "Could use more accurate words",
             "suggestion": "Use synonyms for variety and accuracy"}
        ],
        "correct_translations": [
            {"original": request.original_text.split('. ')[0] if '. ' in request.original_text else request.original_text,
             "student": request.user_translation.split('。')[0] if '。' in request.user_translation else request.user_translation,
             "ideal": ideal_translation.split('。')[0] if '。' in ideal_translation else ideal_translation}
        ],
        "recommendations": """
        1. Pay attention to word order in sentences
        2. Use dictionaries to find more accurate equivalents
        3. Practice translating different text types
        4. Read original Chinese texts to understand natural style
        5. Check particle usage 了, 的, 地, 得
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
    """Simple text translation (stub)"""
    # In real project there would be real translation
    # Now return template text
    translations = {
        3: "这是一个简单的翻译示例。中文很重要。",
        4: "昨天在公园里有很多人。天气很好，阳光明媚。",
        5: "随着中国经济的发展，越来越多的外国人来到中国工作和学习。",
        6: "中国传统文化博大精深，源远流长。它不仅包括丰富的哲学思想，还涵盖了独特的艺术形式和生活智慧。"
    }
    
    return translations.get(hsk_level, "这是一个翻译文本。")

@app.get("/text/history/{user_id}")
async def get_text_generation_history(user_id: str, limit: int = 20):
    """Get text generation history"""
    try:
        # Load from file
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
        raise HTTPException(status_code=500, detail=f"History load error: {str(e)}")

@app.get("/words/level/{level}")
async def get_level_words(level: int, limit: int = 10000, offset: int = 0):
    """Get words of specific HSK level"""
    level_words = get_words_by_level(level, 20000)
    
    if not level_words:
        raise HTTPException(status_code=404, detail=f"HSK {level} words not found")
    
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
    """Summary of all HSK levels"""
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
    """User progress"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    progress = word_progress_db.get(user_id, {})
    
    # Statistics by level
    level_stats = {}
    for level in range(1, 7):
        level_words = get_words_by_level(level, 1000)
        total_level_words = len(level_words)
        
        # Count learned words of this level
        learned = 0
        for word_id, word_progress in progress.items():
            if word_progress.get("remembered", False):
                # Check word is of this level
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


# Add to backend models:
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

# Add to backend routes:
@app.post("/essay/analysis/generate")
async def generate_essay_analysis(request: EssayAnalysisRequest):
    """Generate essay assignment"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_analysis(request)
        
        # Determine time based on difficulty
        time_limits = {
            "beginner": 45,
            "intermediate": 60,
            "advanced": 75,
            "exam": 90
        }
        
        system_prompt = f"""You create essay assignments in Chinese.
        
# TASK:
Create essay assignment on topic: "{request.topic}"
Difficulty level: {request.difficulty}
Target length: {request.target_length} characters
Additional details: {request.details}

# ASSIGNMENT REQUIREMENTS:
1. Clearly formulated topic and task
2. Specific content requirements
3. Evaluation criteria by 4 categories:
   - Content (40%)
   - Grammar (30%) 
   - Vocabulary (20%)
   - Structure (10%)
4. Time limit: {time_limits.get(request.difficulty, 60)} minutes

# RESPONSE FORMAT JSON:
{{
    "prompt": "Full assignment for student with instructions...",
    "requirements": "Specific essay requirements...",
    "evaluation_criteria": [
        "Content: relevance to topic, arguments, examples (40%)",
        "Grammar: correctness of constructions, particles, tenses (30%)",
        "Vocabulary: vocabulary diversity, word appropriateness (20%)",
        "Structure: logic, organization, coherence (10%)"
    ],
    "time_limit_minutes": {time_limits.get(request.difficulty, 60)},
    "suggested_structure": ["Introduction", "2-3 arguments", "Conclusion"]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create essay assignment on topic: {request.topic}"}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
        result.update({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "target_length": request.target_length,
            "generated_at": datetime.now().isoformat(),
            "ai_generated": True
        })
        
        return result
        
    except Exception as e:
        print(f"Assignment generation error: {str(e)}")
        return generate_fallback_essay_analysis(request)

def generate_fallback_essay_analysis(request: EssayAnalysisRequest):
    """Fallback essay assignment generation"""
    difficulty_texts = {
        "beginner": "Use simple sentences and basic HSK 1-3 vocabulary.",
        "intermediate": "Use complex sentences and diverse HSK 4-5 vocabulary.",
        "advanced": "Demonstrate mastery of complex grammatical constructions.",
        "exam": "Demonstrate all aspects of language proficiency at high level."
    }
    
    time_limits = {
        "beginner": 45,
        "intermediate": 60,
        "advanced": 75,
        "exam": 90
    }
    
    return {
        "prompt": f"""
<h4>Topic: {request.topic}</h4>
<p><strong>Assignment:</strong> Write essay on given topic. Your essay should include:</p>
<ul>
    <li>Introduction presenting topic and your position</li>
    <li>2-3 main arguments with specific examples</li>
    <li>Conclusion with conclusions and summary</li>
</ul>
<p><strong>Requirements:</strong></p>
<ul>
    <li>Length: {request.target_length} characters</li>
    <li>{difficulty_texts.get(request.difficulty, 'Use complex sentences')}</li>
    <li>Use transition words and connecting elements</li>
    <li>Avoid repetitions and grammatical errors</li>
</ul>
<p><strong>Time limit:</strong> {time_limits.get(request.difficulty, 60)} minutes</p>
        """,
        "requirements": f"Length: {request.target_length} characters. {difficulty_texts.get(request.difficulty, 'Use complex sentences')}",
        "evaluation_criteria": [
            "Content: relevance to topic, arguments, examples (40%)",
            "Grammar: correctness of constructions, particles, tenses (30%)",
            "Vocabulary: vocabulary diversity, word appropriateness (20%)",
            "Structure: logic, organization, coherence (10%)"
        ],
        "time_limit_minutes": time_limits.get(request.difficulty, 60),
        "suggested_structure": ["Introduction", "2-3 arguments", "Conclusion"],
        "topic": request.topic,
        "difficulty": request.difficulty,
        "target_length": request.target_length,
        "generated_at": datetime.now().isoformat(),
        "ai_generated": False,
        "fallback": True
    }

@app.post("/essay/analysis/check")
async def check_essay_analysis(request: EssaySubmitRequest):
    """Strict essay checking for analysis"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_essay_check_analysis(request)
        
        system_prompt = f"""You are a STRICT and DEMANDING Chinese teacher.
        
# TASK:
Check essay on topic: "{request.topic}"
Difficulty level: {request.difficulty}
Target length: {request.target_length} characters
Student essay length: {len(request.essay_text)} characters

# BE MAXIMALLY STRICT:
- Don't inflate scores by even one point!
- Deduct points for each error
- Demand perfection
- Don't make allowances

# EVALUATION CRITERIA:
1. **Content** (40%) - accuracy, arguments, examples, depth
2. **Grammar** (30%) - perfect grammar, no errors
3. **Vocabulary** (20%) - rich vocabulary, accuracy, diversity
4. **Structure** (10%) - perfect organization, logic, coherence

# RESPONSE FORMAT JSON:
{{
    "overall_score": 65,  // BE STRICT!
    "categories": [
        {{"name": "Content", "score": 70, "feedback": "STRICT feedback pointing out ALL shortcomings"}},
        {{"name": "Grammar", "score": 60, "feedback": "STRICT feedback with LIST OF ALL errors"}},
        {{"name": "Vocabulary", "score": 75, "feedback": "STRICT feedback about vocabulary"}},
        {{"name": "Structure", "score": 80, "feedback": "STRICT feedback about structure"}}
    ],
    "errors": [
        {{"type": "grammar", "position": 15, "description": "SPECIFIC error", "correction": "EXACT correction", "severity": "high"}},
        {{"type": "vocabulary", "position": 42, "description": "INACCURATE word", "correction": "CORRECT option", "severity": "medium"}}
    ],
    "strengths": "Only real strengths, don't invent!",
    "weaknesses": "DETAILED list of weak points",
    "recommendations": "SPECIFIC and HARSH improvement recommendations",
    "estimated_level": "Real student level (DON'T inflate!)",
    "would_pass_exam": false  // Honestly evaluate, would pass exam?
}}

# STUDENT ESSAY:
{request.essay_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Check this essay MAXIMALLY STRICTLY and give honest evaluation."}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,  # Low temperature for strictness
            max_tokens=2500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = generate_fallback_essay_check_analysis(request)
        
        # Add metadata
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
        print(f"Strict check error: {str(e)}")
        return generate_fallback_essay_check_analysis(request)
    
@app.post("/ai/search-universities")
async def search_universities(request: dict):
    """
    Main function: AI searches universities on the internet
    """
    try:
        query = request.get("query", "")
        filters = request.get("filters", {})
        
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")
        
        # 1. Form SMART prompt for AI
        system_prompt = f"""
        You are expert in Chinese education. User is searching: "{query}"
        
        YOUR TASK: FIND CURRENT INFORMATION ON THE INTERNET
        
        INSTRUCTIONS:
        1. USE INTERNET SEARCH to find fresh data
        2. Search in Russian, English, Chinese languages
        3. Main sources: official university sites (.edu.cn), csc.edu.cn, studyinchina.edu.cn
        4. Consider filters: HSK {filters.get('hsk_level', 'any')}, budget {filters.get('max_budget', 'any')}
        5. Compare minimum 3-5 options
        6. Give specific data: prices, deadlines, contacts
        
        RESPONSE FORMAT:
        - University name (city)
        - Requirements: HSK, exams, documents
        - Tuition cost (in yuan)
        - Scholarships: available, how to get
        - Application deadlines
        - Links to official pages
        - Pros and cons of each option
        - Admission tips
        
        IMPORTANT: All data must be CURRENT (2024-2025 year).
        """
        
        # 2. Call DeepSeek with ENABLED internet search
        client = get_deepseek_client()
        if not client:
            return {"error": "API key not configured"}
        
        # CRITICALLY IMPORTANT: Enable web search!
        # Check exact parameter name in DeepSeek documentation
        response = client.chat.completions.create(
            model="deepseek-chat",  # Or other model with search
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Find information for query: {query}"}
            ],
            # PARAMETER FOR WEB SEARCH (example names):
            # web_search=True, 
            # use_web=True,
            # search_online=True,
            max_tokens=4000  # Many tokens for detailed response
        )
        
        ai_response = response.choices[0].message.content
        
        # 3. Return result
        return {
            "success": True,
            "query": query,
            "analysis": ai_response,  # Text from AI
            "count": len(ai_response.split('\n')) // 10,  # Approximate number of options
            "search_performed": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in AI search: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback": "Will show local data...",
            # Can add fallback data from your database
        }

def generate_fallback_essay_check_analysis(request: EssaySubmitRequest):
    """Fallback strict check"""
    text = request.essay_text
    char_count = len(text)
    
    # STRICT evaluation based on length
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
    
    # STRICT category evaluations
    content_score = max(0, min(100, base_score + random.randint(-20, 10)))
    grammar_score = max(0, min(100, base_score + random.randint(-25, 5)))
    vocab_score = max(0, min(100, base_score + random.randint(-15, 10)))
    structure_score = max(0, min(100, base_score + random.randint(-10, 15)))
    
    overall_score = int((content_score + grammar_score + vocab_score + structure_score) / 4)
    
    # HARSH errors
    errors = []
    if char_count > 50:
        errors.append({
            "type": "grammar",
            "position": min(30, char_count - 20),
            "description": "SERIOUS error in using 了",
            "correction": "NEVER use 了 in this context",
            "severity": "high"
        })
        
    if char_count > 100:
        errors.append({
            "type": "vocabulary", 
            "position": min(70, char_count - 30),
            "description": "THIS word is INCORRECT in this context",
            "correction": "Use ONLY correct word: ...",
            "severity": "medium"
        })
    
    would_pass = overall_score >= 70  # STRICT passing score
    
    return {
        "overall_score": overall_score,
        "categories": [
            {"name": "Content", "score": content_score,
             "feedback": "INSUFFICIENTLY deep analysis. Need SPECIFIC examples and details."},
            {"name": "Grammar", "score": grammar_score,
             "feedback": "MANY grammar errors. Unacceptable for this level."},
            {"name": "Vocabulary", "score": vocab_score,
             "feedback": "Vocabulary VERY limited. Learn more words."},
            {"name": "Structure", "score": structure_score,
             "feedback": "Structure chaotic. Follow plan: introduction-arguments-conclusion."}
        ],
        "errors": errors,
        "strengths": "Only one plus: relevance to topic (but weak).",
        "weaknesses": "EVERYTHING else: grammar, vocabulary, structure, argumentation.",
        "recommendations": """
1. RELEARN grammar. Errors are UNACCEPTABLE.
2. INCREASE vocabulary 2x. CURRENTLY insufficient.
3. ALWAYS write with plan. Chaos is failure.
4. PRACTICE every day. Once a week is TOO LITTLE.
5. HIRE tutor if can't manage alone.
        """,
        "estimated_level": f"Real level: HSK {max(1, min(6, overall_score // 15))}",
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
    estimated_duration: int  # in seconds
    generated_at: str

# REPLACE the generate_audio_lesson function in backend with this:
@app.post("/audio/generate-lesson")
async def generate_audio_lesson(request: AudioLessonRequest):
    """Generate full audio lesson (podcast) in Chinese"""
    try:
        client = get_deepseek_client()
        if not client:
            return generate_fallback_audio_lesson(request)
        
        # Determine text length based on user choice
        length_targets = {
            "short": 300,    # 1-2 minutes
            "medium": 600,   # 3-5 minutes
            "long": 1000     # 5-10 minutes
        }
        
        target_chars = length_targets.get(request.target_length, 600)
        
        system_prompt = f"""You are professional creator of Chinese podcasts for language learners.

# TASK:
Create full podcast on topic: "{request.topic}"
Topic details: {request.description or 'Not specified'}
HSK level: {request.hsk_level}
Difficulty: {request.difficulty}
Duration: {request.target_length}
Approximate volume: {target_chars} characters

# PODCAST REQUIREMENTS:
1. Should be COMPLETE audio lesson with:
   - Greeting and topic introduction
   - Main part with topic development
   - Specific examples and details
   - Useful expressions and vocabulary
   - Questions for listeners
   - Summary and conclusion

2. LENGTH: At least {target_chars} characters
3. STRUCTURE:
   - Introduction (20%)
   - Main part (60%)
   - Conclusion (20%)
4. STYLE: Natural, conversational, but clear
5. INCLUDE: 
   - Dialogues or example dialogues
   - Cultural notes
   - Useful tips
   - Specific language usage examples

# AVOID:
- Template phrases
- Too academic language
- Repetitions
- Too short sentences

# RESPONSE FORMAT JSON:
{{
    "title": "Podcast title",
    "chinese_text": "Full podcast text here...",
    "pinyin_text": "Text with pinyin (if include_pinyin=true)",
    "translation": "Full Russian translation (if include_translation=true)",
    "vocabulary": [
        {{
            "chinese": "词语",
            "pinyin": "cíyǔ", 
            "translation": "translation",
            "example": "Example sentence",
            "category": "part of speech"
        }}
    ],
    "comprehension_questions": [
        {{
            "question": "Comprehension question",
            "options": ["A", "B", "C", "D"],
            "correct_answer": 0,
            "explanation": "Answer explanation"
        }}
    ],
    "estimated_duration": 180,
    "word_count": 500,
    "character_count": 800,
    "difficulty_analysis": {{
        "grammar_complexity": "medium",
        "vocabulary_level": "HSK {request.hsk_level}",
        "speed_recommendation": "1.0x"
    }}
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Create full podcast in Chinese.

Topic: {request.topic}
Description: {request.description or 'Not specified'}
Level: HSK {request.hsk_level}
Difficulty: {request.difficulty}
Duration: {request.target_length}

Please make text NATURAL and CONVERSATIONAL, like real podcast."""}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.9,  # More creative approach
            max_tokens=4000,   # Increase for long texts
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Generate lesson ID
        lesson_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.topic[:20])}"
        
        # Add metadata
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
        
        # If user didn't request pinyin, remove it
        if not request.include_pinyin:
            result["pinyin_text"] = None
        
        # If user didn't request translation, remove it
        if not request.include_translation:
            result["translation"] = None
        
        return result
        
    except Exception as e:
        print(f"Audio lesson generation error: {str(e)}")
        # Always use fallback with longer text
        return generate_improved_fallback_audio_lesson(request)

def generate_improved_fallback_audio_lesson(request: AudioLessonRequest):
    """Improved fallback for podcast generation"""
    
    # Create longer and more diverse texts
    topic = request.topic
    difficulty = request.difficulty
    
    # Base text based on topic and difficulty
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
    
    # Add variations based on HSK level
    if request.hsk_level <= 2:
        # Simplify for beginners
        base_text = f"""你好！我是你的中文老师。

今天我们学习：{topic}。

{topic}很有意思。我们来看看。

这是什么？这是{topic}。你喜欢{topic}吗？

我喜欢{topic}。你呢？

我们一起学习。慢慢说，不要急。

好，今天学到这里。再见！"""
    
    elif request.hsk_level >= 5:
        # Make more complex for advanced
        base_text = f"""各位听众朋友，大家好。

欢迎收听本期深度汉语学习播客。今天我们将围绕"{topic}"这一主题展开探讨。

在当前全球化语境下，{topic}作为一个跨文化议题，引起了广泛关注。从本质上看，{topic}不仅涉及语言层面的表达，更蕴含着深刻的文化内涵。

首先，让我们从历史维度审视{topic}的演变过程。自古以来，{topic}在中国传统文化体系中占据重要位置。相关文献记载表明，早在先秦时期，{topic}的概念就已初步形成，并随着时代变迁不断丰富发展。

其次，现代社会的{topic}呈现出新的特点。在数字化转型的背景下，{topic}的表现形式和实践方式都发生了显著变化。这种变化既带来机遇，也带来挑战。

从语言学习的角度而言，掌握与{topic}相关的专业术语和表达方式至关重要。这不仅有助于提升语言能力，更能促进跨文化理解。

值得注意的是，不同文化背景的学习者对{topic}的认知可能存在差异。因此，在讨论{topic}时，我们需要保持开放的态度，尊重多元视角。

总而言之，{topic}是一个值得深入研究的复杂课题。通过系统学习，我们不仅能够提升汉语水平，更能深化对中国文化的理解。

感谢收听，我们下期再见。"""
    
    # Generate vocabulary
    vocabulary = [
        {
            "chinese": "话题",
            "pinyin": "huàtí", 
            "translation": "topic, subject of conversation",
            "example": "今天的话题很有意思。",
            "category": "noun"
        },
        {
            "chinese": "学习",
            "pinyin": "xuéxí",
            "translation": "study, learn",
            "example": "我喜欢学习中文。",
            "category": "verb"
        },
        {
            "chinese": "文化",
            "pinyin": "wénhuà",
            "translation": "culture",
            "example": "中国文化很有特色。",
            "category": "noun"
        },
        {
            "chinese": "重要",
            "pinyin": "zhòngyào",
            "translation": "important",
            "example": "这个问题很重要。",
            "category": "adjective"
        }
    ]
    
    # Add more words for advanced levels
    if request.hsk_level >= 4:
        vocabulary.extend([
            {
                "chinese": "探讨",
                "pinyin": "tàntǎo",
                "translation": "discuss, research",
                "example": "我们来探讨一下这个问题。",
                "category": "verb"
            },
            {
                "chinese": "理解",
                "pinyin": "lǐjiě",
                "translation": "understand",
                "example": "我理解你的意思。",
                "category": "verb"
            }
        ])
    
    # Comprehension questions
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
    
    # Calculate approximate duration (about 150 characters per minute)
    estimated_duration = max(120, len(base_text) // 2)
    
    return {
        "id": lesson_id,
        "title": f"汉语学习播客：{topic}",
        "chinese_text": base_text,
        "pinyin_text": None if not request.include_pinyin else "pinyin text would be here",
        "translation": None if not request.include_translation else "translation would be here",
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
            "grammar_complexity": "medium" if request.hsk_level <= 3 else "high",
            "vocabulary_level": f"HSK {request.hsk_level}",
            "speed_recommendation": "0.8x" if request.hsk_level <= 2 else "1.0x"
        },
        "note": "This is automatically generated podcast. For better quality content check AI connection."
    }

def generate_fallback_audio_lesson(request: AudioLessonRequest):
    """Fallback audio lesson generation"""
    
    # Base text based on HSK level
    base_texts = {
        1: "你好！我是你的中文老师。今天我们来学习中文。中文很有意思。",
        2: "大家好！欢迎来到中文课。今天天气很好。我想去公园散步。你呢？",
        3: "同学们好！今天我们要学习关于中国文化的主题。中国有很长的历史。中国的食物很好吃。",
        4: "欢迎收听我们的中文播客！今天我们来聊聊中国的传统节日。春节是最重要的节日。",
        5: "在这个数字时代，学习语言变得更加容易。通过互联网，我们可以接触到丰富的学习资源。",
        6: "中华文明源远流长，博大精深。从古代的四大发明到现代的科技创新，中国一直在为世界做出贡献。"
    }
    
    base_text = base_texts.get(request.hsk_level, base_texts[3])
    
    # Add topic to text
    chinese_text = f"今天的话题是：{request.topic}。{base_text} 希望你喜欢这个内容。再见！"
    
    # Basic vocabulary
    vocabulary = [
        {
            "chinese": "话题",
            "pinyin": "huàtí", 
            "translation": "topic",
            "example": "今天的话题很有意思。"
        },
        {
            "chinese": "学习",
            "pinyin": "xuéxí",
            "translation": "study",
            "example": "我喜欢学习中文。"
        }
    ]
    
    lesson_id = f"audio_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "id": lesson_id,
        "title": f"Audio lesson: {request.topic}",
        "chinese_text": chinese_text,
        "pinyin_text": None,
        "translation": f"Today's topic: {request.topic}. {base_text} Hope you enjoy this content. Goodbye!",
        "vocabulary": vocabulary,
        "difficulty": request.difficulty,
        "hsk_level": request.hsk_level,
        "estimated_duration": len(chinese_text) * 0.5,  # ~0.5 sec per character
        "generated_at": datetime.now().isoformat(),
        "topic": request.topic,
        "ai_generated": False,
        "fallback": True,
        "speech_rate": 1.0,
        "word_count": len(chinese_text.split()),
        "character_count": len(chinese_text.replace(" ", "")),
        "study_questions": [
            "What is the main topic of this lesson?",
            "What new words did you hear?"
        ]
    }

class WordStatus(BaseModel):
    user_id: str
    word_id: str          # format: "你好_1"
    status: str           # "saved" or "learned"

class WordTestRequest(BaseModel):
    user_id: str
    source: str = "all"          # "all", "saved", "learned"
    count: int = 20
    test_type: str               # "pinyin_from_char", "char_from_pinyin", "translation_from_char", "translation_from_pinyin"

class WordTestSubmit(BaseModel):
    user_id: str
    test_id: str
    answers: Dict[str, str]      # question_id -> user answer

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
    # Get word pool
    if req.source == "all":
        all_words = []
        for level in range(1, 7):
            all_words.extend(words_db.get(level, []))
    else:
        if req.user_id not in user_word_status:
            raise HTTPException(404, "No saved/learned words")
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
        raise HTTPException(400, "No words for test")

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
            q["prompt"] = f"Pinyin for: {word['character']}"
            q["correct"] = word["pinyin"]
        elif req.test_type == "char_from_pinyin":
            q["prompt"] = f"Characters for: {word['pinyin']}"
            q["correct"] = word["character"]
        elif req.test_type == "translation_from_char":
            q["prompt"] = f"Translation for: {word['character']}"
            q["correct"] = word["translation"]
        elif req.test_type == "translation_from_pinyin":
            q["prompt"] = f"Translation for: {word['pinyin']}"
            q["correct"] = word["translation"]
        else:
            raise HTTPException(400, "Invalid test_type")
        questions.append(q)

    test_id = f"word_{req.user_id}_{datetime.now().timestamp()}"
    
    # Save active test for checking later
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
        raise HTTPException(status_code=404, detail="Test not found or already completed")

    test = tests_db[test_key]
    questions = test["questions"]

    correct = 0
    total = len(questions)
    results = []

    for q in questions:
        qid = q["id"]
        correct_ans = q["correct"].strip().lower()

        user_answer_raw = submit.answers.get(qid)

        # EXPLICITLY: if no answer or empty string — INCORRECT
        if user_answer_raw is None or user_answer_raw.strip() == "":
            is_correct = False
            user_display = "(not answered)"
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

    message = f"{correct} out of {total} correct ({percentage}%)"
    if percentage >= 90:
        message += " Excellent! You know these words well!"
    elif percentage >= 70:
        message += " Not bad, but can be better."
    else:
        message += " Need more practice!"

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
    """AI checks word knowledge test"""
    try:
        client = get_deepseek_client()
        if not client:
            raise HTTPException(status_code=503, detail="AI service unavailable")

        # Form prompt for AI
        system_prompt = """You are a strict and accurate Chinese teacher.
Your task: check student's answers to Chinese word test.

CHECKING RULES:
1. Consider synonyms and similar meaning answers
2. For pinyin: ignore tones and spaces (nǐhǎo = nihao = nǐ hǎo)
3. For translation: allow translation variants if meaning preserved
4. Empty answer — always INCORRECT
5. Be objective but fair

RESPONSE FORMAT — ONLY JSON:
{
    "correct_count": 12,
    "total": 15,
    "percentage": 80,
    "results": [
        {
            "id": "0",
            "prompt": "Pinyin for: 你好",
            "user_answer": "nihao",
            "correct_answer": "nǐ hǎo",
            "is_correct": true,
            "feedback": "Correct! Tones can be omitted in test."
        },
        {
            "id": "1",
            "prompt": "Translation for: 谢谢",
            "user_answer": "пожалуйста",
            "correct_answer": "спасибо",
            "is_correct": false,
            "feedback": "Incorrect. 谢谢 = спасибо. 'Пожалуйста' = 请 or 不客气."
        }
    ],
    "summary": "Good result! Main errors — in translation of polite expressions."
}"""

        # Form list of questions with answers
        questions_text = ""
        for q in request.questions:
            user_ans = request.answers.get(q["id"], "(not answered)")
            questions_text += f"""
Question {q['id']}: {q['prompt']}
Correct answer: {q['correct']}
Student answer: {user_ans}
"""

        user_prompt = f"""Check student's answers.

Questions and answers:
{questions_text}

Evaluate each answer and give overall result."""

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
            # Fallback if AI didn't return JSON
            result = fallback_word_test_check(request.questions, request.answers)

        return result

    except Exception as e:
        print(f"Error in AI word test checking: {e}")
        # Always return fallback
        return fallback_word_test_check(request.questions, request.answers)

def fallback_word_test_check(questions, answers):
    """Fallback check if AI unavailable"""
    correct = 0
    total = len(questions)
    results = []

    for q in questions:
        qid = q["id"]
        user_raw = answers.get(qid, "")
        user_answer = user_raw.strip().lower() if user_raw else ""

        correct_ans = q["correct"].strip().lower()

        # Pinyin normalization
        if "пиньинь" in q["prompt"].lower() or "pinyin" in q["prompt"].lower():
            user_answer = user_answer.replace(" ", "").replace("v", "ü")
            correct_ans = correct_ans.replace(" ", "").replace("v", "ü")

        is_correct = bool(user_answer and user_answer == correct_ans)

        if is_correct:
            correct += 1

        results.append({
            "id": qid,
            "prompt": q["prompt"],
            "user_answer": user_raw.strip() if user_raw else "(not answered)",
            "correct_answer": q["correct"],
            "is_correct": is_correct,
            "feedback": "Correct!" if is_correct else "Incorrect. Check answer." if user_answer else "Not answered — considered incorrect."
        })

    percentage = round(correct / total * 100, 1) if total > 0 else 0

    return {
        "correct_count": correct,
        "total": total,
        "percentage": percentage,
        "results": results,
        "summary": f"{correct}/{total} correct ({percentage}%). {'Excellent!' if percentage >= 90 else 'Good!' if percentage >= 70 else 'Practice more!'}"
    }

@app.get("/user/progress/{user_id}")
async def get_user_progress(user_id: str):
    if user_id not in users_db:
        raise HTTPException(404, "User not found")
    
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

# Load saved data on startup
try:
    with open("data.pkl", "rb") as f:
        loaded = pickle.load(f)
        globals().update(loaded)
except FileNotFoundError:
    pass

# ========== SERVER STARTUP ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🎌 HSK AI Tutor - Pragmatic tutor")
    print("=" * 60)
    print(f"📚 Database: {len(words_db)} words HSK 1-6")
    print(f"👥 Registered users: {len(users_db)}")
    print(f"🧪 Created tests: {len(tests_db)}")
    print("=" * 60)
    print("🚀 Starting server on https://saiyan-3s8s.onrender.com/")
    print("📚 Documentation: https://saiyan-3s8s.onrender.com/docs")
    print("🌐 Frontend: open frontend.html in browser")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on changes
    )