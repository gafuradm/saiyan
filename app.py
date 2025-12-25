import sys
import os

# Добавляем корень проекта в PATH (это ключевой фикс для Render)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)