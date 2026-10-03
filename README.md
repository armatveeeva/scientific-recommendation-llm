#  Обработка и структурирование научных публикаций в области материаловедения

Система семантического поиска и рекомендаций научных публикаций с использованием **векторных баз данных (FAISS)** и **NLP-суммаризации**.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)
[![CI/CD](https://github.com/armatveeeva/scientific-recommendation-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/armatveeeva/scientific-recommendation-llm/actions)

Проект демонстрирует построение **production-ready ML-сервиса** для семантического поиска научных статей. Система принимает текстовый запрос, находит наиболее релевантные публикации с помощью векторного поиска (FAISS) и генерирует краткую выжимку (summary) с использованием NLP-алгоритмов.

## Описание предметной области

В научно-исследовательских и инженерных организациях ежедневно публикуются тысячи новых научных статей, технических отчетов и патентов. 

Традиционный поиск по ключевым словам часто неэффективен. Он не учитывает семантический смысл, синонимы и контекст, заставляя исследователей тратить огромное количество времени на ручной просмотр нерелевантных документов.

## Технологический стек

- **Язык:** Python 3.11+
- **Backend & API:** FastAPI (асинхронный фреймворк), Uvicorn (ASGI-сервер), Pydantic (валидация данных)
- **Тестирование:** Pytest + httpx (интеграционные тесты API)
- **Векторизация текста:** Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Векторный поиск:** FAISS (`faiss-cpu`) — библиотека от Meta для поиска похожих векторов
- **NLP & Суммаризация:** Sumy (алгоритм TextRank для экстрактивной суммаризации)
- **Хранение данных:** SQLite + JSON (метаданные статей)
- **CI/CD:** GitHub Actions (автоматический пайплайн тестирования)

## Быстрый старт

### Предварительные требования

- Python 3.11 или выше
- Менеджер пакетов pip

### Установка и настройка
```bash
git clone https://github.com/armatveeeva/scientific-recommendation-llm.git
cd scientific-recommendation-llm
