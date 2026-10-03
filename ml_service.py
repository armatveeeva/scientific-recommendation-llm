import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

print("Загрузка ML-моделей и FAISS индекса...")

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

summarizer = TextRankSummarizer()

index = faiss.read_index("articles_faiss.index")
with open("articles_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Все компоненты загружены успешно!")


def find_similar_articles(query_text: str, top_k: int = 2):
    query_emb = embedding_model.encode(query_text)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_array = np.array([query_emb]).astype('float32')

    distances, indices = index.search(query_array, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "similarity": float(distances[0][i]),
                "title": metadata[idx]["title"],
                "abstract": metadata[idx]["abstract"],
                "authors": metadata[idx]["authors"],
                "year": metadata[idx]["year"]
            })
    return results


def summarize_text(text: str) -> str:
    """Экстрактивная суммаризация с защитой от сбоев"""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("russian"))
        sentences_count = min(2, len(text.split('.')))
        summary = summarizer(parser.document, sentences_count=sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception:
        return text.split('. ')[0] + '.'
