# backend/nlp/segmenter.py
import re
from typing import List
from dataclasses import dataclass


@dataclass
class ReviewSegment:
    text: str
    index: int


def segment_review(review_text: str) -> List[ReviewSegment]:
    if not review_text or not review_text.strip():
        return []

    # Разбиваем по предложениям
    sentences = re.split(r'(?<=[.!?])\s+', review_text.strip())

    # Разбиваем по контрастным союзам внутри предложений
    refined = []
    contrast_pattern = r'\s*[,;]\s*(?:но|однако|хотя|зато|впрочем|тем не менее|при этом)\s+'

    for sentence in sentences:
        parts = re.split(contrast_pattern, sentence, flags=re.IGNORECASE)
        refined.extend([p.strip() for p in parts if p.strip()])

    # Объединяем слишком короткие сегменты (< 20 символов)
    merged = []
    buffer = ""
    for part in refined:
        if len(part) < 20 and buffer:
            buffer += " " + part
        elif len(part) < 20 and not buffer:
            buffer = part
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(part)
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    segments = [
        ReviewSegment(text=text, index=i)
        for i, text in enumerate(merged)
        if text.strip()
    ]
    return segments