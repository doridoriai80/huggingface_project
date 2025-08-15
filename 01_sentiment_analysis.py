# 01_sentiment_analysis.py
from transformers import pipeline

# 1) 파이프라인 로드
classifier = pipeline("sentiment-analysis")

# 2) 테스트 문장
sentences = [
    "I love Hugging Face! It's amazing.",
    "This movie was really bad..."
]

# 3) 결과 출력
for text in sentences:
    result = classifier(text)
    print(f"문장: {text}")
    print(f"결과: {result}\n")
