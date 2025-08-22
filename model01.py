from transformers import pipeline

# 감정 분석 파이프라인을 생성합니다. 이 파이프라인은 사전 학습된 모델을 사용하여 텍스트의 감정을 분석합니다.
# 여기서는 'cardiffnlp/twitter-roberta-base-sentiment-latest' 모델을 사용합니다.

# 분석할 텍스트를 정의합니다. 이 예제에서는 단일 문장을 사용합니다.

# 분석 결과를 출력합니다. 결과는 감정 레이블과 해당 확신도로 구성됩니다.

# 여러 개의 텍스트를 동시에 처리할 수 있습니다. 이 경우, 각 텍스트에 대한 감정 분석 결과를 출력합니다.

# 감정 분석 파이프라인 생성
classifier = pipeline("sentiment-analysis", 
                     model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# 텍스트 감정 분석 실행
text = "오늘 날씨가 정말 좋네요!"
result = classifier(text)
print(f"감정: {result[0]['label']}, 확신도: {result[0]['score']:.4f}")

# 여러 텍스트 동시 처리
texts = [
    "이 제품 정말 마음에 들어요!",
    "서비스가 너무 느려서 짜증나네요.",
    "그냥 보통이에요.",
    "싫어요.",
    "좋아요."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"'{text}' → {result['label']} ({result['score']:.4f})")

# 출력 결과는 각 텍스트에 대한 감정 레이블과 해당 확신도로 구성됩니다.
# 예를 들어, '감정: POSITIVE, 확신도: 0.9876'이라는 출력은 해당 텍스트가 긍정적인 감정을 가지고 있으며,
# 모델이 이 결과에 대해 98.76%의 확신을 가지고 있음을 나타냅니다.
# 여러 텍스트를 처리할 때는 각 텍스트에 대해 개별적으로 결과가 출력됩니다.