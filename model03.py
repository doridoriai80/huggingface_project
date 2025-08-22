from transformers import pipeline

# 한국어-영어 번역 파이프라인 생성 (SentencePiece 라이브러리 불필요)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

def translate_korean_to_english(korean_text):
    # 번역 실행
    result = translator(korean_text)
    return result[0]['translation_text']

# 번역 테스트
korean_sentences = [
    "안녕하세요, 반갑습니다.",
    "오늘 날씨가 정말 좋네요.",
    "인공지능 기술이 빠르게 발전하고 있습니다."
]

for korean in korean_sentences:
    english = translate_korean_to_english(korean)
    print(f"한국어: {korean}")
    print(f"영어: {english}\n")