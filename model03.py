from transformers import MarianMTModel, MarianTokenizer

# 한국어-영어 번역 모델 설정
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_korean_to_english(korean_text):
    # 토큰화
    tokens = tokenizer(korean_text, return_tensors="pt", padding=True)
    
    # 번역 실행
    translated = model.generate(**tokens, max_length=512)
    
    # 결과 디코딩
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return english_text

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