import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 한국어 GPT-2 모델 로드
model_name = "gpt2"  # Changed to a more reliable model

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Tokenizer loading failed: {e}")
    # Handle the error, e.g., by using a default tokenizer or exiting
    exit(1)

try:
    model = GPT2LMHeadModel.from_pretrained(model_name)
except Exception as e:
    print(f"Model loading failed: {e}")
    # Handle the error, e.g., by using a default model or exiting
    exit(1)

# 특별 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 텍스트 생성 함수
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 사용 예제
prompt = "인공지능의 미래는"
generated = generate_text(prompt)
print(f"생성된 텍스트: {generated}")