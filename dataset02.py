from datasets import load_dataset

# 한국어 자연어 추론 데이터셋 로드
dataset = load_dataset("kor_nli")

print("한국어 NLI 데이터셋 정보:")
print(dataset)

# 데이터 샘플 확인
for i in range(3):
    sample = dataset['train'][i]
    print(f"\n샘플 {i+1}:")
    print(f"전제: {sample['premise']}")
    print(f"가설: {sample['hypothesis']}")
    print(f"라벨: {sample['label']}")
    
    # 라벨을 텍스트로 변환
    label_names = ['함의', '중립', '모순']
    print(f"라벨(텍스트): {label_names[sample['label']]}")