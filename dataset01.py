from datasets import load_dataset

# IMDB 영화 리뷰 데이터셋 로드
dataset = load_dataset("imdb")

# 데이터셋 구조 확인
print("데이터셋 정보:")
print(dataset)
print(f"\n훈련 데이터 크기: {len(dataset['train'])}")
print(f"테스트 데이터 크기: {len(dataset['test'])}")

# 샘플 데이터 확인
print("\n첫 번째 훈련 샘플:")
sample = dataset['train'][0]
print(f"리뷰: {sample['text'][:200]}...")
print(f"라벨: {sample['label']} ({'긍정' if sample['label'] == 1 else '부정'})")

# 라벨 분포 확인
train_labels = dataset['train']['label']
positive_count = sum(train_labels)
negative_count = len(train_labels) - positive_count
print(f"\n라벨 분포:")
print(f"긍정: {positive_count}, 부정: {negative_count}")