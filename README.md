# FirstAI - Hugging Face 기반 AI 프로젝트

이 프로젝트는 Hugging Face의 transformers 라이브러리를 사용한 다양한 자연어 처리(NLP) 예제들을 포함하고 있습니다.

## 📋 프로젝트 개요

- **목적**: Hugging Face를 활용한 AI 모델 학습 및 활용 방법 학습
- **주요 기능**: 감정 분석, 텍스트 분류, 웹 인터페이스 데모
- **사용 기술**: Python, Transformers, Datasets, Gradio

## 🛠️ 개발 환경 구축

### 1. Python 환경 설정
```bash
# Python 3.8 이상 필요
python --version

# 가상환경 생성 (권장)
python -m venv firstai_env
source firstai_env/bin/activate  # macOS/Linux
# firstai_env\Scripts\activate  # Windows
```

### 2. 필요한 패키지 설치
```bash
# 기본 허깅페이스 라이브러리
pip install transformers datasets torch torchvision torchaudio

# 추가 유용한 라이브러리들
pip install accelerate evaluate gradio streamlit

# 시각화 및 데이터 처리
pip install matplotlib seaborn pandas numpy pillow
```

### 3. 설치 확인
```bash
python -c "import transformers; print('Transformers 설치 완료')"
python -c "import gradio; print('Gradio 설치 완료')"
```

## 📁 프로젝트 구조

```
FirstAI/
├── README.md                    # 프로젝트 설명서
├── 01_sentiment_analysis.py     # 기본 감정 분석 예제
├── 02_text_classification.py    # 텍스트 분류 모델 학습
└── 03_gradio_demo.py           # 웹 인터페이스 데모
```

## 🚀 예제 소스 설명

### 1. 01_sentiment_analysis.py - 기본 감정 분석
**목적**: Hugging Face 파이프라인을 사용한 간단한 감정 분석

**주요 기능**:
- 사전 훈련된 감정 분석 모델 사용
- 텍스트의 긍정/부정 감정 분류
- 신뢰도 점수 제공

**실행 방법**:
```bash
python 01_sentiment_analysis.py
```

**결과 예시**:
```
문장: I love Hugging Face! It's amazing.
결과: [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. 02_text_classification.py - 텍스트 분류 모델 학습
**목적**: IMDB 데이터셋을 사용한 텍스트 분류 모델 직접 학습

**주요 기능**:
- DistilBERT 모델을 사용한 분류 모델 학습
- IMDB 영화 리뷰 데이터셋 활용
- 커스텀 모델 훈련 및 저장

**실행 방법**:
```bash
python 02_text_classification.py
```

**특징**:
- 1% 데이터 샘플로 빠른 학습
- 결과는 `./results` 디렉토리에 저장
- 로그는 `./logs` 디렉토리에 저장

### 3. 03_gradio_demo.py - 웹 인터페이스 데모
**목적**: 감정 분석 모델을 웹 인터페이스로 배포

**주요 기능**:
- Gradio를 사용한 웹 기반 UI
- 실시간 감정 분석
- 사용자 친화적 인터페이스

**실행 방법**:
```bash
python 03_gradio_demo.py
```

**접속 방법**:
- 브라우저에서 `http://localhost:7860` 접속
- 텍스트 입력 후 즉시 감정 분석 결과 확인

## 📚 학습 목표

이 프로젝트를 통해 다음을 학습할 수 있습니다:

1. **Hugging Face 파이프라인 사용법**
   - 사전 훈련된 모델 활용
   - 간단한 텍스트 분석

2. **커스텀 모델 학습**
   - 데이터셋 로드 및 전처리
   - 모델 훈련 및 저장

3. **AI 모델 배포**
   - 웹 인터페이스 구축
   - 사용자 친화적 서비스 제공

## 🔧 문제 해결

### 일반적인 오류 및 해결방법

1. **CUDA 관련 오류**
   ```bash
   # CPU만 사용하도록 설정
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **메모리 부족 오류**
   - 배치 크기 줄이기
   - 더 작은 모델 사용

3. **패키지 설치 오류**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📖 추가 학습 자료

- [Hugging Face 공식 문서](https://huggingface.co/docs)
- [Transformers 튜토리얼](https://huggingface.co/docs/transformers/tutorials)
- [Gradio 가이드](https://gradio.app/docs/)

## 🤝 기여하기

1. 이슈 등록
2. 포크 생성
3. 브랜치 생성
4. 변경사항 커밋
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**시작하기**: `01_sentiment_analysis.py`부터 실행해보세요!
