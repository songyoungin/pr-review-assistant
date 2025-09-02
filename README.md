# PR Review Assistant

PR 리뷰를 위한 지능형 어시스턴트 도구입니다. 코드 변경사항을 분석하고, 품질을 평가하며, 개선 방안을 제시합니다.

## 주요 기능

### 🔍 코드 분석 도구 (Tools)

#### Python AST 분석기 (`tools/code_analysis/python_ast.py`)
Python 코드의 구조와 품질을 종합적으로 분석하는 도구입니다.

**주요 기능:**
- **AST 파싱**: Python 코드를 Abstract Syntax Tree로 변환하여 구조 분석
- **복잡도 측정**: 순환 복잡도(Cyclomatic Complexity) 계산
- **코드 품질 지표**: 함수/클래스 수, 중첩 레벨, 주석 비율 등
- **코드 스멜 감지**: 매직 넘버, 하드코딩된 문자열, 긴 함수 등
- **품질 점수**: 0-100점 척도로 코드 품질 평가
- **개선 권장사항**: 구체적인 리팩토링 제안

**사용법:**
```bash
# 단일 파일 분석
python tools/code_analysis/python_ast.py path/to/file.py

# 디렉토리 전체 분석
python tools/code_analysis/python_ast.py path/to/directory/

# 현재 디렉토리 분석
python tools/code_analysis/python_ast.py
```

#### Git 변경사항 감지기 (`tools/git/git_changes.py`)
Git 저장소의 변경사항을 상세하게 분석하는 도구입니다.

**주요 기능:**
- 브랜치 간 변경사항 비교
- 커밋별 상세 분석
- 파일별 diff 정보 추출
- 변경 통계 및 요약

#### 코드-문서 매칭기 (`tools/comparison/code_doc_matcher.py`)
코드와 문서의 일치성을 검증하는 도구입니다.

### 🤖 지능형 에이전트 (Agents)

#### 변경사항 요약 에이전트 (`agents/change_summarizer/`)
코드 변경사항을 자동으로 요약하고 핵심 내용을 추출합니다.

#### 리뷰 에이전트 (`agents/reviewer/`)
코드 리뷰를 수행하고 품질 평가를 제공합니다.

#### 문서 감사 에이전트 (`agents/doc_auditor/`)
문서의 품질과 일관성을 검사합니다.

#### 스키마 비교 에이전트 (`agents/schema_comparator/`)
데이터 스키마 변경사항을 분석하고 비교합니다.

## 설치 및 설정

### 요구사항
- Python 3.12+
- Git 저장소

### 설치
```bash
# 저장소 클론
git clone <repository-url>
cd pr-review-assistant

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

# 의존성 설치
# uv 사용 권장
python -m pip install -U pip
python -m pip install uv
uv sync --yes
```

### 사용법
```bash
# 가상환경 활성화
source .venv/bin/activate

# Python AST 분석 실행
python tools/code_analysis/python_ast.py

# Git 변경사항 분석
python tools/git/git_changes.py
```

## 프로젝트 구조

```
pr-review-assistant/
├── tools/                    # 분석 도구들
│   ├── code_analysis/       # 코드 분석 도구
│   ├── git/                 # Git 관련 도구
│   └── comparison/          # 비교 분석 도구
├── agents/                   # 지능형 에이전트들
│   ├── change_summarizer/   # 변경사항 요약
│   ├── reviewer/            # 코드 리뷰
│   ├── doc_auditor/         # 문서 감사
│   └── schema_comparator/   # 스키마 비교
├── graph/                    # 워크플로우 그래프
├── prompts/                  # 프롬프트 템플릿
└── tests/                    # 테스트 코드
```

## 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
