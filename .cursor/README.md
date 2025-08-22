# Cursor Rules for PR Review Assistant

이 디렉토리는 PR Review Assistant 프로젝트의 개발 가이드라인과 규칙을 정의합니다.

## 규칙 파일 구조

### 00-architecture.mdc
- 시스템 아키텍처 및 설계 원칙
- 멀티 에이전트 구조 및 워크플로
- 기술 스택 요구사항

### 10-coding-conventions.mdc
- Python/TypeScript 코딩 컨벤션
- 테스트 작성 규칙
- 코드 품질 기준

### 20-agent-contracts.mdc
- 에이전트 간 메시지 계약
- JSON Schema 정의
- 오류 처리 및 재시도 정책

### 25-data-models.mdc
- 데이터 모델 및 스키마 설계
- 타입 정의 및 검증 규칙
- 스키마 버전 관리

### 30-tool-specs.mdc
- 툴 인터페이스 정의
- 공통 응답 형식
- 성능 및 리소스 가드레일

### 35-workflow-orchestration.mdc
- 워크플로 설계 및 실행
- 에이전트 오케스트레이션
- 상태 관리 및 모니터링

### 40-prompt-style.mdc
- LLM 프롬프트 설계 가이드
- 에이전트별 프롬프트 템플릿
- 프롬프트 최적화 기법

### 45-testing-quality.mdc
- 테스트 전략 및 구조
- 품질 보증 방법
- 성능 테스트 가이드라인

### 50-io-and-state.mdc
- 입출력 파일 레이아웃
- 상태 스냅샷 관리
- 캐시 및 재실행 규칙

### 55-deployment-operations.mdc
- 배포 전략 및 CI/CD
- 환경별 설정 관리
- 모니터링 및 로깅

### 60-quality-gates.mdc
- 품질 게이트 및 등급 체계
- 차단 및 경고 규칙
- 품질 점수 계산 방법

### 70-ci-and-repo-policy.mdc
- GitHub Actions 워크플로우
- 브랜치 보호 규칙
- PR 자동화 정책

### 80-security-and-privacy.mdc
- 보안 및 개인정보보호 정책
- 민감 정보 마스킹 규칙
- 접근 제어 및 권한 관리

### 90-observability-and-performance.mdc
- 로깅 및 메트릭 수집
- 성능 모니터링 가이드라인
- 비용 관리 및 최적화

### 95-project-structure.mdc
- 프로젝트 디렉토리 구조
- 파일 명명 규칙
- 모듈 구조 가이드라인

## 사용 방법

1. **개발 시작 전**: 관련 규칙 파일을 먼저 읽고 이해
2. **코드 작성 시**: 규칙에 정의된 가이드라인 준수
3. **리뷰 시**: 규칙 준수 여부 확인
4. **규칙 업데이트**: 필요 시 규칙 파일 수정 및 팀 공유

## 규칙 준수 확인

- `pre-commit` 훅을 통한 자동 검증
- 정기적인 규칙 준수 검토
- 팀 내 규칙 인지도 향상 교육

## 참고 사항

- 모든 규칙은 PRD 요구사항을 기반으로 작성
- 실제 구현 코드는 별도 파일에서 작성
- 규칙은 지속적으로 개선 및 업데이트
