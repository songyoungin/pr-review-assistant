# MVP 엔드-투-엔드 Orchestrator 뼈대 설계서

목표: PRD의 FR-1~FR-5를 하나의 MVP 엔드-투-엔드 파이프라인으로 실행하고, 최종 리포트를 JSON/Markdown로 생성하여 CI에 게시한다.

범위: DiffSummarizer, CodeReviewer, DocsConsistencyChecker, SchemaChangeAnalyst의 기본 실행 흐름을 Orchestrator가 조정하고, 모든 결과를 일관된 리포트로 합친다. 샘플 PR에 한해 로컬에서 동작 가능하도록 구성한다.

## 입력/출력 정의(개요)
- 입력: PR 컨텍스트(저장소 정보, PR 번호/헤드-베이스, Diff 경로), 변경 파일 목록, 확장 가능한 API 힌트
- 출력: 최종 리포트 JSON/Markdown + 각 에이전트별 근거(파일:라인) 포함 요약

## 아키텍처 개요
- Orchestrator: 전체 흐름 조율, 상태 관리, 에이전트 실행 순서 결정 및 결과 집계
- 에이전트 인터페이스: DiffSummarizer, CodeReviewer, DocsConsistencyChecker, SchemaChangeAnalyst의 표준화된 진입점 재사용
- 병렬 실행: 가능하면 에이전트를 병렬로 실행하고 결과를 수집해 최종 리포트 구성
- 데이터 계약: 상태 스키마와 각 에이전트 입력/출력 스키마를 JSON Schema 수준으로 맞춰 계약화

## MVP 흐름(고수준)
1) 초기화: PR 컨텍스트 및 Diff 정보를 수집하고 상태를 초기화한다.
2) 병렬 실행 준비: 네 에이전트의 입력 구성 및 호출 경로를 준비한다.
3) 에이전트 실행: DiffSummarizer, CodeReviewer, DocsConsistencyChecker, SchemaChangeAnalyst를 호출하고 결과를 수집한다.
4) 결과 집계: 수집된 결과를 하나의 리포트로 합치고 TL;DR/하이라이트를 생성한다.
5) 출력/게시: 최종 리포트를 JSON/MD로 노출하고 CI에 아티팩트로 저장/게시한다.
6) 검증 포인트: 각 에이전트 출력에 근거 파일:라인이 포함되었는지 간단 검증을 수행한다.

## 데이터 계약(초안)
```json
{
  "OrchestratorInput": {
    "repo": {"provider": "string", "url": "string", "default_branch": "string", "clone_path": "string"},
    "pr": {"number": "number", "title": "string", "base": "string", "head": "string", "author": "string", "url": "string", "created_at": "string", "updated_at": "string"},
    "diff": {"unified_patch_path": "string", "changed_files_path": "string", "total_lines": "number", "files_count": "number"},
    "config": {"max_highlights": "number", "include_risks": "boolean", "parallel_execution": "boolean"}
  },
  "exportFormat": {"format": "json|markdown"}
}
```

```json
{
  "OrchestratorOutput": {
    "final_report": {
      "diff_summary": {},
      "code_review": {},
      "docs_consistency": {},
      "schema_analysis": {},
      "summary": "string TL;DR"
    },
    "evidence": [
      {"type": "diff|file|line|rule|api|schema", "target": "string", "line_range": {"start": 1, "end": 10}, "file_path": "string", "rule_id": "string", "rule_version": "string", "description": "string", "confidence": 0.9}
    ],
    "execution_time_ms": 0
  }
}
```

## 에이전트 입력/출력 요약
- DiffSummarizerInput: diff_path, files_path, max_highlights, include_risks
- DiffSummarizerOutput: DiffSummary(tldr, highlights, risks, deployment_impact, compatibility_impact)
- CodeReviewerInput: diff_path, files, language_hints, ruleset_version, severity_threshold
- CodeReviewerOutput: CodeReviewResult(findings, coverage_hints, quality_score, security_score, performance_score)
- DocsConsistencyCheckerInput: diff_path, doc_globs, api_hints, check_types
- DocsConsistencyCheckerOutput: DocsConsistencyResult(mismatches, score, missing_docs, outdated_docs, patch_suggestions)
- SchemaChangeAnalystInput: diff_path, schema_roots, engine, include_ops_guide
- SchemaChangeAnalystOutput: SchemaAnalysisResult(ddl_changes, breaking_changes, ops_guide, migration_complexity, total_impact_score)

## 에러/재시도 전략
- 부분 실패 시 해당 섹션은 partial로 표시하고, 나머지 섹션은 정상 진행
- 재시도는 E001/E002/E005 등 재시도 가능 오류에 한해 지수 백오프 최대 3회

## 검증 및 테스트 개요
- 간단한 단위/통합 테스트 설계와 MVP 시나리오 기반 실험 계획 포함
- 샘플 PR로 로컬에서 엔드-투-엔드 검증

## 비고
- MVP는 기존 모듈 재사용에 집중하며, 인터페이스 계약의 엄격한 준수로 차후 확장을 용이하게 한다.
