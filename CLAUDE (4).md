# CLAUDE.md

레벨미업학원 워크북 앱 — FastAPI + Supabase 기반 영어 지문 워크북/문제 자동 생성 시스템.

## 브랜치

**`feature/backend-from-design`** 에서만 작업한다. (main보다 최신)

## 작업 규칙

- **코드·템플릿 커밋은 수지님이 한다.** Claude는 열람·분석·검증·수정안 제시까지.
  수정본을 보여주고 판단을 맡긴다. 올리는 건 수지님이다.
- **예외 — 이 두 문서만 Claude가 직접 커밋해도 된다(상시 허용):**
  `CLAUDE.md`, `docs/2회독_변형문제.md`
- `git push`는 이 환경에서 막혀 있다(403). 문서 커밋은 GitHub API로 한다.

## 보고 방식

- 사용자는 **수지님**이다. 그렇게 부른다.
- **`(가)` `(서)` `(누)` 같은 항목 기호를 보고에 쓰지 않는다.** 무슨 문제인지 말로 푼다.
  그 기호는 검수 기준 문서와 검사기 안에서만 쓰는 내부 코드다.
  검사기 출력에 기호가 박혀 나오니 **그대로 붙여넣지 말 것.**
- 문항 오류는 **지문 범위 → 문제 번호 → 문제 유형 → 무슨 오류** 순으로 밝힌다.

```
공통영어2 능률(오) 01과 04번 — 4번 불일치, 진술 '라' — 5번 빈칸 정답이 그대로 노출됨
└ 지문 범위 ──────────┘ └ 번호 ┘ └ 유형 ┘ └────── 무슨 오류 ──────┘
```

  교재명을 빼면 안 된다 — 같은 번호가 교재마다 있다.
  문제 유형을 빼도 안 된다 — 무엇을 묻는 문항인지 알아야 판단이 선다.

## 문서를 고칠 때

`CLAUDE.md`·`docs/2회독_변형문제.md` 는 **마크다운 원문 그대로** 올려야 한다.
GitHub 미리보기 화면의 글을 복사해 붙이면 `#` 제목·`**굵게**`·표의 `|` 가 전부 날아가
구조가 사라진다(실제로 한 번 그렇게 올라갔다). 파일 내용을 그대로 복사할 것.

## 지금 하는 작업

**2회독 변형문제(유형 A/B) 수정** 중이다.

👉 2회독 관련 작업이면 **`docs/2회독_변형문제.md` 를 먼저 읽을 것.**
   문항 구성, 파일 지도, 캐시 버전 규칙, 알려진 함정이 전부 거기 있다.

## 시스템 구성

| 모듈 | 기능 | 회독 |
|---|---|---|
| `pipeline.py` + `template.html` | 워크북 8단계 (Lv.1~Lv.10) | 1회독 |
| `variation/` + `variation.html` / `variation_b.html` | 변형문제 유형 A·B | **2회독** |
| `seosul/` | 서술형 종합 | 2회독 |
| `sheet/` | 선생님 분석지 | 0회독 |
| `main.py` | 지문 관리·인증·라우터 등록 | 공통 |

## 환경변수

`SUPABASE_URL`, `SUPABASE_KEY`(또는 `SUPABASE_SERVICE_KEY`), `ANTHROPIC_API_KEY`,
`APP_PASSWORD`(기본 `levelmeup2026`), `CLAUDE_MODEL`, `VARIATION_OUTPUT_DIR`

없으면 Supabase는 `data/passages.json` 로컬 폴백으로 돌고 생성 기능은 멈춘다.
배포는 Dockerfile(uvicorn). 로그는 Railway 대시보드.

## 로컬 확인

```bash
pip install -r requirements.txt
python3 -c "import main"          # 라우터 3개(variation/seosul/sheet) 다 떠야 정상
python3 tests/test_vocab_gate.py  # 어휘 게이트 테스트
```

## 미해결 이슈

- `passages` 테이블에 **`order_index` 컬럼이 없는데** `main.py`가 upsert에 실어 보낸다
  → `/api/passages/reorder`로 바꾼 순서가 조용히 사라진다. (컬럼 추가 또는 코드 제거 필요)
- `notice`, `seosul_types` 테이블은 **RLS 비활성 + 정책 0개** — anon 키로 읽기·쓰기 전부 열림.
