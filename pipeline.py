#!/usr/bin/env python3
"""Workbook generation pipeline"""
PIPELINE_VERSION = "v9-curl-final"
import asyncio, json, os, sys, time, random, re
from pathlib import Path

# ============================================================
# 설정
# ============================================================
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
TEMPLATE_DIR = Path(__file__).parent
DATA_DIR = TEMPLATE_DIR / "data"

# date-based output folder
from datetime import datetime
try:
    TODAY = datetime.now().strftime("%-m월%-d일") if os.name != 'nt' else datetime.now().strftime("%#m월%#d일")
except:
    TODAY = datetime.now().strftime("%m월%d일")
OUTPUT_DIR = TEMPLATE_DIR / "output" / TODAY

# ============================================================
# Safe print (encoding-safe)
# ============================================================
def _safe_print(msg):
    try:
        print(str(msg))
    except Exception:
        pass

# ============================================================
# 문장 분리 (Dr. Mr. Ms. Mrs. Prof. etc. 경칭 보호)
# ============================================================
# 마침표 뒤 공백에서 분리하되, 경칭/약어 뒤는 분리하지 않음
_ABBREVS = r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bSt)(?<!\bvs)(?<!\bNo)(?<!\bJr)(?<!\bSr)(?<!\bet)(?<!\bMrs)(?<!\bal)(?<!\bProf)(?<!\bGen)(?<!\bGov)(?<!\bSgt)(?<!\bCpl)(?<!\bLt)(?<!\bCo)(?<!\bInc)(?<!\bLtd)(?<!\bCorp)(?<!\bDept)(?<!\bEst)(?<!\bFig)(?<!\bVol)(?<!\bRev)'

def split_sentences(text: str) -> list:
    """영어 지문을 문장 단위로 분리 (경칭/약어 마침표 보호)"""
    # 1단계: 경칭/약어의 마침표를 임시 토큰으로 치환 (단어 경계 기반)
    protected = text
    abbrevs = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Jr.', 'Sr.', 'St.',
        'vs.', 'etc.', 'No.', 'Vol.', 'Fig.', 'Gen.', 'Gov.', 'Rev.',
        'Sgt.', 'Cpl.', 'Lt.', 'Co.', 'Inc.', 'Ltd.', 'Corp.', 'Dept.',
        'Est.', 'al.', 'e.g.', 'i.e.', 'U.S.', 'U.K.', 'U.N.',
    ]
    replacements = {}
    for ab in abbrevs:
        token = ab.replace('.', '§DOT§')
        # 단어 경계(\b)를 사용하여 정확한 약어만 매치
        # 예: 'al.'은 단독 단어일 때만 (et al.), 'meal.'의 'al.'은 매치 안 됨
        pattern = r'(?<!\w)' + re.escape(ab)
        if re.search(pattern, protected):
            replacements[token] = ab
            protected = re.sub(pattern, token, protected)
    
    # 2단계: 일반 문장 분리 (닫는 따옴표 뒤 대문자 일반문장만 분리, 따옴표 연속은 합침)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])(?!["\u201c])|(?<=[.!?]["\u201d])\s+(?=[A-Z])(?!["\u201c])', protected) if s.strip()]
    
    # 3단계: 토큰을 원래 마침표로 복원
    restored = []
    for s in sentences:
        for token, original in replacements.items():
            s = s.replace(token, original)
        restored.append(s)
    
    return restored

# ============================================================
# Claude API call (curl subprocess - ONLY method that bypasses Python latin-1)
# ============================================================
API_URL = "https://api.anthropic.com/v1/messages"

def call_claude(system_prompt: str, user_prompt: str, max_retries=2, max_tokens=4096) -> str:
    """Claude API via curl subprocess - zero Python encoding involvement"""
    import subprocess, tempfile
    if not API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    body_json = json.dumps(body, ensure_ascii=False)
    
    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', delete=False) as tmp:
                tmp.write(body_json)
                tmp_path = tmp.name
            
            result = subprocess.run(
                [
                    'curl', '-s', '-X', 'POST', API_URL,
                    '-H', f'x-api-key: {API_KEY}',
                    '-H', 'anthropic-version: 2023-06-01',
                    '-H', 'content-type: application/json; charset=utf-8',
                    '-d', f'@{tmp_path}'
                ],
                capture_output=True,
                timeout=120
            )
            
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass
            
            if result.returncode != 0:
                raise Exception(f"curl error: {result.stderr.decode('utf-8','replace')[:200]}")
            
            data = json.loads(result.stdout.decode('utf-8'))
            if 'error' in data:
                raise Exception(f"API error: {json.dumps(data['error'])[:200]}")
            return data["content"][0]["text"].strip()
        except Exception as e:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass
            _safe_print(f"  [WARN] API attempt {attempt+1} failed: {str(e)[:100]}")
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
            else:
                raise

def call_claude_json(system_prompt: str, user_prompt: str, max_retries=3, max_tokens=4096) -> dict:
    """Claude API call -> JSON parse with retry"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            text = call_claude(system_prompt, user_prompt, max_retries=0, max_tokens=max_tokens)
            return _parse_json_robust(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            try:
                _safe_print(f"  [WARN] JSON parse fail (try {attempt+1}/{max_retries+1}): {str(e)[:80]}")
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(2)
    raise ValueError(f"JSON parse final fail: {str(last_error)[:200]}")

def _parse_json_robust(text: str) -> dict:
    """여러 전략으로 JSON 파싱 시도"""
    # 1) 코드블록 제거
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    text = text.strip()
    
    # 2) 직접 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 3) JSON 부분만 추출 (가장 바깥 { } 매칭)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # 4) 이스케이프 안 된 따옴표 수정: value 안의 " → \"
    try:
        fixed = _fix_json_quotes(text if not match else match.group())
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass
    
    # 5) 줄바꿈/탭 이스케이프
    try:
        cleaned = text if not match else match.group()
        # JSON 문자열 안의 실제 줄바꿈을 \n으로 변환
        cleaned = re.sub(r'(?<=": ")([^"]*?)(?=")', lambda m: m.group(1).replace('\n', '\\n').replace('\t', '\\t'), cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, Exception):
        pass
    
    raise json.JSONDecodeError("모든 파싱 전략 실패", text[:200], 0)

def _fix_json_quotes(text: str) -> str:
    """JSON 문자열 안의 이스케이프 안 된 따옴표를 수정"""
    result = []
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                # 다음 문자 확인: , } ] : 공백이면 문자열 끝
                rest = text[i+1:i+10].lstrip()
                if not rest or rest[0] in ',}]:':
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')  # 이스케이프
                    continue
        else:
            result.append(ch)
    return ''.join(result)

# ============================================================
# 단계별 저장/로드
# ============================================================
def _run_async(coro):
    """Run an async coroutine from sync context (pipeline runs in a thread)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We're inside an async event loop (FastAPI thread) – schedule as a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

def save_step(passage_dir: Path, step_name: str, data: dict):
    # Save locally
    passage_dir.mkdir(parents=True, exist_ok=True)
    path = passage_dir / f"{step_name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _safe_print(f"  Saved: {step_name}.json")
    # Save to Supabase
    try:
        import supa
        if supa._enabled():
            cache_key = passage_dir.name
            _run_async(supa.save_step_supa(cache_key, step_name, data))
            _safe_print(f"  Saved to Supabase: {cache_key}/{step_name}")
    except Exception as e:
        _safe_print(f"  [supa] save error: {str(e)[:80]}")

def load_step(passage_dir: Path, step_name: str) -> dict | None:
    # Try local first
    path = passage_dir / f"{step_name}.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Try Supabase
    try:
        import supa
        if supa._enabled():
            cache_key = passage_dir.name
            data = _run_async(supa.get_step(cache_key, step_name))
            if data:
                # Save locally for future use
                passage_dir.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                _safe_print(f"  Loaded from Supabase: {step_name}")
                return data
    except Exception as e:
        _safe_print(f"  [supa] load error: {str(e)[:80]}")
    return None

# ============================================================
# SYSTEM PROMPT (공통)
# ============================================================
SYS_JSON = """You are an English exam content generator for Korean high school students.
Return ONLY valid JSON. No markdown fences. No explanations. No preamble.
All Korean text must use proper Korean. All English must be grammatically correct."""

SYS_JSON_KR = """당신은 한국 고등학생을 위한 영어 시험 콘텐츠 생성기입니다.
반드시 유효한 JSON만 반환하세요. 마크다운, 설명, 서문 없이 JSON만 출력하세요.
한국어는 자연스럽게, 영어는 문법적으로 정확하게 작성하세요."""

# ============================================================
# STEP 1: 기본 분석 (어휘 + 번역 + 핵심문장)
# ============================================================
def step1_basic_analysis(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step1_basic")
    if cached:
        _safe_print("  step1: using cache")
        return cached

    sentences_regex = split_sentences(passage)
    sent_count = len(sentences_regex)

    _safe_print("  step1: basic analysis...")
    prompt = f"""다음 영어 지문을 분석하여 JSON을 생성하세요.

[지문 - 총 {sent_count}개 문장]
{passage}

[생성 항목]
1. vocab: 핵심 어휘 14개 (각각 word, meaning(한국어), synonyms(영어 동의어 4개 쉼표구분))
2. translation: 지문 전체의 자연스러운 한국어 번역
3. sentences: 지문의 모든 문장을 개별 배열로 분리 (정확히 {sent_count}개!)
   - 짧은 문장도 절대 합치지 마세요 (예: "That's not loyalty." 는 독립 문장)
   - 문장을 절대 분리하지 마세요 (세미콜론 ; 으로 연결된 것은 1문장)
4. sentence_translations: 각 문장의 한국어 번역 (sentences와 정확히 같은 수, 같은 순서!)
5. key_sentences: 시험 출제 가능성이 높은 핵심 문장 8개 (원문 그대로)
6. test_a: vocab에서 뜻 쓰기 테스트용 5개 단어 (영어)
7. test_b: vocab에서 동의어 테스트용 5개 단어 (test_a와 겹치지 않게, 영어)
8. test_c: vocab에서 철자 테스트용 5개 (한국어 뜻)

JSON 형식:
{{
  "vocab": [{{"word":"...", "meaning":"...", "synonyms":"..."}}],
  "translation": "...",
  "sentences": ["...", "..."],
  "sentence_translations": ["첫째 문장 해석...", "둘째 문장 해석...", ...],
  "key_sentences": ["...", "..."],
  "test_a": ["...", "..."],
  "test_b": ["...", "..."],
  "test_c": ["...", "..."]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=4096)
    
    # 🔒 검증: API 문장 분리 대신 항상 regex 사용 (AI가 문장을 합치거나 쪼개는 것 방지)
    data["sentences"] = sentences_regex
    _safe_print(f"  Sentence count: {sent_count}")
    
    save_step(passage_dir, "step1_basic", data)
    return data

# ============================================================
# 순서 선지 코드 생성 유틸리티
# ============================================================
_CIRCLE_NUMS = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"]

def _generate_order_choices(data):
    """
    1) order_paragraphs (A)(B)(C) 라벨을 셔플 → 정답이 항상 ABC가 아니게
    2) order_choices 5지선다를 코드로 생성
    3) full_order_blocks 순서도 셔플
    """
    from itertools import permutations
    
    # === 1. 3단락 라벨 셔플 ===
    paras = data.get("order_paragraphs", [])
    if len(paras) == 3:
        # 현재: [[A, text1], [B, text2], [C, text3]] (원문 순서)
        # 원문 순서 기억 (인덱스 0,1,2 = 정답 순서)
        labels = ["A", "B", "C"]
        random.shuffle(labels)
        # 새 라벨 부여: 첫번째 단락 → labels[0], 두번째 → labels[1], ...
        new_paras = [[labels[i], paras[i][1]] for i in range(3)]
        # 정답 = labels를 원래 순서대로 읽은 것 (labels[0] → labels[1] → labels[2])
        correct = tuple(labels)  # 예: ("C", "A", "B") = 정답
        # 표시할 때는 라벨 알파벳 순으로 정렬
        new_paras.sort(key=lambda x: x[0])
        data["order_paragraphs"] = new_paras
    else:
        correct = ("A", "B", "C")
    
    # === 2. 선지 5개 생성 ===
    all_perms = list(permutations(["A", "B", "C"]))
    wrong = [p for p in all_perms if p != correct]
    selected_wrong = random.sample(wrong, 4)
    all_choices = [correct] + selected_wrong
    random.shuffle(all_choices)
    
    choices = []
    answer = ""
    for i, perm in enumerate(all_choices):
        text = f"({perm[0]})-({perm[1]})-({perm[2]})"
        choices.append(f"{_CIRCLE_NUMS[i]} {text}")
        if perm == correct:
            answer = f"{_CIRCLE_NUMS[i]} {text}"
    data["order_choices"] = choices
    data["order_answer"] = answer
    
    # === 3. 전체 문장 배열 (심화) 셔플 ===
    blocks = data.get("full_order_blocks", [])
    if len(blocks) >= 2:
        # 원문 순서 기억 (정답)
        original_labels = [b[0] for b in blocks]
        # 새 라벨 부여 + 셔플
        n = len(blocks)
        alpha = [chr(65+i) for i in range(n)]  # A, B, C, D, E, ...
        random.shuffle(alpha)
        # 각 원문 문장에 새 라벨
        new_blocks = [[alpha[i], blocks[i][1]] for i in range(n)]
        # 정답 순서 = alpha[0] → alpha[1] → ... (원문 순서대로 라벨 읽기)
        correct_order = "→".join([f"({alpha[i]})" for i in range(n)])
        data["full_order_answer"] = correct_order
        # 표시는 라벨 알파벳 순으로 정렬 (셔플 효과!)
        new_blocks.sort(key=lambda x: x[0])
        data["full_order_blocks"] = new_blocks

# ============================================================
# STEP 2: Lv.5 순서/삽입
# ============================================================
def step2_order(passage: str, sentences: list, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step2_order")
    if cached:
        _safe_print("  step2: using cache")
        return cached

    _safe_print("  step2: generating Lv.5 order...")
    prompt = f"""다음 영어 지문으로 순서 배열 + 문장 삽입 문제를 생성하세요.

[지문]
{passage}

[개별 문장]
{json.dumps(sentences, ensure_ascii=False)}

[생성 항목]
1. order_intro: 제시문 (첫 1~2문장)
2. order_paragraphs: (A)(B)(C) 3개 단락 (각각 label과 text). 정답 순서는 원문 순서대로.
   - 모든 문장이 빠짐없이 포함되어야 함
3. order_choices: 5지선다 (형식: "① (A)-(C)-(B)" 등). 정답 1개 포함.
4. order_answer: 정답 번호 (예: "④ (C)-(A)-(B)")
5. insert_sentence: 삽입할 문장 1개 (앞뒤 문맥 단서가 명확한 것)
6. insert_passage: 삽입 문장을 뺀 나머지 지문에 ( ① )~( ⑤ ) 위치 표시
7. insert_answer: 삽입 정답 번호
8. full_order_blocks: 전체 문장을 (A)~끝까지 개별 블록으로 분할 (각각 label, text)
9. full_order_answer: 정답 순서 (예: "(C)→(G)→(D)→...")

JSON 형식:
{{
  "order_intro": "...",
  "order_paragraphs": [{{"label":"A","text":"..."}}, ...],
  "order_choices": ["① ...", "② ...", ...],
  "order_answer": "...",
  "insert_sentence": "...",
  "insert_passage": "...",
  "insert_answer": "...",
  "full_order_blocks": [{{"label":"A","text":"..."}}, ...],
  "full_order_answer": "..."
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
    # 변환: order_paragraphs를 [label, text] 형태로
    if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
        data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]
    if data.get("full_order_blocks") and isinstance(data["full_order_blocks"][0], dict):
        data["full_order_blocks"] = [[b["label"], b["text"]] for b in data["full_order_blocks"]]

    # ★ 순서 선지를 코드로 직접 생성 (AI가 다양하게 안 만드는 문제 해결)
    _generate_order_choices(data)

    # 🔒 검증: 전체배열 블록 수 vs 원문 문장 수
    block_count = len(data.get("full_order_blocks", []))
    sentence_count = len(sentences)
    if block_count != sentence_count:
        _safe_print(f"  WARNING: sentence mismatch! original {sentence_count} vs generated {block_count}, retrying...")
        # 캐시 삭제 후 재시도 (1회)
        cache_path = passage_dir / "step2_order.json"
        if cache_path.exists():
            cache_path.unlink()
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
        if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
            data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]
        if data.get("full_order_blocks") and isinstance(data["full_order_blocks"][0], dict):
            data["full_order_blocks"] = [[b["label"], b["text"]] for b in data["full_order_blocks"]]
        block_count2 = len(data.get("full_order_blocks", []))
        if block_count2 != sentence_count:
            _safe_print(f"  WARNING: still mismatch ({block_count2} vs {sentence_count}), using original")
            data["full_order_blocks"] = [[chr(65+i), s] for i, s in enumerate(sentences)]

    save_step(passage_dir, "step2_order", data)
    return data

# ============================================================
# STEP 3: Stage 6 빈칸 추론
# ============================================================
def step3_blank(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step3_blank")
    if cached:
        _safe_print("  step3: using cache")
        return cached

    _safe_print("  step3: generating Lv.6 blanks...")
    prompt = f"""다음 영어 지문으로 빈칸 추론 문제를 생성하세요.

[지문]
{passage}

[규칙]
- 주제문(결론문)의 핵심 부분을 빈칸으로 만들기
- 빈칸은 15단어 이내로 (너무 긴 빈칸 금지)
- 빈칸 문장 외의 다른 문장은 원문 그대로 유지 (생략/축약/변형 절대 금지)
- 빈칸을 제외한 나머지 문장 부분도 절대 변형하지 말 것
- 선지 12개: 정답 6~7개 + 오답 5~6개
- 정답: 원문 핵심 표현을 동의어/비유적 표현으로 변형
- 오답: 지문 내용 왜곡, 반대 의미, 미언급 내용
- 각 선지는 15단어 이내로 간결하게

[JSON 형식]
{{
  "blank_passage": "빈칸이 포함된 전체 지문 (빈칸은 ____로 표시)",
  "blank_answer_korean": "빈칸 정답 내용 한국어",
  "blank_options": ["① ...", "② ...", ... "⑫ ..."],
  "blank_correct": ["②", "③", "⑤", ...],
  "blank_wrong": ["①", "④", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)
    save_step(passage_dir, "step3_blank", data)
    return data

# ============================================================
# STEP 4: Stage 7 주제 찾기
# ============================================================
def step4_topic(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step4_topic")
    if cached:
        _safe_print("  step4: using cache")
        return cached

    _safe_print("  step4: generating Lv.7 topic...")
    prompt = f"""다음 영어 지문으로 주제 찾기 문제를 생성하세요.

[지문]
{passage}

[규칙]
- 지문은 원문 그대로 (생략/변형 금지)
- 선지 12개: 정답 5개 + 오답 7개
- 선지는 반드시 영어로 작성 (한국어 금지)
- 정답: 주제문 키워드를 동의어로 치환한 영어 표현
- 오답: 지문 미언급, 부분적 내용, 왜곡 (영어)
- 추론적 사고 금지: 글에서 직접 언급된 내용만 정답
- 각 선지는 30단어 이내로 간결하게

[JSON 형식]
{{
  "topic_passage": "원문 전문 (그대로)",
  "topic_options": ["① the importance of...", "② how to...", ... "⑫ ..."],
  "topic_correct": ["②", "④", ...],
  "topic_wrong": ["①", "③", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)
    save_step(passage_dir, "step4_topic", data)
    return data

# ============================================================
# STEP 5: Lv.8 어법
# ============================================================
def step5_grammar(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step5_grammar")
    if cached:
        _safe_print("  step5: using cache")
        return cached

    sentences = split_sentences(passage)
    sent_count = len(sentences)
    error_count = min(8, sent_count)  # 문장 수보다 많은 오류 불가
    bracket_count = min(20, sent_count * 2)  # 문장당 최대 2개 괄호
    # bracket_count = sent_count * 2  # 문장당 최대 2개 괄호 / 예: 12문장이면 최대 24개 괄호 문제 + 24개 답안 박스가 생성됩니다.
    
    _safe_print("  step5: generating Lv.8 grammar...")
    prompt = f"""다음 영어 지문으로 어법 문제 2종류를 생성하세요.

[원문 - 총 {sent_count}개 문장]
{passage}

[⚠️ 가장 중요한 규칙]
1. 원문은 정확히 {sent_count}개 문장입니다
2. 출력 지문도 반드시 정확히 {sent_count}개 문장이어야 합니다
3. 절대 문장을 추가/삭제/분리/합치기 하지 마세요
4. 원문 문장에 괄호나 오류만 삽입하고, 나머지는 원문 그대로 유지
5. 문장 수가 부족하면 오류/괄호 수를 줄이세요 (문장 추가는 절대 금지!)

[어법 괄호형 Lv.8-1]
- 원문 {sent_count}개 문장 모두 포함 (출제 안 하는 문장도 원문 그대로)
- {bracket_count}개 괄호: (N)[정답 / 오답] 형태
- 한 문장에 여러 괄호 가능
- 출제: 시제, 대명사, 동명사, to부정사, 형용사/부사, 관계대명사, 분사, 사역동사 등

[어법 서술형 Lv.8-2]
- 원문 {sent_count}개 문장 모두 포함
- {error_count}개 문법 오류 삽입 (밑줄 없이)
- 한 문장에 최대 1개 오류
- 문장이 {sent_count}개뿐이므로 오류도 최대 {error_count}개만!

[JSON 형식]
{{
  "grammar_bracket_passage": "괄호 포함 전체 지문 (정확히 {sent_count}문장)",
  "grammar_bracket_count": {bracket_count},
  "grammar_bracket_answers": [{{"num":1, "answer":"go", "wrong":"will go", "reason":"if 조건절 현재시제"}}, ...],
  "grammar_error_passage": "오류 포함 전체 지문 (정확히 {sent_count}문장)",
  "grammar_error_count": {error_count},
  "grammar_error_answers": [{{"num":1, "original":"watch", "error":"watching", "reason":"tend to + 동사원형"}}, ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
    
    # 🔒 검증: 문장 수 체크
    for key in ['grammar_bracket_passage', 'grammar_error_passage']:
        gen_text = data.get(key, '')
        gen_sents = len(split_sentences(gen_text))
        if gen_sents > sent_count + 1:
            _safe_print(f"  WARNING: {key}: {gen_sents} sentences (original {sent_count}), retrying...")
            cache_path = passage_dir / "step5_grammar.json"
            if cache_path.exists():
                cache_path.unlink()
            data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
            break
    
    save_step(passage_dir, "step5_grammar", data)
    return data

# ============================================================
# STEP 6: Lv.9 어휘심화 + 내용일치
# ============================================================
def step6_vocab_content(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step6_vocab_content")
    if cached:
        _safe_print("  step6: using cache")
        return cached

    _safe_print("  step6: generating Lv.9 vocab...")
    prompt = f"""다음 영어 지문으로 어휘 심화 + 내용 일치 문제를 생성하세요.

[지문]
{passage}

[Lv.9-1 Part A 규칙]
- 원문의 모든 문장을 빠짐없이 포함
- 7~9개 괄호: (N)[정답 / 반의어] 형태
- 정답과 오답은 의미가 반대인 단어 쌍으로 구성 (예: regarded/overlooked, effective/futile, mild/severe, constant/intermittent)
- 발음 유사 단어 절대 금지. 반드시 반의어로 출제
- 문맥을 읽어야 정답을 고를 수 있는 수능 수준 반의어 쌍

[Lv.9-1 Part B 규칙]
- 10개 단어 (최소 8개, 가능하면 10개), 각 5개 선택지
- 5개 중 동의어 2개 + 반의어 3개로 구성
- "모두 고르시오" 형태: 동의어만 골라야 정답
- 동의어: 수능 수준의 정확한 유의어
- 반의어: 해당 단어와 의미가 반대인 단어 3개
- 발음/철자 유사 단어 절대 금지. 의미 기반으로만 출제

[내용 일치 규칙]
- content_match_kr: 반드시 10개 한국어 선지 (①~⑩). 일치 3~5개 + 불일치 5~7개
- content_match_en: 반드시 10개 영어 선지 (①~⑩). 일치 3~5개 + 불일치 5~7개
- 한국어와 영어 선지의 순서는 서로 다르게 랜덤 배치

[JSON 형식]
{{
  "vocab_advanced_passage": "괄호 포함 지문",
  "vocab_parta_answers": [{{"num":1, "answer":"regarded", "wrong":"overlooked", "reason":"~로 여겨지다 vs 간과하다"}}, ...],
  "vocab_partb": [{{"word":"regarded", "choices":"considered / perceived / overlooked / neglected / dismissed"}}, ...],
  "vocab_partb_answers": [{{"num":1, "correct":["considered", "perceived"], "wrong":["overlooked", "neglected", "dismissed"]}}, ...],
  "content_match_kr": ["① 평소 극장에서 혼자 영화를 본다.", ... (반드시 10개 선지)],
  "content_match_kr_answer": ["②", "③", ...],
  "content_match_en": ["① The writer normally watches movies alone.", ... (반드시 10개 선지)],
  "content_match_en_answer": ["②", "④", ...]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=4000)

    # 한국어 선지 셔플
    kr_items = data.get("content_match_kr", [])
    kr_answers = set(data.get("content_match_kr_answer", []))
    if kr_items:
        kr_texts = [re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', item) for item in kr_items]
        kr_correct = [_CIRCLE_NUMS[i] in kr_answers for i in range(len(kr_texts))]
        kr_pairs = list(zip(kr_texts, kr_correct))
        random.shuffle(kr_pairs)
        data["content_match_kr"] = [f"{_CIRCLE_NUMS[i]} {kr_pairs[i][0]}" for i in range(len(kr_pairs))]
        data["content_match_kr_answer"] = [_CIRCLE_NUMS[i] for i in range(len(kr_pairs)) if kr_pairs[i][1]]

    # Part B 영어 선지 셔플 (번호는 오름차순 유지, 문장만 랜덤)
    en_items = data.get("content_match_en", [])
    en_answers = set(data.get("content_match_en_answer", []))
    if en_items:
        # 번호와 문장 분리
        texts = [re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', item) for item in en_items]
        is_correct = [_CIRCLE_NUMS[i] in en_answers for i in range(len(texts))]
        # 문장+정답 쌍을 셔플
        pairs = list(zip(texts, is_correct))
        random.shuffle(pairs)
        # 번호 재부여 + 정답 갱신
        data["content_match_en"] = [f"{_CIRCLE_NUMS[i]} {pairs[i][0]}" for i in range(len(pairs))]
        data["content_match_en_answer"] = [_CIRCLE_NUMS[i] for i in range(len(pairs)) if pairs[i][1]]

    save_step(passage_dir, "step6_vocab_content", data)
    return data

# ============================================================
# STEP 7: Stage 10 영작 (API 불필요 - 프로그래밍으로 처리)
# ============================================================
def step7_writing(sentences: list, translation: str, passage_dir: Path, sentence_translations: list = None) -> dict:
    cached = load_step(passage_dir, "step7_writing")
    if cached:
        _safe_print("  step7: using cache")
        return cached

    _safe_print("  step7: generating Lv.10 writing...")
    # 한국어 문장: sentence_translations 우선, 없으면 translation 분리
    if sentence_translations and len(sentence_translations) >= len(sentences):
        kr_sentences = sentence_translations
    else:
        kr_sentences = [s.strip() for s in re.split(r'(?<=[.!?다요음임])\s+', translation) if s.strip()]

    writing_items = []
    for i, eng in enumerate(sentences):
        words = eng.split()
        # 대문자→소문자 변환 (첫 단어, I/고유명사 제외)
        processed = []
        for j, w in enumerate(words):
            if j == 0 and w[0].isupper() and w not in ['I', 'I,']:
                # 고유명사 체크 (간단히: 2글자 이상 대문자 시작)
                if not (len(w) > 1 and w[1:].islower() and w[0].isupper() and any(c.isupper() for c in w)):
                    w = w[0].lower() + w[1:]
            processed.append(w)
        # 마침표/느낌표/물음표 제거
        last = processed[-1]
        if last.endswith(('.', '!', '?')):
            processed[-1] = last[:-1]
        # 셔플
        shuffled = processed.copy()
        random.shuffle(shuffled)
        scrambled = ' / '.join(shuffled)

        kr = kr_sentences[i] if i < len(kr_sentences) else f"문장 {i+1}"
        writing_items.append({
            "korean": kr,
            "scrambled": scrambled,
            "answer": eng
        })

    data = {"writing_items": writing_items}
    save_step(passage_dir, "step7_writing", data)
    return data

# ============================================================
# STEP 8: 정답 생성
# ============================================================
def step8_answers(all_data: dict, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step8_answers")
    if cached:
        _safe_print("  step8: using cache")
        return cached

    _safe_print("  step8: generating answer page...")
    # 정답 HTML 생성 (레벨별 블록화)
    blocks = []

    # Lv.1
    blocks.append('<div class="ablock"><p class="ast">Stage 1 어휘 테스트</p>'
                   '<p>A. (어휘 테스트 정답은 학생이 직접 확인)</p></div>')

    # Lv.5
    s2 = all_data.get("step2", {})
    blocks.append(f'<div class="ablock"><p class="ast">Stage 5 순서 배열</p>'
                   f'<p>정답: {s2.get("order_answer","")}</p>'
                   f'<p>삽입 정답: {s2.get("insert_answer","")}</p>'
                   f'<p>전체 배열: {s2.get("full_order_answer","")}</p></div>')

    # Lv.6
    s3 = all_data.get("step3", {})
    correct = ', '.join(s3.get("blank_correct", []))
    blocks.append(f'<div class="ablock"><p class="ast">Stage 6 빈칸 추론</p>'
                   f'<p>정답: {correct}</p></div>')

    # Lv.7
    s4 = all_data.get("step4", {})
    correct = ', '.join(s4.get("topic_correct", []))
    blocks.append(f'<div class="ablock"><p class="ast">Stage 7 주제 찾기</p>'
                   f'<p>정답: {correct}</p></div>')

    # Lv.8 괄호
    s5 = all_data.get("step5", {})
    lv8_bracket = ['<div class="ablock"><p class="ast">Stage 8 어법 (괄호)</p>']
    for a in s5.get("grammar_bracket_answers", []):
        if isinstance(a, dict):
            lv8_bracket.append(f'<p>({a.get("num","")}) {a.get("answer","")}</p>')
    lv8_bracket.append('</div>')
    blocks.append(''.join(lv8_bracket))

    # Stage 8 서술형
    lv8_error = ['<div class="ablock"><p class="ast">Stage 8 서술형</p>']
    for a in s5.get("grammar_error_answers", []):
        if isinstance(a, dict):
            lv8_error.append(f'<p>{a.get("error","")}->{a.get("original","")}({a.get("reason","")})</p>')
    lv8_error.append('</div>')
    blocks.append(''.join(lv8_error))

    # Lv.9-1 Part A
    s6 = all_data.get("step6", {})
    lv9a = ['<div class="ablock"><p class="ast">Stage 9-1 어휘 Part A</p>']
    for a in s6.get("vocab_parta_answers", []):
        if isinstance(a, dict):
            lv9a.append(f'<p>({a.get("num","")}) {a.get("answer","")}</p>')
    lv9a.append('</div>')
    blocks.append(''.join(lv9a))

    # Lv.9-1 Part B + Lv.9-2
    lv9b = ['<div class="ablock"><p class="ast">Stage 9-1 어휘 Part B</p>']
    for a in s6.get("vocab_partb_answers", []):
        if isinstance(a, dict):
            correct_list = ', '.join(a.get("correct", []))
            lv9b.append(f'<p>{a.get("num","")}: {correct_list}</p>')
    kr_ans = ', '.join(s6.get("content_match_kr_answer", []))
    en_ans = ', '.join(s6.get("content_match_en_answer", []))
    lv9b.append(f'<p class="ast">Lv.9-2 내용일치</p>')
    lv9b.append(f'<p>한국어: {kr_ans}</p>')
    lv9b.append(f'<p>영어: {en_ans}</p>')
    lv9b.append('</div>')
    blocks.append(''.join(lv9b))

    # Lv.10
    s7 = all_data.get("step7", {})
    lv10 = ['<div class="ablock"><p class="ast">Stage 10 영작</p>']
    for idx, item in enumerate(s7.get("writing_items", []), start=1):
        lv10.append(f'<p>{idx}. {item.get("answer","")}</p>')
    lv10.append('</div>')
    blocks.append(''.join(lv10))

    answers_html = '\n'.join(blocks)
    data = {"answers_html": answers_html}
    save_step(passage_dir, "step8_answers", data)
    return data

# ============================================================
# 전체 데이터 → 템플릿 변수로 변환
# ============================================================
def merge_to_template_data(passage: str, meta: dict, all_steps: dict) -> dict:
    """모든 단계 결과를 템플릿 변수로 병합"""
    s1 = all_steps["step1"]
    s2 = all_steps["step2"]
    s3 = all_steps["step3"]
    s4 = all_steps["step4"]
    s5 = all_steps["step5"]
    s6 = all_steps["step6"]
    s7 = all_steps["step7"]
    s8 = all_steps["step8"]

    return {
        # 메타 정보
        "subject": meta.get("subject", ""),
        "publisher": meta.get("publisher", ""),
        "lesson_num": meta.get("lesson_num", ""),
        "lesson_n": meta.get("lesson_n", ""),
        "challenge_title": meta.get("challenge_title", ""),
        # 지문/번역
        "passage": passage,
        "translation": s1.get("translation", ""),
        # Lv.1 어휘
        "vocab": s1.get("vocab", []),
        "test_a": s1.get("test_a", []),
        "test_b": s1.get("test_b", []),
        "test_c": s1.get("test_c", []),
        # Lv.3 문장분석 (전체 문장) + 핵심문장
        "sentences": s1.get("sentences", []),
        "key_sentences": s1.get("key_sentences", []),
        # Lv.5 순서/삽입
        "order_intro": s2.get("order_intro", ""),
        "order_paragraphs": s2.get("order_paragraphs", []),
        "order_choices": s2.get("order_choices", []),
        "insert_sentence": s2.get("insert_sentence", ""),
        "insert_passage": s2.get("insert_passage", ""),
        "full_order_blocks": s2.get("full_order_blocks", []),
        # Lv.6 빈칸
        "blank_passage": s3.get("blank_passage", ""),
        "blank_options": s3.get("blank_options", []),
        # Lv.7 주제
        "topic_passage": s4.get("topic_passage", ""),
        "topic_options": s4.get("topic_options", []),
        # Lv.8 어법
        "grammar_bracket_passage": s5.get("grammar_bracket_passage", ""),
        "grammar_bracket_count": s5.get("grammar_bracket_count", 13),
        "grammar_error_passage": s5.get("grammar_error_passage", ""),
        "grammar_error_count": s5.get("grammar_error_count", 8),
        # Lv.9
        "vocab_advanced_passage": s6.get("vocab_advanced_passage", ""),
        "vocab_parta_answers": s6.get("vocab_parta_answers", []),
        "vocab_partb": s6.get("vocab_partb", []),
        "content_match_kr": s6.get("content_match_kr", []),
        "content_match_en": s6.get("content_match_en", []),
        # Stage 10 영작
        "writing_items": s7.get("writing_items", []),
        # 정답
        "answers_html": s8.get("answers_html", ""),
    }

# ============================================================
# PDF 렌더링
# ============================================================
def _unique_path(directory: Path, base_name: str, ext: str) -> Path:
    """같은 이름 파일이 있으면 _v2, _v3 등 붙여서 고유 경로 반환"""
    path = directory / f"{base_name}{ext}"
    if not path.exists() and not path.with_suffix('.html').exists():
        return path
    v = 2
    while True:
        path = directory / f"{base_name}_v{v}{ext}"
        if not path.exists() and not path.with_suffix('.html').exists():
            return path
        v += 1

def render_pdf(template_data: dict, output_path: Path, levels=None):
    """Jinja2 → HTML 저장 (크롬에서 PDF 인쇄)"""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    tmpl = env.get_template("template.html")
    template_data["levels"] = levels  # None이면 전체 출력
    html = tmpl.render(**template_data)

    # WeasyPrint 시도, 없으면 HTML로 저장
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(output_path))
        _safe_print(f"  PDF created: {output_path.name}")
    except (ImportError, OSError):
        html_path = output_path.with_suffix('.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        _safe_print(f"  HTML created: {html_path.name}")
        _safe_print(f"  Open in Chrome -> Ctrl+P -> Save as PDF")

# ============================================================
# 메인: 단일 지문 처리
# ============================================================
def process_passage(passage: str, meta: dict, passage_id: str, force=False, levels=None):
    """지문 1개 → 전체 워크북 생성"""
    passage_dir = DATA_DIR / passage_id
    if force:
        import shutil
        if passage_dir.exists():
            shutil.rmtree(passage_dir)

    _safe_print(f"\n{'='*50}")
    _safe_print(f"Processing: {passage_id} ({meta.get('challenge_title','')})")
    _safe_print(f"{'='*50}")

    sentences = split_sentences(passage)
    all_steps = {}

    # Step 1: 기본 분석
    all_steps["step1"] = step1_basic_analysis(passage, passage_dir)
    sentences_from_api = all_steps["step1"].get("sentences", sentences)

    # Step 2: Lv.5 순서/삽입
    all_steps["step2"] = step2_order(passage, sentences_from_api, passage_dir)

    # Step 3: Lv.6 빈칸
    all_steps["step3"] = step3_blank(passage, passage_dir)

    # Step 4: Lv.7 주제
    all_steps["step4"] = step4_topic(passage, passage_dir)

    # Step 5: Lv.8 어법
    all_steps["step5"] = step5_grammar(passage, passage_dir)

    # Step 6: Lv.9 어휘+내용일치
    all_steps["step6"] = step6_vocab_content(passage, passage_dir)

    # Step 7: Stage 10 영작 (로컬)
    translation = all_steps["step1"].get("translation", "")
    sentence_translations = all_steps["step1"].get("sentence_translations", [])
    all_steps["step7"] = step7_writing(sentences_from_api, translation, passage_dir, sentence_translations)

    # Step 8: 정답
    all_steps["step8"] = step8_answers(all_steps, passage_dir)

    # 병합 + PDF
    template_data = merge_to_template_data(passage, meta, all_steps)

    # 🔒 콘텐츠 길이 검증 (페이지 밀림 방지)
    warnings = []
    bp = template_data.get("blank_passage", "")
    if len(bp) > 1200:
        warnings.append(f"blank_passage 길이 {len(bp)} (권장 1200 이내)")
    gp = template_data.get("grammar_bracket_passage", "")
    if len(gp) > 1600:
        warnings.append(f"grammar_bracket_passage 길이 {len(gp)} (권장 1600 이내)")
    gep = template_data.get("grammar_error_passage", "")
    if len(gep) > 1200:
        warnings.append(f"grammar_error_passage 길이 {len(gep)} (권장 1200 이내)")
    if warnings:
        _safe_print(f"  WARNING: content length warning:")
        for w in warnings:
            _safe_print(f"     - {w}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{meta.get('lesson_num','')}과_{meta.get('challenge_title','워크북')}_워크북"
    pdf_path = _unique_path(OUTPUT_DIR, base_name, ".pdf")
    render_pdf(template_data, pdf_path, levels=levels)

    _safe_print(f"Done: {pdf_path.name}")
    return pdf_path

# ============================================================
# 배치 처리: 여러 지문
# ============================================================
def process_batch(passages: list[dict], levels=None):
    """여러 지문을 순차 처리
    passages: [{"passage": "...", "meta": {...}, "id": "01"}, ...]
    """
    results = []
    total = len(passages)
    for i, item in enumerate(passages):
        _safe_print(f"\n[{i+1}/{total}] Processing...")
        try:
            pdf = process_passage(item["passage"], item["meta"], item["id"], levels=levels)
            results.append({"id": item["id"], "status": "done", "pdf": str(pdf)})
        except Exception as e:
            _safe_print(f"FAILED: {item['id']} - {e}")
            results.append({"id": item["id"], "status": "error", "error": str(e)})

    # 결과 요약
    _safe_print(f"\n{'='*50}")
    _safe_print(" Results summary")
    done = sum(1 for r in results if r["status"] == "done")
    err = sum(1 for r in results if r["status"] == "error")
    _safe_print(f"  Success: {done}/{total}")
    if err:
        _safe_print(f"  Failed: {err}/{total}")
        for r in results:
            if r["status"] == "error":
                _safe_print(f"     - {r['id']}: {r['error']}")
    return results


# ============================================================
# 단일 파일에서 여러 지문 자동 분리 + 실행
# ============================================================
def split_and_run(filepath: str, lesson_num: str = "5", levels=None):
    """
    ###제목### 구분자로 나뉜 단일 파일에서 지문 추출 → 순차 실행
    
    파일 형식:
        ###05강 01번###
        지문 영어 텍스트...
        
        ###05강 02번###
        지문 영어 텍스트...
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ###...### 패턴으로 분리
    parts = re.split(r'###(.+?)###', content)
    # parts = ['', '05강 01번', '지문내용', '05강 02번', '지문내용', ...]
    
    passages = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        text = parts[i+1].strip() if i+1 < len(parts) else ""
        if text:
            # 제목에서 과번호 + 번호 추출
            lesson_match = re.search(r'(\d+)강', title)
            num_match = re.search(r'(\d+)번', title)
            lnum = lesson_match.group(1) if lesson_match else lesson_num
            pid = num_match.group(1).zfill(2) if num_match else str(len(passages)+1).zfill(2)
            passages.append({
                "id": f"{lnum}_{pid}",
                "passage": text,
                "meta": {
                    "subject": "수특 영어", "publisher": "EBS",
                    "lesson_num": lnum, "lesson_n": lnum,
                    "challenge_title": title
                }
            })
    
    if not passages:
        _safe_print("ERROR: No passages found. Check ### format.")
        return
    
    _safe_print(f"Found {len(passages)} passages")
    for p in passages:
        _safe_print(f"  - {p['meta']['challenge_title']}")
    print()
    
    process_batch(passages, levels=levels)

    # 자동으로 HTML 합치기
    merge_html_files()


# ============================================================
# HTML 합치기 (여러 워크북 → 하나의 HTML)
# ============================================================
def merge_html_files(output_dir=None):
    """output 폴더의 모든 HTML을 하나로 합침 (합본 파일명 자동 생성)"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    html_files = sorted([f for f in output_dir.glob("*워크북.html") if "합본" not in f.name])
    if len(html_files) < 2:
        return
    
    _safe_print(f"\nMerging {len(html_files)} HTML files...")
    
    # 파일명에서 제목 추출하여 합본명 생성
    import re as _re
    titles = []
    for hf in html_files:
        # "3과_03강_02번_워크북.html" → "03강 02번"
        m = _re.search(r'(\d+강)[_ ](\w+)', hf.stem.replace('_워크북',''))
        if m:
            titles.append(f"{m.group(1)} {m.group(2)}")
    
    if titles:
        first = titles[0]   # 예: "03강 01번"
        last = titles[-1]   # 예: "04강 04번"
        merge_name = f"{first}~{last} 합본.html"
    else:
        merge_name = "전체_워크북_합본.html"
    
    # 첫 파일에서 CSS 추출
    first_html = html_files[0].read_text(encoding='utf-8')
    style_match = _re.search(r'<style[^>]*>(.*?)</style>', first_html, _re.DOTALL)
    css = style_match.group(1) if style_match else ""
    
    # 각 파일에서 <body> 내용만 추출
    all_bodies = []
    for hf in html_files:
        html = hf.read_text(encoding='utf-8')
        body_match = _re.search(r'<body[^>]*>(.*?)</body>', html, _re.DOTALL)
        if body_match:
            all_bodies.append(body_match.group(1))
    
    # 합친 HTML 생성
    merged_path = _unique_path(output_dir, merge_name.replace('.html', ''), '.html')
    merged = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{merged_path.stem}</title>
<style>
{css}
</style>
</head>
<body>
{''.join(all_bodies)}
</body>
</html>"""
    
    merged_path.write_text(merged, encoding='utf-8')
    _safe_print(f"  Merged: {merged_path.name}")
    _safe_print(f"  Open in Chrome -> Ctrl+P -> Save as PDF")
if __name__ == "__main__":
    # --level 파싱 (어디서든 사용 가능)
    levels = None
    filtered_args = []
    for i, arg in enumerate(sys.argv):
        if arg == "--level" and i+1 < len(sys.argv):
            levels = [int(x) for x in sys.argv[i+1].split(",")]
            continue
        if i > 0 and sys.argv[i-1] == "--level":
            continue
        filtered_args.append(arg)
    sys.argv = filtered_args

    if len(sys.argv) < 2:
        print("Usage:")
        _safe_print("  Multiple: py pipeline.py --all all.txt")
        _safe_print("  Levels: py pipeline.py --all all.txt --level 1,2,5,8")
        _safe_print("  Single: py pipeline.py passage.txt 5 \"05-01\"")
        _safe_print("  Merge: py pipeline.py --merge")
        print()
        _safe_print("  --level option: select levels (0=cover+answers)")
        _safe_print("    e.g.) --level 1,2,3,4")
        _safe_print("    e.g.) --level 5,6,7,8")
        _safe_print("    e.g.) --level 0,1,2")
        sys.exit(1)

    if levels:
        _safe_print(f"Level filter: Lv.{','.join(str(l) for l in levels)}")

    if sys.argv[1] == "--merge":
        merge_html_files()
    elif sys.argv[1] == "--all":
        filepath = sys.argv[2]
        lesson = sys.argv[3] if len(sys.argv) > 3 else "5"
        split_and_run(filepath, lesson, levels=levels)
    elif sys.argv[1] == "--batch":
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            batch = json.load(f)
        process_batch(batch, levels=levels)
    else:
        passage_file = sys.argv[1]
        with open(passage_file, 'r', encoding='utf-8') as f:
            passage = f.read().strip()
        lesson_num = sys.argv[2] if len(sys.argv) > 2 else "1"
        title = sys.argv[3] if len(sys.argv) > 3 else Path(passage_file).stem
        meta = {
            "subject": "수특 영어", "publisher": "EBS",
            "lesson_num": lesson_num, "lesson_n": lesson_num,
            "challenge_title": title
        }
        process_passage(passage, meta, f"passage_{lesson_num}_{title}", levels=levels)
