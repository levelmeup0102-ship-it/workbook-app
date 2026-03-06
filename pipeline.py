#!/usr/bin/env python3
"""Workbook generation pipeline"""
PIPELINE_VERSION = "v9-curl-final"
import asyncio, json, os, sys, time, random, re, math, logging

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("pipeline")
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
    """영어 지문 문장 분리
    핵심 규칙:
    - "word." Only → 따옴표 닫힌 뒤 새 대문자 → 분리
    - "text. More text" → 따옴표 안 마침표+대문자 → 분리 안함
    - "text? More?" She → 따옴표 안 물음표 → 분리 안함
    """
    protected = text

    # 1단계: 따옴표 안의 내부 문장경계([.!?] + 공백 + 대문자)만 토큰으로 보호
    def protect_quote_internals(match):
        inner = match.group(1)
        open_q = match.group(0)[0]
        close_q = match.group(0)[-1]
        protected_inner = re.sub(
            r'([.!?])\s+([A-Z])',
            lambda m2: f"{m2.group(1)}§QSEP§{m2.group(2)}",
            inner
        )
        return open_q + protected_inner + close_q

    protected = re.sub(r'["“](.*?)["”]', protect_quote_internals, protected, flags=re.DOTALL)

    # 2단계: 약어 마침표 보호
    abbrevs = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Jr.', 'Sr.', 'St.',
        'vs.', 'etc.', 'No.', 'Vol.', 'Fig.', 'Gen.', 'Gov.', 'Rev.',
        'Sgt.', 'Cpl.', 'Lt.', 'Co.', 'Inc.', 'Ltd.', 'Corp.', 'Dept.',
        'Est.', 'al.', 'e.g.', 'i.e.', 'U.S.', 'U.K.', 'U.N.',
    ]
    replacements = {}
    for ab in abbrevs:
        token = ab.replace('.', '§DOT§')
        pattern = r'(?<!\w)' + re.escape(ab)
        if re.search(pattern, protected):
            replacements[token] = ab
            protected = re.sub(pattern, token, protected)

    # 2.5단계: 1글자 이니셜 마침표 보호 (G. W. Bush, J. K. Rowling 등)
    def protect_initial(m):
        return m.group(0).replace('.', '§DOT§')
    protected = re.sub(r'(?<!\w)([A-Z])\.\s*(?=[A-Z][\.\s]|[A-Z][a-z])', protect_initial, protected)

    # 3단계: 문장 분리
    sentences = [s.strip() for s in re.split(
        r'(?<=[.!?])\s+(?=[\u201c\u201d\u0022]?[A-Z])|(?<=[.!?][\u201c\u201d\u0022])\s+(?=[\u201c\u201d\u0022]?[A-Z])',
        protected
    ) if s.strip()]

    # 4단계: 토큰 복원
    restored = []
    for s in sentences:
        for token, original in replacements.items():
            s = s.replace(token, original)
        s = s.replace('§DOT§', '.')
        s = s.replace('§QSEP§', ' ')
        restored.append(s)

    return restored


def _is_dialogue(sentences: list) -> bool:
    """대화문 지문인지 판별: 문장의 20% 이상이 '이름:' 패턴으로 시작하면 대화문"""
    if len(sentences) < 3:
        return False
    speaker_count = sum(1 for s in sentences if re.match(r'^[A-Z][a-z]+\s*:', s))
    return speaker_count / len(sentences) >= 0.2


def _merge_short_dialogue(sentences: list, min_words: int = 6) -> list:
    """대화문에서 짧은 문장(6단어 이하) 합치기
    규칙:
    - 같은 화자의 다음 문장과 합침
    - 화자가 바뀌면 합치지 않음
    - 대화문이 아니면 원본 그대로 반환
    """
    if not _is_dialogue(sentences) or len(sentences) < 2:
        return sentences

    _safe_print(f"  대화문 감지 → 짧은 문장 병합 (≤{min_words}단어)")

    # 각 문장의 화자 추적
    def _get_speaker(sent):
        m = re.match(r'^([A-Z][a-z]+)\s*:', sent)
        return m.group(1) if m else None

    speakers = []
    current_sp = None
    for s in sentences:
        sp = _get_speaker(s)
        if sp:
            current_sp = sp
        speakers.append(current_sp)

    # 1차: 짧은 문장 → 같은 화자의 다음 문장과 합침
    merged = []
    merged_sp = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        sp = speakers[i]
        while len(current.split()) <= min_words and i + 1 < len(sentences):
            if speakers[i + 1] != sp:
                break
            i += 1
            current = current + " " + sentences[i]
        merged.append(current)
        merged_sp.append(sp)
        i += 1

    # 2차: 여전히 짧은 문장 → 같은 화자의 앞 문장과 합침
    final = [merged[0]]
    final_sp = [merged_sp[0]]
    for i in range(1, len(merged)):
        current = merged[i]
        if len(current.split()) <= min_words and final_sp[-1] == merged_sp[i]:
            # 현재 문장이 짧으면 앞 문장에 붙임
            final[-1] = final[-1] + " " + current
        else:
            final.append(current)
            final_sp.append(merged_sp[i])

    # 마지막이 짧으면 앞과 합침 (같은 화자만)
    if len(final) >= 2 and len(final[-1].split()) <= min_words:
        if final_sp[-1] == final_sp[-2]:
            final[-2] = final[-2] + " " + final[-1]
            final.pop()
            final_sp.pop()

    _safe_print(f"  병합 결과: {len(sentences)}문장 → {len(final)}문장 (-{len(sentences)-len(final)})")
    return final

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
    sentences_regex = _merge_short_dialogue(sentences_regex)
    sent_count = len(sentences_regex)

    _safe_print("  step1: basic analysis...")
    # 대화문 병합된 문장 리스트를 API에 명시적으로 전달
    numbered_sentences = "\n".join([f"[문장{i+1}] {s}" for i, s in enumerate(sentences_regex)])
    prompt = f"""다음 영어 지문을 분석하여 JSON을 생성하세요.

[지문 - 총 {sent_count}개 문장]
{passage}

[문장 분리 기준 - 반드시 이 기준을 따르세요!]
{numbered_sentences}

[생성 항목]
1. vocab: 핵심 어휘 14개 (각각 word, meaning(한국어), synonyms(영어 유의어 4개 쉼표구분))
2. translation: 지문 전체의 자연스러운 한국어 번역
   - 자연스럽고 읽기 좋은 한국어로 번역 (의역 OK)
   - 단, 주어와 핵심 동사는 정확히 반영 (met→만났다, said→말했다 등 동사 혼동 방지)
   - 한국어 화법: 나는 "~~~"라고 말했다 (O) / 나는 말했다, "~~~" (X)
   - 대화체는 자연스러운 존댓말/반말 사용
3. sentences: 위 [문장 분리 기준]과 정확히 동일하게 배열 (정확히 {sent_count}개!)
   - 위에서 제공한 문장 분리를 그대로 사용하세요
4. sentence_translations: 위 [문장 분리 기준]의 각 문장에 대한 한국어 번역 (정확히 {sent_count}개, 같은 순서!)
   - ⚠ 영어 1문장 = 한국어 1문장! 영어가 긴 문장이어도 한국어 번역을 절대 2개로 나누지 마세요!
   - 한국어 번역 중간에 마침표(.)를 찍어 문장을 나누면 안 됩니다. 쉼표(,)로 이어주세요.
   - 자연스럽고 읽기 좋은 한국어 (의역 OK, 어색한 직역 금지)
   - 단, 주어와 핵심 동사만 정확히 (met→만났다, said→말했다, prepared→준비하다 등)
   - to부정사를 동사처럼 해석하지 말 것 (학생이 영작 시 진짜 동사를 찾을 수 있게)
   - 한국어 화법: 나는 "~~~"라고 말했다 형태로 (나는 말했다, "~~~" 금지)
   - 따옴표(" " 또는 " ") 안의 마침표는 문장의 끝으로 처리하지 말 것
   - 따옴표가 열렸으면 반드시 닫힌 후에야 다음 문장으로 넘어감
   - 예: "연설 때문에 부끄럽습니다."라고 말했다. → 이것은 하나의 번역!
   - 영어 sentences 배열 수와 반드시 일치해야 함 (정확히 {sent_count}개!)
5. key_sentences: 시험 출제 가능성이 높은 핵심 문장 8개 (원문 그대로)
6. test_a: vocab에서 뜻 쓰기 테스트용 5개 단어 (영어)
7. test_b: vocab에서 유의어 테스트용 5개 단어 (test_a와 겹치지 않게, 영어)
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

    # ★ sentence_translations: API 결과 개수 검증 → 불일치시 개별 번역 API 호출
    # 한국어 코드 분리는 불완전하므로, 개수 불일치시 API에 문장별 번역을 다시 요청
    st = data.get("sentence_translations", [])
    
    # 지문에 따옴표가 포함되어 있으면 무조건 재요청 (잘못 분리 위험 높음)
    passage_has_quotes = any('"' in s or '\u201c' in s or '\u201d' in s for s in sentences_regex)
    if len(st) != sent_count or (passage_has_quotes and len(st) == sent_count):
        reason = f"{len(st)}개 ≠ {sent_count}개" if len(st) != sent_count else "지문에 따옴표 포함 → 안전 재요청"
        _safe_print(f"  WARNING: sentence_translations {reason} → 문장별 번역 재요청")
        # 영어 문장 리스트를 넘겨서 1:1 번역 요청
        numbered_sents = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences_regex)])
        retry_prompt = f"""다음 영어 문장들을 각각 한국어로 번역하세요.

[필수 규칙]
- 반드시 정확히 {sent_count}개의 번역을 배열로 반환
- ⚠ 영어 1문장 = 한국어 1번역! 긴 문장이어도 한국어를 2개로 나누지 마세요!
- 한국어 번역 중간에 마침표를 찍어 문장을 분리하지 마세요. 쉼표로 이어주세요.
- 따옴표 안의 내용도 하나의 문장에 포함 (절대 분리하지 말 것)
- 자연스럽고 읽기 좋은 한국어로 번역 (의역 OK, 어색한 직역 금지)
- 단, 주어와 핵심 동사만 정확히 반영 (met→만났다, said→말했다 등)
- to부정사를 동사처럼 해석하지 말 것 (학생이 영작 시 진짜 동사를 찾을 수 있게)
- 한국어 화법: 나는 "~~~"라고 말했다 (O) / 나는 말했다, "~~~" (X)

[영어 문장 - 총 {sent_count}개]
{numbered_sents}

JSON 형식:
{{"translations": ["1번 문장 번역", "2번 문장 번역", ...]}}"""
        try:
            retry_data = call_claude_json(SYS_JSON_KR, retry_prompt, max_tokens=3000)
            retry_st = retry_data.get("translations", [])
            if len(retry_st) == sent_count:
                st = retry_st
                _safe_print(f"  ✅ 문장별 번역 성공: {len(st)}개")
            else:
                _safe_print(f"  ⚠ 문장별 번역도 {len(retry_st)}개, 원본 사용 후 보정")
        except Exception as e:
            _safe_print(f"  ⚠ 문장별 번역 실패: {str(e)[:80]}")
        
        # 최종 보정
        while len(st) < sent_count:
            st.append(f"문장 {len(st)+1}")
        data["sentence_translations"] = st[:sent_count]
    else:
        data["sentence_translations"] = st

    _safe_print(f"  Sentence count: {sent_count}")
    
    save_step(passage_dir, "step1_basic", data)
    return data

# ============================================================
# 순서 선지 코드 생성 유틸리티
# ============================================================
_CIRCLE_NUMS = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"]

def _generate_order_choices(data, passage=""):
    """
    1) order_paragraphs의 각 단락이 원문에서 어떤 순서인지 확인
    2) 라벨 셔플 → 정답이 항상 ABC가 아니게
    3) order_choices 5지선다를 코드로 생성
    4) full_order_blocks 순서도 셔플
    """
    from itertools import permutations
    
    # === 1. 3단락의 원문 순서 파악 ===
    paras = data.get("order_paragraphs", [])
    if len(paras) == 3:
        # 각 단락 텍스트가 원문에서 어디에 있는지 위치로 정렬
        def _find_pos(text):
            # 단락 텍스트의 첫 30자로 원문에서 위치 찾기
            snippet = re.sub(r'\s+', ' ', text.strip())[:50]
            pos = passage.find(snippet[:30])
            if pos == -1:
                # 첫 단어 몇 개로 재시도
                words = snippet.split()[:5]
                search = ' '.join(words)
                pos = passage.find(search)
            return pos if pos >= 0 else 999999
        
        # 원문 순서대로 정렬 (위치 기반)
        indexed = [(i, _find_pos(paras[i][1])) for i in range(3)]
        indexed.sort(key=lambda x: x[1])
        original_order = [idx for idx, pos in indexed]  # 원문 순서의 인덱스
        
        # 라벨 셔플: 정답이 ABC가 되지 않도록
        labels = ["A", "B", "C"]
        for _ in range(10):
            random.shuffle(labels)
            # 원문 순서대로 라벨을 읽었을 때 ABC가 아니면 OK
            correct_labels = tuple(labels[original_order.index(i)] for i in range(3))
            if correct_labels != ("A", "B", "C"):
                break
        
        # 각 단락에 새 라벨 부여
        new_paras = [[labels[i], paras[i][1]] for i in range(3)]
        
        # 정답 = 원문 순서대로 라벨 읽기
        correct = tuple(labels[original_order.index(i)] for i in range(3))
        _safe_print(f"  순서 정답: {correct} (원문위치: {original_order})")

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
6. insert_passage: insert_sentence를 뺀 나머지 원문 전체에 ( ① )~( ⑤ ) 위치 표시
   - ⚠ [절대 규칙] insert_sentence 1개만 빼고 나머지 원문의 모든 문장을 그대로 유지!
   - 원문 축소/생략/요약 절대 금지! 삽입 문장 외의 모든 문장이 빠짐없이 포함되어야 함
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
    _generate_order_choices(data, passage=passage)

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
        # ★ 순서 선지를 코드로 직접 생성 (재시도 후에도 반드시 재생성해야 정답/정답지 불일치가 안 생김)
        _generate_order_choices(data, passage=passage)
        block_count2 = len(data.get("full_order_blocks", []))
        if block_count2 != sentence_count:
            _safe_print(f"  WARNING: still mismatch ({block_count2} vs {sentence_count}), using original")
            data["full_order_blocks"] = [[chr(65+i), s] for i, s in enumerate(sentences)]

    # 🔒 삽입 지문: API 결과를 신뢰하지 않고 항상 코드로 재구성
    # (API가 마커를 앞에 몰아넣거나, 정답 위치에 마커를 안 넣는 문제 방지)
    insert_sent = data.get("insert_sentence", "")
    if insert_sent:
        _safe_print("  insert_passage: 코드로 재구성 중...")
        _orig_sents = split_sentences(passage)
        _ins_norm = re.sub(r'\s+', ' ', insert_sent.strip())
        
        # insert_sentence의 원문 위치 찾기
        _ins_idx = -1
        for _si, _s in enumerate(_orig_sents):
            if re.sub(r'\s+', ' ', _s.strip()) == _ins_norm:
                _ins_idx = _si
                break
        
        # 정확히 일치 안 하면 부분 매칭
        if _ins_idx == -1:
            _ins_words = _ins_norm.split()[:8]
            _ins_search = ' '.join(_ins_words)
            for _si, _s in enumerate(_orig_sents):
                if _ins_search in re.sub(r'\s+', ' ', _s.strip()):
                    _ins_idx = _si
                    break
        
        if _ins_idx == -1:
            # 그래도 못 찾으면 가운데 문장
            _ins_idx = len(_orig_sents) // 2
            data["insert_sentence"] = _orig_sents[_ins_idx]
            _safe_print(f"  WARNING: insert_sentence 원문에서 못 찾음 → {_ins_idx}번째 문장 사용")
        
        _remaining = [s for i, s in enumerate(_orig_sents) if i != _ins_idx]
        _n = len(_remaining)
        _markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
        
        # 정답 위치: 삽입 문장이 원래 있던 자리
        _correct_pos = min(_ins_idx, _n)
        
        # 5개 위치를 균등 배치
        if _n >= 5:
            _interval = _n / 5
            _positions = [int(_interval * (i + 0.5)) for i in range(5)]
        else:
            _positions = list(range(min(5, _n + 1)))
        
        # 정답 위치가 포함되도록 조정
        if _correct_pos not in _positions and _positions:
            _closest = min(range(len(_positions)), key=lambda x: abs(_positions[x] - _correct_pos))
            _positions[_closest] = _correct_pos
        
        # 중복 제거 및 정렬
        _positions = sorted(set(_positions))
        # 5개가 안 되면 빈 자리 채우기
        _all_possible = [i for i in range(_n + 1) if i not in _positions]
        while len(_positions) < 5 and _all_possible:
            _positions.append(_all_possible.pop(0))
        _positions.sort()
        
        # 정답 번호 결정
        _answer_idx = _positions.index(_correct_pos) if _correct_pos in _positions else 2
        data["insert_answer"] = f"{_answer_idx + 1}"
        
        # 지문 재구성
        _rebuilt = []
        _positions_set = set(_positions)
        _marker_i = 0
        for _si in range(_n + 1):
            if _si in _positions_set and _marker_i < 5:
                _rebuilt.append(_markers[_marker_i])
                _marker_i += 1
            if _si < _n:
                _rebuilt.append(_remaining[_si])
        while _marker_i < 5:
            _rebuilt.append(_markers[_marker_i])
            _marker_i += 1
        
        data["insert_passage"] = " ".join(_rebuilt).strip()
        _safe_print(f"  재구성 완료: 정답 {data['insert_answer']}번, 마커 위치 {_positions}")

    # ★ 삽입 선지 후처리: ( ④ )와 ( ⑤ )가 연속으로 나오면 ( ⑤ ) 제거
    insert_p = data.get("insert_passage", "")
    if insert_p:
        import re as _re
        # ④⑤ 연속 패턴 제거 (다양한 표기 대응)
        insert_p = _re.sub(
            r'\(\s*[④④]\s*\)(.{0,30})\(\s*[⑤⑤]\s*\)',
            r'( ④ )\g<1>',
            insert_p
        )
        data["insert_passage"] = insert_p

    
    # 🔒 검증: 삽입 문제는 원문을 절대 축약/변형하지 않음
    # - insert_sentence는 원문 문장 중 하나여야 함 (요약/패러프레이즈 금지)
    # - insert_passage에는 insert_sentence를 제외한 나머지 원문 문장이 모두 포함되어야 함
    def _norm(s: str) -> str:
        return re.sub(r'\s+', ' ', (s or '').strip())

    orig_sents = split_sentences(passage)
    orig_norm = [_norm(s) for s in orig_sents]
    ins_sent = _norm(data.get("insert_sentence", ""))
    ins_passage = _norm(data.get("insert_passage", ""))

    need_retry_insert = False
    if ins_sent and ins_sent not in orig_norm:
        need_retry_insert = True

    # 원문 문장 중 insert_sentence로 선택된 1개를 제외한 나머지가 insert_passage에 모두 포함되는지 체크
    if not need_retry_insert and ins_sent:
        missing = []
        for s in orig_norm:
            if s == ins_sent:
                continue
            if s and s not in ins_passage:
                missing.append(s)
        if missing:
            need_retry_insert = True

    if need_retry_insert:
        _safe_print("  WARNING: insert problem changed/truncated source sentence → retrying insert generation once...")
        cache_path = passage_dir / "step2_order.json"
        if cache_path.exists():
            cache_path.unlink()
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
        if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
            data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]
        if data.get("full_order_blocks") and isinstance(data["full_order_blocks"][0], dict):
            data["full_order_blocks"] = [[b["label"], b["text"]] for b in data["full_order_blocks"]]
        _generate_order_choices(data)

        # 재검증 후에도 실패하면, 원문 기반으로 강제 구성(축약 방지)
        ins_sent = _norm(data.get("insert_sentence", ""))
        ins_passage = _norm(data.get("insert_passage", ""))
        if not ins_sent or ins_sent not in orig_norm:
            # 원문 가운데 문장을 삽입문장으로 선택
            pick_idx = max(0, min(len(orig_sents)-1, len(orig_sents)//2))
            data["insert_sentence"] = orig_sents[pick_idx]
            ins_sent = _norm(data["insert_sentence"])

        # insert_passage는 원문에서 insert_sentence 1개만 제거한 본문으로 강제
        remaining = [s for s in orig_sents if _norm(s) != ins_sent]
        # 위치표시는 간단히 5개 구간으로 균등 배치 (본문은 절대 변형하지 않기)
        markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
        rebuilt = []
        for i, s in enumerate(remaining):
            rebuilt.append(s)
            # 문장 사이에 마커를 분산 삽입
            if i < len(remaining) and i < len(markers):
                rebuilt.append(markers[i])
        data["insert_passage"] = " ".join(rebuilt).strip()
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
- 정답: 원문 핵심 표현을 유의어/비유적 표현으로 변형
- 오답: 지문 내용 왜곡, 반대 의미, 미언급 내용
- 각 선지는 15단어 이내로 간결하게
- ★표현 중복 금지★ 정답 선지끼리 의미가 비슷한 건 OK, 하지만 거의 같은 문장을 단어만 바꿔 반복하면 안 됨
- 예시(금지): "all products were designed for usability by everyone" / "all environments combined usability for everyone" → 문장 구조와 핵심어가 너무 유사
- 정답 선지들은 같은 주제를 서로 다른 각도/표현 방식으로 설명해야 함 (예: 비유적 표현, 추상적 요약, 구체적 서술 등 다양하게)

[JSON 형식]
{{
  "blank_passage": "빈칸이 포함된 전체 지문 (빈칸은 ____로 표시)",
  "blank_answer_korean": "빈칸 정답 내용 한국어",
  "blank_options": ["① ...", "② ...", ... "⑫ ..."],
  "blank_correct": ["②", "③", "⑤", ...],
  "blank_wrong": ["①", "④", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)

    # === 후처리: 선지 순서 랜덤 셔플 (정답 위치 고정 방지) ===
    options = data.get("blank_options", [])
    correct_set = set(data.get("blank_correct", []))
    wrong_set = set(data.get("blank_wrong", []))
    if options and len(options) >= 2:
        CIRCLE_NUMS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"]
        import re as _re_blank
        # 선지 텍스트만 추출 (번호 제거)
        texts = []
        old_correct_texts = []
        old_wrong_texts = []
        for opt in options:
            text = _re_blank.sub(r'^[①-⑫]\s*', '', opt).strip()
            texts.append(text)
        for c in correct_set:
            idx_c = CIRCLE_NUMS.index(c) if c in CIRCLE_NUMS else -1
            if 0 <= idx_c < len(texts):
                old_correct_texts.append(texts[idx_c])
        for w in wrong_set:
            idx_w = CIRCLE_NUMS.index(w) if w in CIRCLE_NUMS else -1
            if 0 <= idx_w < len(texts):
                old_wrong_texts.append(texts[idx_w])
        # 셔플
        import random as _rand_blank
        _rand_blank.shuffle(texts)
        # 새 번호 부여 + 정답/오답 재매핑
        new_options = []
        new_correct = []
        new_wrong = []
        for i, text in enumerate(texts):
            label = CIRCLE_NUMS[i]
            new_options.append(f"{label} {text}")
            if text in old_correct_texts:
                new_correct.append(label)
            elif text in old_wrong_texts:
                new_wrong.append(label)
        data["blank_options"] = new_options
        data["blank_correct"] = new_correct
        data["blank_wrong"] = new_wrong
        _safe_print(f"  🔀 빈칸 선지 셔플 완료: 정답 위치 {new_correct}")

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
- 정답: 주제문 키워드를 유의어로 치환한 영어 표현
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
    error_count = max(5, min(8, sent_count))  # 최소 5개, 최대 8개
    bracket_count = min(14, sent_count * 2)  # 문장당 최대 2개 괄호, 최대 14개 (A4 페이지 넘침 방지)
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
6. 새로운 문장을 만들어 넣지 마세요! 원문에 있는 문장만 사용!
7. 출력 결과의 문장을 하나씩 세어보고, {sent_count}개가 아니면 수정하세요

[어법 오류 출제 금지 유형 - 아래는 오류가 아님, 절대 괄호로 출제하지 마세요!]
- start/continue/love/like/hate 뒤: to부정사 = ing (둘 다 허용)
- 주어 자리: to부정사 = 동명사 (둘 다 허용)
- help + 목적어 + 목적격보어: to부정사 = 동사원형 (둘 다 허용) → handle / to handle 출제 금지!
- help + 동사원형/to부정사: help draw = help to draw (둘 다 허용) → draw / to draw 출제 금지!
- 지각동사(see/watch/hear/feel/notice) + 목적어 + 동사원형/현재분사 (둘 다 허용) → look / looking 출제 금지!
- 사역동사(make/let/have) + 목적어 + 동사원형 (이것만 정답, 단 have는 p.p.도 가능)
- and/or 병렬구조에서 to 생략: to A and B = to A and to B (둘 다 허용) → to draw / draw, to label / label 출제 금지!
- as ~ as 원급: 형용사/부사는 문맥으로 판단 (단순 형태만으로 오류 불가)
- 목적격 관계대명사: who = whom (둘 다 허용, 단 전치사 바로 뒤는 whom만)
- 목적격 관계대명사 생략: which/that/who(m) 생략 가능 → which we / we, that we / we 출제 금지!
- ⚠ 위 유형으로 괄호를 만들면 둘 다 정답이 되어 문제가 성립하지 않습니다!
- ⚠ 특히 help/지각동사/병렬구조는 가장 흔한 실수입니다. 반드시 피하세요!

[어법 괄호형 Lv.8-1]
- 원문 {sent_count}개 문장 모두 포함 (출제 안 하는 문장도 원문 그대로)
- {bracket_count}개 괄호: (N)[정답 / 오답] 형태 ← 반드시 이 형식! 예: (1)[looked / look]
- ⚠ 괄호가 없으면 출제 실패입니다! 반드시 (숫자)[A / B] 형태의 괄호를 삽입하세요!
- 한 문장에 여러 괄호 가능
- 정답이 왼쪽인 경우 50%, 오른쪽인 경우 50%가 되도록 반드시 균등 배치 (예: 10개면 5개는 정답이 왼쪽, 5개는 오른쪽)
- 출제: 시제, 대명사, 동명사, to부정사, 형용사/부사, 관계대명사, 분사, 사역동사 등

[어법 서술형 Lv.8-2]
- 원문 {sent_count}개 문장 모두 포함
- 실제 출제 가능한 오류만 삽입 (위 금지 유형 제외)
- 반드시 최소 5개 이상 오류 삽입 (지문이 짧아도 최소 5개!)
- 한 문장에 최대 1개 오류
- 오류를 삽입한 실제 개수를 grammar_error_count에 정확히 기록할 것
- 오류 개수 = grammar_error_answers 배열 길이와 반드시 일치

[JSON 형식]
{{
  "grammar_bracket_passage": "괄호 포함 전체 지문 (정확히 {sent_count}문장)",
  "grammar_bracket_count": {bracket_count},
  "grammar_bracket_answers": [{{"num":1, "answer":"go", "wrong":"will go"}}, ...],
  "grammar_error_passage": "오류 포함 전체 지문 (정확히 {sent_count}문장)",
  "grammar_error_count": 실제삽입개수,
  "grammar_error_answers": [{{"num":1, "original":"watch", "error":"watching"}}, ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
    
    # 🔒 검증: 문장 수 체크
    for key in ['grammar_bracket_passage', 'grammar_error_passage']:
        gen_text = data.get(key, '')
        gen_sents = len(split_sentences(gen_text))
        if gen_sents != sent_count:
            _safe_print(f"  WARNING: {key}: {gen_sents} sentences (original {sent_count}), retrying...")
            cache_path = passage_dir / "step5_grammar.json"
            if cache_path.exists():
                cache_path.unlink()
            data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
            break
    
    
    # 🔒 최종 가드: 8-1/8-2에서 원문에 없는 문장이 추가되었는지 확인
    # 괄호/오류를 제거한 후 원문과 길이 비교 → 20% 이상 길어졌으면 원문 외 내용 추가된 것
    def _strip_brackets(t: str) -> str:
        """어법 괄호 (N)[A / B]를 정답만 남기고 제거"""
        stripped = re.sub(r'\(\d+\)\[([^/\]]+)\s*/\s*[^\]]+\]', r'\1', t)
        return re.sub(r'\s+', ' ', stripped).strip()
    
    orig_len = len(re.sub(r'\s+', '', passage))
    
    # 🔒 8-1 괄호 존재 검증: 괄호가 하나도 없으면 재시도 (최대 5회)
    bracket_text = data.get("grammar_bracket_passage", "")
    for _bracket_retry in range(5):
        if bracket_text and re.search(r'\(\d+\)\[', bracket_text):
            break
        if _bracket_retry == 0 and not bracket_text:
            break  # 지문 자체가 없으면 스킵
        _safe_print(f"  WARNING: grammar_bracket_passage에 괄호 없음 → {_bracket_retry+1}차 재시도")
        cache_path = passage_dir / "step5_grammar.json"
        if cache_path.exists():
            cache_path.unlink()
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
        bracket_text = data.get("grammar_bracket_passage", "")

    for key in ["grammar_bracket_passage", "grammar_error_passage"]:
        gen_text = data.get(key, "")
        if not gen_text:
            continue
        gen_sents_list = split_sentences(gen_text)
        
        # 문장 수 체크
        if len(gen_sents_list) > sent_count:
            _safe_print(f"  WARNING: {key}: {len(gen_sents_list)} sentences > original {sent_count}, trimming...")
            data[key] = " ".join(gen_sents_list[:sent_count]).strip()
        elif len(gen_sents_list) < sent_count:
            _safe_print(f"  WARNING: {key}: {len(gen_sents_list)} sentences < original {sent_count}, using original")
            data[key] = passage
        
        # 길이 체크: 괄호 제거 후 원문 대비 20% 이상 길면 내용 추가된 것
        stripped = _strip_brackets(data.get(key, ""))
        stripped_len = len(re.sub(r'\s+', '', stripped))
        if orig_len > 0 and stripped_len > orig_len * 1.2:
            _safe_print(f"  WARNING: {key} length {stripped_len} >> original {orig_len} (>20%), retrying...")
            cache_path = passage_dir / "step5_grammar.json"
            if cache_path.exists():
                cache_path.unlink()
            retry_data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)
            retry_text = retry_data.get(key, "")
            retry_stripped = _strip_brackets(retry_text)
            retry_len = len(re.sub(r'\s+', '', retry_stripped))
            if retry_len <= orig_len * 1.2:
                data[key] = retry_text
                # 관련 답안도 갱신
                if "bracket" in key:
                    data["grammar_bracket_answers"] = retry_data.get("grammar_bracket_answers", data.get("grammar_bracket_answers", []))
                else:
                    data["grammar_error_answers"] = retry_data.get("grammar_error_answers", data.get("grammar_error_answers", []))
                _safe_print(f"  ✅ {key} retry successful (length {retry_len})")
            else:
                _safe_print(f"  ⚠ {key} retry still too long ({retry_len}), keeping best version")
# ★ 서술형 error_count를 실제 answers 길이로 보정 (문제 텍스트와 일치시키기 위해 반드시 이후에 처리)
    actual_errors = data.get("grammar_error_answers", [])
    actual_error_count = len(actual_errors)
    data["grammar_error_count"] = actual_error_count

    # ★ bracket_count를 지문 내 실제 괄호 수로 보정 (오답박스 수 = 지문 괄호 수)
    actual_brackets = data.get("grammar_bracket_answers", [])
    bracket_passage = data.get("grammar_bracket_passage", "")
    # 지문에서 실제 (N)[...] 패턴 수를 카운트
    actual_bracket_in_text = len(re.findall(r'\(\d+\)\[', bracket_passage))
    if actual_bracket_in_text > 0 and actual_bracket_in_text != len(actual_brackets):
        _safe_print(f"  WARNING: 지문 괄호 {actual_bracket_in_text}개 ≠ answers {len(actual_brackets)}개 → 지문 기준으로 보정")
        # answers가 더 많으면 지문 괄호 수에 맞춰 자름
        if len(actual_brackets) > actual_bracket_in_text:
            # 지문에 실제 존재하는 번호만 유지
            text_nums = set(int(m) for m in re.findall(r'\((\d+)\)\[', bracket_passage))
            actual_brackets = [a for a in actual_brackets if a.get("num") in text_nums]
            data["grammar_bracket_answers"] = actual_brackets
    data["grammar_bracket_count"] = actual_bracket_in_text if actual_bracket_in_text > 0 else len(actual_brackets)

    # ★ 8-1 괄호 자동 검증: 둘 다 정답인 괄호를 올바른 출제로 교체
    bracket_passage_val = data.get("grammar_bracket_passage", "")
    bracket_answers_val = data.get("grammar_bracket_answers", [])
    if bracket_passage_val and bracket_answers_val:
        
        def _make_ing(verb):
            """동사원형 → ~ing 변환"""
            v = verb.strip()
            if v.endswith('e') and not v.endswith('ee'):
                return v[:-1] + 'ing'
            if len(v) >= 3 and v[-1] in 'bdfgklmnprst' and v[-2] in 'aeiou' and v[-3] not in 'aeiou':
                return v + v[-1] + 'ing'
            return v + 'ing'
        
        def _get_base(ing_form):
            """~ing → 동사원형 추출"""
            w = ing_form.strip()
            if not w.endswith('ing') or len(w) <= 4:
                return w[:-3] if w.endswith('ing') else w
            stem = w[:-3]
            if len(stem) >= 2 and stem[-1] == stem[-2]:  # running→run
                return stem[:-1]
            return stem + 'e' if not stem.endswith('e') else stem  # making→make
        
        fixed_nums = {}  # {num: (new_correct, new_wrong, reason)}
        all_brackets_raw = re.findall(r'\((\d+)\)\[([^\]]+)\]', bracket_passage_val)
        
        for num_str, content in all_brackets_raw:
            choices_raw = [c.strip() for c in content.split(' / ')]
            if len(choices_raw) != 2:
                continue
            a_raw, b_raw = choices_raw
            a, b = a_raw.lower(), b_raw.lower()
            
            bracket_pos = bracket_passage_val.find(f'({num_str})[')
            context = bracket_passage_val[max(0, bracket_pos-100):bracket_pos].lower() if bracket_pos > 0 else ""
            
            # 1. help 뒤 to부정사/동사원형 → 정답: 원형, 오답: ~ing
            if (a.startswith('to ') and a[3:] == b) or (b.startswith('to ') and b[3:] == a):
                if 'help' in context:
                    base = b if a.startswith('to ') else a  # 원형
                    wrong = _make_ing(base)
                    # 대소문자 보존
                    correct_raw = b_raw if a.lower().startswith('to ') else a_raw
                    fixed_nums[int(num_str)] = (correct_raw, wrong, "help+to/원형 → 원형/ing")
                elif 'and' in context or 'or' in context:
                    # 병렬구조 to 생략 → 정답: to부정사, 오답: ~ing
                    if a.startswith('to '):
                        base = a[3:]
                        fixed_nums[int(num_str)] = (a_raw, _make_ing(base), "병렬to생략 → to부정사/ing")
                    else:
                        base = b[3:]
                        fixed_nums[int(num_str)] = (b_raw, _make_ing(base), "병렬to생략 → to부정사/ing")
            
            # 1.5. start/begin/continue/love/hate/like/prefer 뒤 to/ing → 둘다 정답이므로 다른 포인트로 교체
            # 이 동사들 뒤에 [to V / V-ing] 또는 [V-ing / to V] 패턴이면 → 시제/수일치로 변환
            dual_ok_verbs = ['start', 'begin', 'continue', 'love', 'hate', 'like', 'prefer',
                             'try', 'remember', 'forget', 'stop', 'regret']
            if any(v + ' ' in context or v + 's ' in context or v + 'ed ' in context for v in dual_ok_verbs):
                # to V / V-ing 패턴인지 확인
                is_to_vs_ing = False
                if a.startswith('to ') and b.endswith('ing') and b.lower() not in _ing_base_verbs:
                    is_to_vs_ing = True
                    base_v = a[3:]
                elif b.startswith('to ') and a.endswith('ing') and a.lower() not in _ing_base_verbs:
                    is_to_vs_ing = True
                    base_v = b[3:]
                if is_to_vs_ing:
                    # 둘 다 정답 → 정답을 원문에 있는 형태로, 오답을 과거시제로 변환
                    # 원문 확인하여 정답 결정
                    if a_raw.lower() in passage.lower():
                        fixed_nums[int(num_str)] = (a_raw, a_raw + 'ed' if not a_raw.endswith('e') else a_raw + 'd', f"둘다가능동사 → 원형/과거")
                    elif b_raw.lower() in passage.lower():
                        fixed_nums[int(num_str)] = (b_raw, b_raw + 'ed' if not b_raw.endswith('e') else b_raw + 'd', f"둘다가능동사 → 원형/과거")
                    else:
                        # 그냥 ing를 정답으로, was+ing를 오답으로
                        ing_form2 = a_raw if a.endswith('ing') else b_raw
                        fixed_nums[int(num_str)] = (ing_form2, 'to have ' + _get_base(ing_form2.lower()), f"둘다가능동사 → ing/to have V")

            # 2. 지각동사 뒤 원형/~ing → 정답: ~ing, 오답: to부정사
            elif (a.endswith('ing') or b.endswith('ing')):
                is_verb_pair = False
                # ing로 끝나는 동사원형 예외 (sing, bring, ring, sting, string, cling, fling, swing, wring, spring, king)
                _ing_base_verbs = {'sing','bring','ring','sting','string','cling','fling','swing','wring','spring','king','thing'}
                # 진짜 ~ing형인지 판별
                a_is_ing = a.endswith('ing') and a.lower() not in _ing_base_verbs
                b_is_ing = b.endswith('ing') and b.lower() not in _ing_base_verbs
                
                if a_is_ing or b_is_ing:
                    ing_form = a_raw if a_is_ing else b_raw
                    base_form = b_raw if a_is_ing else a_raw
                # 원형과 ing가 쌍인지 확인
                if _make_ing(base_form.lower()) == ing_form.lower() or \
                   a.endswith('ing') and (_get_base(a) == b or a[:-3] == b) or \
                   b.endswith('ing') and (_get_base(b) == a or b[:-3] == a):
                    is_verb_pair = True
                
                if is_verb_pair and any(v in context for v in ['see ', 'watch ', 'hear ', 'feel ', 'notice ', 'observe ']):
                    # 지각동사: 정답=ing, 오답=to부정사
                    fixed_nums[int(num_str)] = (ing_form, 'to ' + base_form.lower(), "지각동사 → ing/to부정사")
                elif is_verb_pair and any(v in context for v in ['help ', 'make ', 'let ', 'have ']):
                    # help/사역동사: 정답=원형, 오답=ing
                    fixed_nums[int(num_str)] = (base_form, ing_form, "help/사역 → 원형/ing")
            
            # 3. 목적격 관계대명사 생략 → 정답: which/that, 오답: where/what
            # 단, which/that 둘 다 관계대명사면 건드리지 않음
            elif (a in ['which', 'that', 'whom', 'who'] or b in ['which', 'that', 'whom', 'who']):
                rel_words = {'which', 'that', 'whom', 'who'}
                # 양쪽 다 관계대명사면 건드리지 않음 (which/that 같은 경우)
                if not (a in rel_words and b in rel_words):
                    rel = a_raw if a in ['which', 'that', 'whom', 'who'] else b_raw
                    if rel.lower() in ['which', 'that']:
                        fixed_nums[int(num_str)] = (rel, 'where', "목적격관계대명사 → which/where")
                    elif rel.lower() in ['whom', 'who']:
                        fixed_nums[int(num_str)] = (rel, 'which', "목적격관계대명사 → whom/which")
            elif any(a.startswith(rp) and a[len(rp):].strip() == b.strip() for rp in ['which ', 'that ', 'whom ']):
                # "which we" / "we" → "which we" / "where we"
                for rp in ['which ', 'that ', 'whom ']:
                    if a.startswith(rp):
                        rest = a_raw[len(rp):]
                        fixed_nums[int(num_str)] = (a_raw, 'where ' + rest, "목적격관계대명사 생략 → which/where")
                        break
            elif any(b.startswith(rp) and b[len(rp):].strip() == a.strip() for rp in ['which ', 'that ', 'whom ']):
                for rp in ['which ', 'that ', 'whom ']:
                    if b.startswith(rp):
                        rest = b_raw[len(rp):]
                        fixed_nums[int(num_str)] = (b_raw, 'where ' + rest, "목적격관계대명사 생략 → which/where")
                        break
        
        # 교체 적용
        if fixed_nums:
            result_passage = bracket_passage_val
            new_answers = list(bracket_answers_val)
            
            for num, (correct, wrong, reason) in fixed_nums.items():
                _safe_print(f"  🔧 8-1 괄호({num}) 교체: [{correct} / {wrong}] ({reason})")
                # 지문에서 괄호 내용 교체
                pat = re.compile(r'\(' + str(num) + r'\)\[[^\]]+\]')
                result_passage = pat.sub(f'({num})[{correct} / {wrong}]', result_passage)
                # answers 업데이트
                for ans in new_answers:
                    if ans.get("num") == num:
                        ans["answer"] = correct
                        ans["wrong"] = wrong
                        break
            
            data["grammar_bracket_passage"] = result_passage
            data["grammar_bracket_answers"] = new_answers
            _safe_print(f"  ✅ 둘 다 정답 괄호 {len(fixed_nums)}개 자동 교체 완료")
    
    # ★ 8-1 정답 좌우 진짜 랜덤 shuffle (각 괄호 개별 50% 확률)
    import re as _re, random as _rand_sh
    bracket_answers = data.get("grammar_bracket_answers", [])
    bracket_text = data.get("grammar_bracket_passage", "")
    if bracket_answers and bracket_text:
        result = bracket_text
        for ans in bracket_answers:
            num = ans.get("num", 0)
            if _rand_sh.random() < 0.5:  # 50% 확률로 각각 독립적으로 swap
                pat = _re.compile(r'\(' + str(num) + r'\)\[([^\]]+)\]')
                def do_swap_81(m, n=num):
                    parts = [p.strip() for p in m.group(1).split(' / ')]
                    return f'({n})[{parts[1]} / {parts[0]}]' if len(parts)==2 else m.group(0)
                result = pat.sub(do_swap_81, result)
        data["grammar_bracket_passage"] = result

        # ★ grammar_bracket_passage / grammar_error_passage 중복 제거
    # API가 지문을 2번 붙여서 반환하는 경우 방어
    for key in ["grammar_bracket_passage", "grammar_error_passage"]:
        val = data.get(key, "")
        if val:
            half = len(val) // 2
            # 앞절반과 뒷절반이 80% 이상 유사하면 앞절반만 사용
            first_half = val[:half].strip()
            second_half = val[half:].strip()
            if first_half and second_half:
                overlap = sum(1 for a, b in zip(first_half[-200:], second_half[:200]) if a == b)
                similarity = overlap / min(200, len(first_half), len(second_half))
                if similarity > 0.7:
                    _safe_print(f"  WARNING: {key} appears duplicated, trimming...")
                    data[key] = first_half

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
- 5~10개 괄호: (N)[정답 / 반의어] 또는 (N)[반의어 / 정답] 형태 (최소 5개, 가능하면 8~10개)
- 정답이 왼쪽인 경우와 오른쪽인 경우가 비슷하게 배분 (정답이 항상 왼쪽이면 안됨)
- 한 문장에 괄호는 반드시 1개만 (한 문장에 2개 이상 절대 금지)
- 정답과 오답은 의미가 반대인 단어 쌍으로 구성 (예: regarded/overlooked, effective/futile, mild/severe, constant/intermittent)
- 발음 유사 단어 절대 금지. 반드시 반의어로 출제
- 문맥을 읽어야 정답을 고를 수 있는 수능 수준 반의어 쌍

[Lv.9-1 Part B 규칙]
- 10개 단어 (최소 8개, 가능하면 10개), 각 5개 선택지
- 5개 중 유의어 2개 + 반의어 3개로 구성
- "모두 고르시오" 형태: 유의어만 골라야 정답
- 정답(유의어) 2개의 위치는 5개 선택지 중 무작위로 배치 (항상 앞에 오면 안됨)
- 유의어: 수능 빈출 고급 어휘만 사용 (예: talented→gifted/skilled, progress→advancement/improvement)
  * 쉬운 단어(good/bad/big/small 등) 절대 금지
  * 수능 3~1등급 수준의 정확한 유의어 쌍
  * 해당 단어와 의미가 완전히 일치하는 단어만 (부분 유의어 금지)
- 반의어: 해당 단어와 의미가 정반대인 수능 빈출 단어 3개
  * incompetent/untalented/mediocre 같은 명확한 반의어
  * 발음/철자 유사 단어 절대 금지. 의미 기반으로만 출제

[내용 일치 규칙 - 매우 중요]
- content_match_kr: 반드시 정확히 10개 한국어 선지 (①~⑩). 5개 미만 금지! 일치 3~5개 + 불일치 5~7개
- content_match_en: 반드시 정확히 10개 영어 선지 (①~⑩). 5개 미만 금지! 일치 3~5개 + 불일치 5~7개
- 한국어와 영어 선지의 순서는 서로 다르게 랜덤 배치
- 10개 미만이면 실패로 간주됨. 반드시 ①②③④⑤⑥⑦⑧⑨⑩ 10개 모두 작성할 것

[JSON 형식]
{{
  "vocab_advanced_passage": "괄호 포함 지문",
  "vocab_parta_answers": [{{"num":1, "answer":"regarded", "wrong":"overlooked", "reason":"~로 여겨지다 vs 간과하다"}}, ...],
  "vocab_partb": [{{"word":"regarded", "choices":"considered / perceived / overlooked / neglected / dismissed"}}, ...],
  "vocab_partb_answers": [{{"num":1, "correct":["considered", "perceived"], "wrong":["overlooked", "neglected", "dismissed"]}}, ...],
  "content_match_kr": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ...", "⑥ ...", "⑦ ...", "⑧ ...", "⑨ ...", "⑩ ..."],
  "content_match_kr_answer": ["②", "③", "⑤", ...],
  "content_match_en": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ...", "⑥ ...", "⑦ ...", "⑧ ...", "⑨ ...", "⑩ ..."],
  "content_match_en_answer": ["②", "④", ...]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=6000)

    # 내용일치 10개 미만이면 1회 재시도
    kr_count = len(data.get("content_match_kr", []))
    en_count = len(data.get("content_match_en", []))
    if kr_count < 10 or en_count < 10:
        _safe_print(f"  step6: content_match count insufficient (kr={kr_count}, en={en_count}), retrying...")
        data2 = call_claude_json(SYS_JSON_KR, prompt, max_tokens=6000)
        if len(data2.get("content_match_kr", [])) >= kr_count:
            data["content_match_kr"] = data2.get("content_match_kr", data.get("content_match_kr", []))
            data["content_match_kr_answer"] = data2.get("content_match_kr_answer", data.get("content_match_kr_answer", []))
        if len(data2.get("content_match_en", [])) >= en_count:
            data["content_match_en"] = data2.get("content_match_en", data.get("content_match_en", []))
            data["content_match_en_answer"] = data2.get("content_match_en_answer", data.get("content_match_en_answer", []))
    kr_items = data.get("content_match_kr", [])
    kr_answers = set(data.get("content_match_kr_answer", []))
    if kr_items:
        kr_texts = [re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', item) for item in kr_items]
        kr_correct = [_CIRCLE_NUMS[i] in kr_answers for i in range(len(kr_texts))]
        kr_pairs = list(zip(kr_texts, kr_correct))
        random.shuffle(kr_pairs)
        data["content_match_kr"] = [f"{_CIRCLE_NUMS[i]} {kr_pairs[i][0]}" for i in range(len(kr_pairs))]
        data["content_match_kr_answer"] = [_CIRCLE_NUMS[i] for i in range(len(kr_pairs)) if kr_pairs[i][1]]

    # ★ Part B choices 안에서 정답 위치 랜덤화
    vocab_partb = data.get("vocab_partb", [])
    vocab_partb_answers = data.get("vocab_partb_answers", [])
    for i, (item, ans) in enumerate(zip(vocab_partb, vocab_partb_answers)):
        choices_str = item.get("choices", "")
        correct_list = ans.get("correct", [])
        wrong_list = ans.get("wrong", [])
        if choices_str and correct_list and wrong_list:
            all_choices = correct_list + wrong_list
            random.shuffle(all_choices)
            vocab_partb[i]["choices"] = " / ".join(all_choices)
            vocab_partb_answers[i]["correct"] = [c for c in all_choices if c in correct_list]
            vocab_partb_answers[i]["wrong"] = [c for c in all_choices if c in wrong_list]
    data["vocab_partb"] = vocab_partb
    data["vocab_partb_answers"] = vocab_partb_answers

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

    # ★ 9-1 Part A 5개 미만이면 재시도 (최소 5개 강제)
    actual_parta = data.get("vocab_parta_answers", [])
    if len(actual_parta) < 5:
        _safe_print(f"  step6: Part A {len(actual_parta)}개 < 5개 최소 기준, 재시도...")
        for _retry in range(4):  # 최대 4회 재시도
            data2 = call_claude_json(SYS_JSON_KR, prompt, max_tokens=6000)
            parta2 = data2.get("vocab_parta_answers", [])
            if len(parta2) >= 5:
                data["vocab_parta_answers"] = parta2
                data["vocab_advanced_passage"] = data2.get("vocab_advanced_passage", data.get("vocab_advanced_passage", ""))
                actual_parta = parta2
                _safe_print(f"  step6: Part A 재시도 성공 → {len(parta2)}개")
                break
            _safe_print(f"  step6: Part A 재시도 {_retry+1} 실패 ({len(parta2)}개)")
        if len(actual_parta) < 5:
            _safe_print(f"  ⚠ Part A 최종 {len(actual_parta)}개 - 재시도 모두 실패")
    data["vocab_parta_count"] = len(actual_parta)

    # ★ 9-1 Part A 정답 좌우 진짜 랜덤 shuffle (각 괄호 개별 50% 확률)
    import re as _re2, random as _rand_va
    va_passage = data.get("vocab_advanced_passage", "")
    va_answers = data.get("vocab_parta_answers", [])
    if va_passage and va_answers:
        result_va = va_passage
        for ans in va_answers:
            num = ans.get("num", "")
            if _rand_va.random() < 0.5:  # 50% 확률로 각각 독립적으로 swap
                pat = _re2.compile(r'\(' + str(num) + r'\)\[([^\]]+)\]')
                def do_swap_va(m, n=num):
                    parts = [p.strip() for p in m.group(1).split(' / ')]
                    return f'({n})[{parts[1]} / {parts[0]}]' if len(parts)==2 else m.group(0)
                result_va = pat.sub(do_swap_va, result_va)
        data["vocab_advanced_passage"] = result_va

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
    # 대화문 여부 확인
    is_dialogue = _is_dialogue(sentences)
    
    # 한국어 문장: sentence_translations 그대로 사용 (Step1에서 sentences와 개수 맞춰짐)
    # Stage2 해석예습의 번호와 Stage10 영작 번호가 완전히 동일하게
    kr_sentences = (sentence_translations or [])[:len(sentences)]
    while len(kr_sentences) < len(sentences):
        kr_sentences.append(f"문장 {len(kr_sentences)+1}")

    writing_items = []
    for i, eng in enumerate(sentences):
        words = eng.split()
        kr = kr_sentences[i] if i < len(kr_sentences) else f"문장 {i+1}"
        
        # 대화문에서 6단어 이하 문장: scramble 안 하고 원문 그대로
        if is_dialogue and len(words) <= 6:
            writing_items.append({
                "korean": kr,
                "scrambled": eng,  # 원문 그대로 (배열 불필요)
                "answer": eng
            })
            continue
        
        if is_dialogue:
            # 대화문: 합쳐진 문장을 원래 문장 단위로 분리
            # "Beth: Hello, everyone. It's truly an honor..." 
            # → ["Beth: Hello, everyone.", "It's truly an honor..."]
            # 6단어 이하 부분은 원문 그대로, 나머지만 scramble
            sub_sents = re.split(r'(?<=[.!?])\s+', eng)
            scramble_parts = []
            speaker_prefix = ""
            speaker_match = re.match(r'^([A-Z][a-z]+\s*:\s*)', eng)
            if speaker_match:
                speaker_prefix = speaker_match.group(1)
            
            for si, sub in enumerate(sub_sents):
                sub_words = sub.split()
                if len(sub_words) <= 6:
                    # 짧은 부분은 원문 그대로
                    scramble_parts.append(sub)
                else:
                    # 긴 부분만 scramble
                    # 화자 태그가 이 sub에 있으면 분리
                    sub_prefix = ""
                    sub_text = sub
                    sub_speaker = re.match(r'^([A-Z][a-z]+\s*:\s*)', sub)
                    if sub_speaker:
                        sub_prefix = sub_speaker.group(1)
                        sub_text = sub[len(sub_prefix):]
                    
                    proc = sub_text.split()
                    # 첫 단어 소문자 변환
                    if proc and proc[0][0].isupper() and proc[0] not in ['I', 'I,']:
                        if not (len(proc[0]) > 1 and proc[0][1:].islower() and any(c.isupper() for c in proc[0])):
                            proc[0] = proc[0][0].lower() + proc[0][1:]
                    # 마지막 구두점 제거
                    if proc and proc[-1].endswith(('.', '!', '?')):
                        proc[-1] = proc[-1][:-1]
                    # 셔플
                    shuffled = proc.copy()
                    random.shuffle(shuffled)
                    scramble_parts.append(sub_prefix + ' / '.join(shuffled))
            
            scrambled = ' '.join(scramble_parts)
            writing_items.append({
                "korean": kr,
                "scrambled": scrambled,
                "answer": eng
            })
            continue
        
        # 비대화문: 기존 로직 그대로
        processed = []
        for j, w in enumerate(words):
            if j == 0 and w[0].isupper() and w not in ['I', 'I,']:
                if not (len(w) > 1 and w[1:].islower() and w[0].isupper() and any(c.isupper() for c in w)):
                    w = w[0].lower() + w[1:]
            processed.append(w)
        if processed:
            last = processed[-1]
            if last.endswith(('.', '!', '?')):
                processed[-1] = last[:-1]
        shuffled = processed.copy()
        random.shuffle(shuffled)
        scrambled = ' / '.join(shuffled)

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
    # 정답은 항상 최신 데이터로 재생성 (다른 step 캐시가 바뀌면 정답도 바뀌어야 함)
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

    # Stage 8 서술형 (해설 없이 오류→정답만)
    lv8_error = ['<div class="ablock"><p class="ast">Stage 8 서술형</p>']
    for a in s5.get("grammar_error_answers", []):
        if isinstance(a, dict):
            lv8_error.append(f'<p>{a.get("error","")}->{a.get("original","")}</p>')
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

def _split_sentences_chunks(sentences: list, max_per_page: int = 8) -> list:
    """문장 리스트를 균등 분배하여 페이지별 청크로 나눈다."""
    total = len(sentences)
    logger.debug(f"[Lv3 chunk] 총 문장 수: {total}, max_per_page: {max_per_page}")
    if total <= max_per_page:
        logger.debug(f"[Lv3 chunk] 1페이지로 처리 (문장 {total}개 <= {max_per_page})")
        return [sentences]
    num_pages = math.ceil(total / max_per_page)
    base = total // num_pages
    extra = total % num_pages
    sizes = [base + 1] * extra + [base] * (num_pages - extra)
    logger.debug(f"[Lv3 chunk] 페이지 수: {num_pages}, base: {base}, extra: {extra}, sizes: {sizes}")
    chunks, idx = [], 0
    for i, size in enumerate(sizes):
        chunk = sentences[idx:idx + size]
        logger.debug(f"[Lv3 chunk] 페이지 {i+1}: 문장 {idx+1}~{idx+size}번 ({size}개)")
        chunks.append(chunk)
        idx += size
    return chunks

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
        "sentence_translations": s1.get("sentence_translations", []),
        # Lv.1 어휘
        "vocab": s1.get("vocab", []),
        "test_a": s1.get("test_a", []),
        "test_b": s1.get("test_b", []),
        "test_c": s1.get("test_c", []),
        # Lv.3 문장분석 (전체 문장) + 핵심문장
        "sentences": s1.get("sentences", []),
        "sentence_chunks": _split_sentences_chunks(s1.get("sentences", [])),
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
    # 대화문이면 짧은 문장 병합 (6단어 이하, 같은 화자만)
    sentences = _merge_short_dialogue(sentences)
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
