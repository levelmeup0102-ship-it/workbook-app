# cache clear
#!/usr/bin/env python3
"""Workbook generation pipeline"""
PIPELINE_VERSION = "v10"

# Step별 버전 관리: 해당 step 코드 수정 시 버전만 올리면 캐시 자동 무효화
STEP_VERSIONS = {
    "step1_basic": "v4",
    "step2_order": "v10",
    "step3_blank": "v6",
    "step4_topic": "v5",
    "step5_grammar": "v15",
    "step6_vocab_content": "v7",
    "step7_writing": "v4",
    "step8_answers": "v10",
    "secret_note_a": "v4",  # v4: 유의어 5개, paraphrase 템플릿에서 삭제
    "secret_note_b": "v1",
    "secret_note_c": "v5",  # v5: 유의어 6-7개, 고난도 4-5개, 요지 2배 길이, 가로 배치
    "topic_background": "v3",  # v3: 이미지 사이즈업·다중주입·keyterm 보강
}
import asyncio, json, os, sys, time, random, re, math, logging
from stage7_1_step1_prompt import PROMPT_TEMPLATE

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("pipeline")
from pathlib import Path

# QA 검증 모듈
try:
    from qa_check import fix_single_html, fix_merged_html
    _QA_AVAILABLE = True
except ImportError:
    _QA_AVAILABLE = False

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
        print(msg)
    except Exception:
        print(str(msg))
        # pass

# ============================================================
# 문장 분리 (Dr. Mr. Ms. Mrs. Prof. etc. 경칭 보호)
# ============================================================
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

    # 3.5단계: 따옴표 내부가 너무 길면 추가 slice (짝 보정 X)
    def inner_slice_long_quote(sentence: str, min_words: int = 6) -> list[str]:
        if '§QSEP§' not in sentence:
            return [sentence]

        parts = sentence.split('§QSEP§')

        def wc(seg: str) -> int:
            cleaned = seg.strip().strip('"\u201c\u201d')  # §DOT§는 일부러 복원 X
            tokens = cleaned.split()
            count, i = 0, 0
            while i < len(tokens):
                t = tokens[i]
                if not any(c.isalpha() for c in t.replace('§DOT§', '')):
                    i += 1
                    continue
                # 약어 토큰(§DOT§로 끝남)은 다음 토큰까지 합쳐서 1단어
                while t.endswith('§DOT§') and i + 1 < len(tokens):
                    i += 1
                    t = tokens[i]
                count += 1
                i += 1
            return count

        # 내부 조각 중 하나라도 단어 6 초과 → 전부 분리
        if max(wc(p) for p in parts) > min_words:
            return [p.strip() for p in parts if p.strip()]
        return [sentence]

    new_sentences = []
    for s in sentences:
        new_sentences.extend(inner_slice_long_quote(s))
    sentences = new_sentences

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

    # _safe_print(f"  대화문 감지 → 짧은 문장 병합 (≤{min_words}단어)")

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

    # _safe_print(f"  병합 결과: {len(sentences)}문장 → {len(final)}문장 (-{len(sentences)-len(final)})")
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
            # _safe_print(f"  [WARN] API attempt {attempt+1} failed: {str(e)[:100]}")
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
    # 버전 태그 삽입
    data["_step_version"] = STEP_VERSIONS.get(step_name, "v0")
    # Save locally
    passage_dir.mkdir(parents=True, exist_ok=True)
    path = passage_dir / f"{step_name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # _safe_print(f"  Saved: {step_name}.json (ver={data['_step_version']})")
    # Save to Supabase
    try:
        import supa
        if supa._enabled():
            cache_key = passage_dir.name
            _run_async(supa.save_step_supa(cache_key, step_name, data))
            # _safe_print(f"  Saved to Supabase: {cache_key}/{step_name}")
    except Exception as e:
        _safe_print(f"  [supa] save error: {str(e)[:80]}")

def load_step(passage_dir: Path, step_name: str) -> dict | None:
    expected_ver = STEP_VERSIONS.get(step_name, "v0")
    # Try local first
    path = passage_dir / f"{step_name}.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 버전 체크: 다르면 캐시 무효
        cached_ver = data.get("_step_version", "")
        if cached_ver != expected_ver:
            # _safe_print(f"  Cache outdated: {step_name} (cached={cached_ver}, need={expected_ver}) → regenerate")
            path.unlink()
            return None
        return data
    # Try Supabase
    try:
        import supa
        if supa._enabled():
            cache_key = passage_dir.name
            data = _run_async(supa.get_step(cache_key, step_name))
            if data:
                # 버전 체크
                cached_ver = data.get("_step_version", "")
                if cached_ver != expected_ver:
                    # _safe_print(f"  Supabase cache outdated: {step_name} (cached={cached_ver}, need={expected_ver}) → regenerate")
                    try:
                        _run_async(supa.delete_step(cache_key, step_name))
                    except:
                        pass
                    return None
                # Save locally for future use
                passage_dir.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                # _safe_print(f"  Loaded from Supabase: {step_name} (ver={cached_ver})")
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
def step1_basic_analysis(passage: str, passage_dir: Path, full_translation: str, user_translations: list = []) -> dict:
    cached = load_step(passage_dir, "step1_basic")
    if cached:
        # 캐시가 있어도 user_translations가 새로 제공되면 덮어쓰기
        if user_translations:
            sentences_regex = split_sentences(passage)
            sentences_regex = _merge_short_dialogue(sentences_regex)
            sent_count = len(sentences_regex)
            if len(user_translations) == sent_count:
                cached["sentence_translations"] = user_translations
                cached["translation"] = ' '.join(user_translations)
                save_step(passage_dir, "step1_basic", cached)
                # _safe_print(f"  step1: cache + 사용자 해석 {sent_count}줄 적용")
            else:
                _safe_print(f"  step1: ⚠️ 사용자 해석 {len(user_translations)}줄 ≠ 영어 {sent_count}문장 → 사용자 해석 무시")
        else:
            _safe_print("  step1: using cache")
        return cached

    sentences_regex = split_sentences(passage)
    sentences_regex = _merge_short_dialogue(sentences_regex)
    sent_count = len(sentences_regex)

    # 대화문 병합된 문장 리스트를 API에 명시적으로 전달
    numbered_sentences = "\n".join([f"[문장{i+1}] {s}" for i, s in enumerate(sentences_regex)])
    prompt = f"""다음 영어 지문을 분석하여 JSON을 생성하세요.

[지문 - 총 {sent_count}개 문장]
{passage}

[문장 분리 기준 - 반드시 하단의 내용 준수]
{numbered_sentences}

[생성 항목]
1. vocab: 핵심 어휘 14개 (각각 word, meaning(한국어), synonyms(영어 유의어 4개 쉼표구분))
2. sentence_translations: 위 [문장 분리 기준]의 각 문장에 대한 한국어 번역 (정확히 {sent_count}개, 같은 순서!)
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
3. key_sentences: 시험 출제 가능성이 높은 핵심 문장 8개 (원문 그대로)
4. test_a: vocab에서 뜻 쓰기 테스트용 5개 단어 (영어)
5. test_b: vocab에서 유의어 테스트용 5개 단어 (test_a와 겹치지 않게, 영어)
6. test_c: vocab에서 철자 테스트용 5개 (한국어 뜻)

JSON 형식:
{{
  "vocab": [{{"word":"...", "meaning":"...", "synonyms":"..."}}],
  "sentence_translations": ["첫째 문장 해석...", "둘째 문장 해석...", ...],
  "key_sentences": ["...", "..."],
  "test_a": ["...", "..."],
  "test_b": ["...", "..."],
  "test_c": ["...", "..."]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=4096)

    logger.debug(f"DEBUG | step1_basic_analysis | 한국어 해석 추가 data check\ndata\n> {data}")

    # 🔒 검증: API 문장 분리 대신 항상 regex 사용 (AI가 문장을 합치거나 쪼개는 것 방지)
    # 영어 지문 문장 분리(알고리즘)
    data["sentences"] = sentences_regex
    
    # ★ 사용자 해석이 있으면 Claude 번역을 덮어쓰기
    if user_translations:
        if len(user_translations) == sent_count:
            data["sentence_translations"] = user_translations
            data["translation"] = full_translation
            # _safe_print(f"  ✅ 사용자 해석 {sent_count}줄 적용 (Claude 번역 대체)")
        else:
            _safe_print(f"  ⚠️ 사용자 해석 {len(user_translations)}줄 ≠ 영어 {sent_count}문장 → Claude 번역 사용")
    
    save_step(passage_dir, "step1_basic", data)
    return data

# ============================================================
# 순서 선지 코드 생성 유틸리티
# ============================================================
_CIRCLE_NUMS = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"]
# ①(A)-(C)-(B) ②(B)-(A)-(C) ③(B)-(C)-(A) ④(C)-(A)-(B) ⑤(C)-(B)-(A) 로 수정
# 학생용 5개 선지(template.html:2060)와 정확히 일치해야 함
ORDER_TABLE = [
    ("① (A)-(C)-(B)", ("A", "C", "B")),
    ("② (B)-(A)-(C)", ("B", "A", "C")),
    ("③ (B)-(C)-(A)", ("B", "C", "A")),
    ("④ (C)-(A)-(B)", ("C", "A", "B")),
    ("⑤ (C)-(B)-(A)", ("C", "B", "A")),
]


def _generate_order_choices(data, passage: str = ""):
    """
    1) order_paragraphs 3단락의 원문 위치를 파악
    2) ORDER_TABLE의 5개 선지 중 하나를 무작위로 정답으로 선택
    3) 그 정답이 되도록 단락에 라벨(A/B/C)을 역산 부여
    """
    paras = data.get("order_paragraphs", [])
    if len(paras) != 3:
        raise ValueError(f"STAGE 5 | 단락 개수 이상(3개 필요): {len(paras)}개")

    # 공백 정규화: AI 출력과 원문 모두 단일 공백으로 정규화 후 매칭
    norm_passage = re.sub(r'\s+', ' ', passage)

    def _find_pos(text):
        snippet = re.sub(r'\s+', ' ', text.strip())
        words = snippet.split()
        # 점진적 fallback: 긴 prefix부터 짧은 prefix 순으로 시도
        for n in [30, 20, 15, 10, 8, 6, 5, 4, 3]:
            if len(snippet) >= n:
                pos = norm_passage.find(snippet[:n])
                if pos != -1:
                    return pos
            if len(words) >= n:
                pos = norm_passage.find(' '.join(words[:n]))
                if pos != -1:
                    return pos
        return -1

    positions = [_find_pos(paras[i][1]) for i in range(3)]
    if -1 in positions or len(set(positions)) != 3:
        raise ValueError(f"STAGE 5 | 단락 위치 매칭 실패: {positions}")

    # 원문에서 k번째인 단락의 paras 인덱스
    original_order = sorted(range(3), key=lambda i: positions[i])

    # 5개 중 하나를 정답으로 직접 선택
    answer_str, correct = random.choice(ORDER_TABLE)

    # labels[paras 인덱스] = 그 단락이 가질 라벨
    # 원문 k번째 단락 = paras[original_order[k]] → 라벨 = correct[k]
    labels = [None] * 3
    for k in range(3):
        labels[original_order[k]] = correct[k]

    new_paras = [[labels[i], paras[i][1]] for i in range(3)]
    new_paras.sort(key=lambda x: x[0])  # 표시 순서 A→B→C

    data["order_paragraphs"] = new_paras
    data["order_answer"] = answer_str
    
def _generate_order_block_shuffled(data: dict):

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
   - 지문 내용에 대한 변형/변경 절대 금지. [지문]의 내용을 그저 3개의 part로 나누는 것만 가능.
   - 모든 문장이 빠짐없이 포함되어야 함
3. insert_sentence: 삽입할 문장 1개 (앞뒤 문맥 단서가 명확한 것, [지문]의 문장으로 변형하지 않고 그대로 반환.)
4. insert_passage: insert_sentence를 뺀 [지문]에서의 나머지 문장들에 ( ① )~( ⑤ ) 위치 표시
4.1 (중요)4번으로 지정된 문장을 제외하고 나머지 [지문]의 모든 문장의 내용을 그대로 반환한다.(변형 금지. 원문 축소/생략/요약 절대 금지! 삽입 문장 외의 모든 문장이 빠짐없이 포함되어야 함)
5. insert_answer: 삽입 정답 번호
6. full_order_answer: 정답 순서 (예: "(C)→(G)→(D)→...")

JSON 형식:
{{
  "order_intro": "...",
  "order_paragraphs": [{{"label":"A","text":"..."}}, ...],
  "insert_sentence": "...",
  "insert_passage": "...",
  "insert_answer": "...",
  "full_order_answer": "..."
}}"""
    
   # 단락 위치 매칭이 성공할 때까지 최대 3회 재시도
    max_step2_retries = 3
    data = None
    for _try in range(max_step2_retries):
        try:
            candidate = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
            # 위치 매칭 사전 검증
            paras_check = candidate.get("order_paragraphs", [])
            if len(paras_check) == 3:
                norm_p = re.sub(r'\s+', ' ', passage)
                def _quick_find(t):
                    if isinstance(t, dict):
                        t = t.get("text", "")
                    elif isinstance(t, list) and len(t) >= 2:
                        t = t[1]
                    s = re.sub(r'\s+', ' ', (t or "").strip())
                    for n in [30, 20, 15, 10, 8, 6, 5, 4, 3]:
                        if len(s) >= n:
                            p = norm_p.find(s[:n])
                            if p != -1:
                                return p
                    return -1
                test_positions = [_quick_find(paras_check[i]) for i in range(3)]
                if -1 not in test_positions and len(set(test_positions)) == 3:
                    data = candidate
                    if _try > 0:
                        _safe_print(f"  step2: 재시도 {_try+1}회만에 성공")
                    break
                else:
                    _safe_print(f"  step2: 시도 {_try+1}/{max_step2_retries} 위치 매칭 실패 {test_positions} → 재시도")
            else:
                _safe_print(f"  step2: 시도 {_try+1}/{max_step2_retries} 단락 개수 이상 → 재시도")
        except Exception as e:
            _safe_print(f"  step2: 시도 {_try+1}/{max_step2_retries} 예외 {str(e)[:80]}")
        # 캐시 무효화 위해 약간 대기
        time.sleep(1)

    if data is None:
        # 마지막 시도: 그래도 받아서 진행 (어차피 _generate_order_choices에서 또 검증)
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
        _safe_print(f"  step2: {max_step2_retries}회 재시도 모두 실패, 마지막 결과로 진행")

    data.setdefault("full_order_blocks", [])

    # ※ order_answer / order_choices는 _generate_order_choices에서 결정·세팅됨

    # 변환: order_paragraphs를 [label, text] 형태로
    if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
        data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]
    # if data.get("full_order_blocks") and isinstance(data["full_order_blocks"][0], dict):
        # data["full_order_blocks"] = [[b["label"], b["text"]] for b in data["full_order_blocks"]]

    for i, s in enumerate(sentences):
        data["full_order_blocks"].append([chr(ord('A')+i), s])

    # ★ 순서 선지를 코드로 직접 생성 (AI가 다양하게 안 만드는 문제 해결)
    _generate_order_block_shuffled(data)

    # ★★ 3단락에 라벨 부여 + 정답 결정 (ORDER_TABLE 5개 중 무작위)
    _generate_order_choices(data, passage=passage)

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
        # _generate_order_choices(data)

        # 내용 수동 구성이라 일단 주석처리
        # # 재검증 후에도 실패하면, 원문 기반으로 강제 구성(축약 방지)
        # ins_sent = _norm(data.get("insert_sentence", ""))
        # ins_passage = _norm(data.get("insert_passage", ""))
        # if not ins_sent or ins_sent not in orig_norm:
        #     # 원문 가운데 문장을 삽입문장으로 선택
        #     pick_idx = max(0, min(len(orig_sents)-1, len(orig_sents)//2))
        #     data["insert_sentence"] = orig_sents[pick_idx]
        #     ins_sent = _norm(data["insert_sentence"])

        # # insert_passage는 원문에서 insert_sentence 1개만 제거한 본문으로 강제
        # remaining = [s for s in orig_sents if _norm(s) != ins_sent]
        # # 위치표시는 간단히 5개 구간으로 균등 배치 (본문은 절대 변형하지 않기)
        # markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
        # rebuilt = []
        # for i, s in enumerate(remaining):
        #     rebuilt.append(s)
        #     # 문장 사이에 마커를 분산 삽입
        #     if i < len(remaining) and i < len(markers):
        #         rebuilt.append(markers[i])
        # data["insert_passage"] = " ".join(rebuilt).strip()
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
- 선지 12개: 정답의 개수를 전체의 50-60%로 구성. 나머지(12개에서 정답의 개수를 뺀) 개수가 오답의 개수.
- 정답: 원문 핵심 표현을 유의어/비유적 표현으로 변형
- 오답: 정답과 반대 의미의 내용, 지문에서 미언급 내용
- 각 선지는 15단어 이내로 간결하게
- ★표현 중복 금지★ 정답 선지끼리 의미가 비슷한 건 OK, 하지만 거의 같은 문장을 단어만 바꿔 반복하면 안 됨
- 예시(금지): "all products were designed for usability by everyone" / "all environments combined usability for everyone" → 문장 구조와 핵심어가 너무 유사
- 정답 선지들은 같은 주제를 서로 다른 각도/표현 방식으로 설명해야 함 (예: 비유적 표현, 추상적 요약, 구체적 서술 등 다양하게)

[blank_wrong_translation에 대한 지침]
- blank_wrong의 값에 대한 한글 해석 내용(topic_wrong의 값들에 대한 한글 해석)
    - 양식: "번호(blank_wrong에 넣어진 값) 번호에 해당하는 영어문장의 한글 해석"
    - 예시 데이터
        - blank_wrong: blank_correct에서 blank_wrong의 값을 제외한 번호들
        - blank_wrong_translation: ["① ①번 내용으로 출제된 영어 문장의 한글 해석", "③ ③번 내용으로 출제된 영어 문장의 한글 해석 내용", "④ ④번 내용으로 출제된 영어 문장의 한글 해석 내용"]

[JSON 형식]
{{
  "blank_passage": "빈칸이 포함된 전체 지문 (빈칸은 ____로 표시)",
  "blank_answer_korean": "빈칸 정답 내용 한국어",
  "blank_options": ["① ...", "② ...", ... "⑫ ..."],
  "blank_correct": ["②", "③", "⑤", ...],
  "blank_wrong": ["①", "④", ...]
  "blank_wrong_translation": ["① ①에 대한 한글 해석 문장", "④ ④에 대한 한글 해석 문장", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)

    # === 후처리: 선지 순서 랜덤 셔플 (정답 위치 고정 방지) ===
    options = data.get("blank_options", [])
    correct_set = set(data.get("blank_correct", []))
    wrong_set = set(data.get("blank_wrong", []))
    ai_wrong_labels = data.get("blank_wrong", [])             # 셔플 전 AI 라벨
    ai_wrong_translations = data.get("blank_wrong_translation", [])

    if options and len(options) >= 2:
        CIRCLE_NUMS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"]
        import re as _re_blank
        import random as _rand_blank

        # 선지 텍스트만 추출 (번호 제거) - 셔플 전 순서 보존
        original_texts = [_re_blank.sub(r'^[①-⑫]\s*', '', opt).strip() for opt in options]

        # 정답/오답 텍스트 스냅샷
        old_correct_texts = []
        old_wrong_texts = []
        for c in correct_set:
            idx_c = CIRCLE_NUMS.index(c) if c in CIRCLE_NUMS else -1
            if 0 <= idx_c < len(original_texts):
                old_correct_texts.append(original_texts[idx_c])
        for w in wrong_set:
            idx_w = CIRCLE_NUMS.index(w) if w in CIRCLE_NUMS else -1
            if 0 <= idx_w < len(original_texts):
                old_wrong_texts.append(original_texts[idx_w])

        # 셔플: 원본 copy해서 사용
        shuffled = original_texts.copy()
        _rand_blank.shuffle(shuffled)

        # 새 번호 부여 + 정답/오답 다시 매핑
        new_options, new_correct, new_wrong = [], [], []
        for i, text in enumerate(shuffled):
            label = CIRCLE_NUMS[i]
            new_options.append(f"{label} {text}")
            if text in old_correct_texts:
                new_correct.append(label)
            elif text in old_wrong_texts:
                new_wrong.append(label)

        # 오답 한글 해석 -> 셔플된 라벨로 매핑
        trans_by_new_label = {}
        for old_label, trans in zip(ai_wrong_labels, ai_wrong_translations):
            if old_label not in CIRCLE_NUMS:
                continue
            old_idx = CIRCLE_NUMS.index(old_label)
            if not (0 <= old_idx < len(original_texts)):
                continue
            text = original_texts[old_idx]
            new_idx = shuffled.index(text)
            new_label = CIRCLE_NUMS[new_idx]
            body = _re_blank.sub(r'^[①-⑫]\s*', '', trans).strip()
            trans_by_new_label[new_label] = body

        new_wrong_translation = [
            f"{lbl} {trans_by_new_label[lbl]}"
            for lbl in new_wrong
            if lbl in trans_by_new_label
        ]

        data["blank_options"] = new_options
        data["blank_correct"] = new_correct
        data["blank_wrong"] = new_wrong
        data["blank_wrong_translation"] = new_wrong_translation
        logger.info(f"INFO | STAGE 6 | 빈칸 선지 셔플: 정답 {new_correct} / 오답해석 {len(new_wrong_translation)}건 매핑")

    save_step(passage_dir, "step3_blank", data)
    return data

# ============================================================
# STEP 4: Stage 7 주제 찾기 -> 현재 Stage 4의 하단 문제로 포함되어 있음.
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
- 선지 12개: 정답 5개 + 오답 7개
- 선지는 반드시 영어로 작성 (한국어 금지)
- 정답: 주제문 키워드를 유의어로 치환한 영어 표현
- ★오답 원칙: 오답의 소재·단어는 지문 안에서 가져오되, 글의 메시지/결론을 명백히 뒤집을 것
  · 금지: 지문 근거로 우길 수 있는 "부분적으로 맞는 내용" (채점 분쟁 원인)
  · 금지: 지문에 없는 생뚱맞은 소재 (너무 쉬움)
  · 뒤집은 지점은 글의 요지와 1:1 대조 시 "명백한 거짓"이어야 함
- 추론적 사고 금지: 글에서 직접 언급된 내용만 정답
- 각 선지는 30단어 이내로 간결하게

[topic_wrong_translation에 대한 지침]
- topic_wrong의 값에 대한 한글 해석 내용(topic_wrong의 값의 순서와 같음)
    - 양식: "번호(topic_wrong에 넣어진 값) 번호에 해당하는 영어문장의 한글 해석"
    - 예시
        - topic_wrong: ["①", "③", "④"]
        - topic_wrong_translation: ["① ①번 내용으로 출제된 영어 문장의 한글 해석 내용", "③ ③번 내용으로 출제된 영어 문장의 한글 해석 내용", "④ ④번 내용으로 출제된 영어 문장의 한글 해석 내용"]


[JSON 형식]
{{
  "topic_options": ["① the importance of...", "② how to...", ... "⑫ ..."],
  "topic_correct": ["②", "④", ...],
  "topic_wrong": ["①", "③", ...],
  "topic_wrong_translation": ["① ①번에 해당하는 영어 문장의 한글 해석 내용", "③ ③번에 해당하는 영어 문장의 한글 해석 내용", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)
    data["topic_passage"] = passage
    save_step(passage_dir, "step4_topic", data)
    return data

# ============================================================
# STEP 5: Lv.8 어법
# => Stage 7-1 어법 | Stage 7-2 어법 최종 체크
# ============================================================
def _apply_critical_grammar_filters(data: dict) -> dict:
    """사용자 명시 절대 금지 자리들을 강제 제거 + 재번호 매기기.
    재생성 후에도 잘못된 자리가 들어오는 걸 막기 위해 별도 함수로 분리."""
    final_bp = data.get("grammar_bracket_passage", "")
    if not final_bp:
        return data

    all_br = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp)
    removed = []

    for num_str, content in all_br:
        parts = [p.strip().lower() for p in content.split('/')]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        should_remove = False
        reason = ""

        # 1. 시제 차이 (의미 차이)
        tense_pairs = [
            {'is', 'was'}, {'are', 'were'}, {'am', 'was'},
            {'has', 'had'}, {'have', 'had'},
            {'do', 'did'}, {'does', 'did'},
            {'go', 'went'}, {'goes', 'went'},
            {'come', 'came'}, {'comes', 'came'},
            {'see', 'saw'}, {'sees', 'saw'},
            {'know', 'knew'}, {'knows', 'knew'},
        ]
        if {a, b} in tense_pairs:
            should_remove = True
            reason = f"시제 차이"

        # 2. 진행시제 (be + V-ing) 포함
        if not should_remove:
            def _has_be_ing(s):
                words = s.strip().split()
                return (len(words) == 2 and
                        words[0] in {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being'} and
                        words[1].endswith('ing'))
            if _has_be_ing(a) or _has_be_ing(b):
                should_remove = True
                reason = "진행시제 포함"

        # 3. 주어 자리 동명사 vs to부정사
        if not should_remove:
            is_ing_to = False
            if a.endswith('ing') and b.startswith('to ') and len(b.split()) == 2:
                is_ing_to = True
            elif b.endswith('ing') and a.startswith('to ') and len(a.split()) == 2:
                is_ing_to = True
            if is_ing_to:
                bracket_pos = final_bp.find(f'({num_str})[')
                if bracket_pos >= 0:
                    last_punct = max(
                        final_bp.rfind('.', 0, bracket_pos),
                        final_bp.rfind('!', 0, bracket_pos),
                        final_bp.rfind('?', 0, bracket_pos),
                    )
                    words_before = final_bp[max(0, last_punct + 1):bracket_pos].strip().split()
                    if len(words_before) <= 3:
                        should_remove = True
                        reason = "주어자리 동명사/to부정사"

        # 4. 관계대명사 that vs which
        if not should_remove:
            if {a, b} == {'that', 'which'}:
                should_remove = True
                reason = "that vs which"

        # 5. 부정/긍정 조동사 (could/couldn't 등)
        if not should_remove:
            negation_pairs = [
                {'could', "couldn't"}, {'can', "can't"}, {'will', "won't"},
                {'would', "wouldn't"}, {'should', "shouldn't"},
                {'may', "may not"}, {'must', "mustn't"},
                {'has', "hasn't"}, {'have', "haven't"}, {'had', "hadn't"},
                {'do', "don't"}, {'does', "doesn't"}, {'did', "didn't"},
                {'is', "isn't"}, {'are', "aren't"},
                {'was', "wasn't"}, {'were', "weren't"},
            ]
            if {a, b} in negation_pairs:
                should_remove = True
                reason = "부정/긍정 의미 차이"

        if should_remove:
            correct_word = ""
            for ans in data.get("grammar_bracket_answers", []):
                if ans.get("num") == int(num_str):
                    correct_word = ans.get("answer", "")
                    break
            if not correct_word:
                correct_word = content.split('/')[0].strip()
            if correct_word:
                final_bp = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp)
                removed.append(int(num_str))
                _safe_print(f"  🚫 [재차단] 괄호({num_str}): {reason} → '{correct_word}'")

    if removed:
        data["grammar_bracket_passage"] = final_bp
        data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed]
        data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp))

    # 재번호 매기기 (1부터 시작)
    remaining_nums = [int(m) for m in re.findall(r'\((\d+)\)\[', data.get("grammar_bracket_passage", ""))]
    expected_nums = list(range(1, len(remaining_nums) + 1))
    if remaining_nums and remaining_nums != expected_nums:
        renumber_map = dict(zip(remaining_nums, expected_nums))
        new_passage = data["grammar_bracket_passage"]
        for old_num in remaining_nums:
            new_num = renumber_map[old_num]
            if old_num != new_num:
                new_passage = new_passage.replace(f'({old_num})[', f'(__TMP{new_num}__)[', 1)
        new_passage = re.sub(r'\(__TMP(\d+)__\)\[', lambda m: f'({m.group(1)})[', new_passage)
        new_answers = []
        for ans in data.get("grammar_bracket_answers", []):
            old_n = ans.get("num", 0)
            if old_n in renumber_map:
                ans["num"] = renumber_map[old_n]
                new_answers.append(ans)
        data["grammar_bracket_passage"] = new_passage
        data["grammar_bracket_answers"] = new_answers
        data["grammar_bracket_count"] = len(new_answers)

    return data


# ============================================================
# 어법 괄호형 (현재 Lv.7-1) 분배 + 조립 헬퍼
# ============================================================
def _distribute_brackets(sent_count: int, total: int, max_per: int = 2) -> list:
    """각 문장에 0~max_per개 무작위 분배. counts[i] 합계 = min(total, sent_count*max_per)."""
    counts = [0] * sent_count
    pool = list(range(sent_count)) * max_per
    random.shuffle(pool)
    for i in pool[:min(total, len(pool))]:
        counts[i] += 1
    return counts


def _assemble_bracket_passage(triples, sentences):
    """
    AI가 반환한 [[원문장, 정답, 오답], ...] 를 받아
    grammar_bracket_passage 문자열 + grammar_bracket_answers 리스트로 변환.
    50% 확률로 정답 좌우 swap.

    Returns: (bracket_passage_str, bracket_answers_list)
    """
    if not isinstance(triples, list) or not sentences:
        return "", []

    bracketed = list(sentences)
    answers = []
    n = 0

    for triple in triples:
        if not (isinstance(triple, (list, tuple)) and len(triple) >= 3):
            continue
        src_sent, ans, wrong = triple[0], triple[1], triple[2]
        if not (isinstance(src_sent, str) and isinstance(ans, str) and isinstance(wrong, str)):
            continue
        if not (ans.strip() and wrong.strip()):
            continue
        # 매칭: 정확히 일치하는 문장 우선, 실패 시 strip 후 비교
        idx = -1
        if src_sent in sentences:
            idx = sentences.index(src_sent)
        else:
            stripped = src_sent.strip()
            for i, s in enumerate(sentences):
                if s.strip() == stripped:
                    idx = i
                    break
        if idx == -1:
            continue
        if ans not in bracketed[idx]:
            continue
        n += 1
        # 50% 확률로 정답 좌우 swap
        if random.random() < 0.5:
            bracket_form = f"({n})[{wrong} / {ans}]"
        else:
            bracket_form = f"({n})[{ans} / {wrong}]"
        bracketed[idx] = bracketed[idx].replace(ans, bracket_form, 1)
        logger.debug(f"STAGE 7-1 | STEP 1 | 괄호 추가한 문장 확인: {bracketed[idx]}")
        answers.append({"num": n, "answer": ans, "wrong": wrong})

    return " ".join(bracketed), answers


def step5_grammar(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step5_grammar")
    if cached:
        _safe_print("  step5: using cache")
        return cached

    sentences = split_sentences(passage)
    sent_count = len(sentences)
    word_count = len(passage.split())

    # ★ 지문 길이에 따라 최소 괄호 수 동적 계산 (사용자 명시 요구)
    #   박스 4줄(~80단어) → 최소 2개 / 6줄(~120단어) → 최소 3개 / 그 이상 → 최소 8개
    if word_count <= 80:
        min_brackets = 2
    elif word_count <= 120:
        min_brackets = 3
    else:
        min_brackets = 8

    bracket_count = sent_count  # 문장당 1개 → 합계 = 총 문장 수
    # ★ 8-1 괄호 분배: AI 호출 전에 어느 문장에 몇 개 출제할지 코드가 결정
    bracket_dist = _distribute_brackets(sent_count, bracket_count, max_per=1)
    bracket_dist_lines = "\n".join(
        f"- 문장 {i}번 (\"{sentences[i][:60]}{'...' if len(sentences[i])>60 else ''}\") → {bracket_dist[i]}개"
        for i in range(sent_count)
    )
    logger.debug(f"STAGE 7-1 | STEP 1 | 지문 {word_count}단어 / {sent_count}문장 → 최소 {min_brackets}개, 권장 {bracket_count}개, 분배 {bracket_dist}")
    
    logger.debug("  step5: generating Lv.8 grammar...")
    prompt = PROMPT_TEMPLATE.format(
        sent_count=sent_count,
        passage=passage,
        bracket_count=bracket_count,
        bracket_dist_lines=bracket_dist_lines,
    )

    # ★ v14: Supabase grammar_points 자동 학습 시스템 (stage7-1 어법 출제용)
    # 한국 수능·내신 빈출 어법 함정을 시스템 프롬프트에 동적 주입
    # → 어법 괄호 출제 시 사용자 9가지 + 김대균 영문법 핵심을 우선 활용
    sys_prompt_stage7 = SYS_JSON
    grammar_addendum = _load_grammar_points_for_prompt()
    if grammar_addendum:
        sys_prompt_stage7 = SYS_JSON + "\n\n" + grammar_addendum
        logger.debug(f"  step5: grammar_points 주입됨 ({len(grammar_addendum):,} chars)")

    def _ai_call():
        """call_claude_json + 8-1 triples → string 조립을 한 번에."""
        d = call_claude_json(sys_prompt_stage7, prompt, max_tokens=4000)
        triples = d.get("grammar_bracket_passage", [])
        logger.debug(f"STAGE 7-1 | STEP 1 | AI 응답의 triples 값 확인\n{triples}")
        bracket_str, bracket_answers = _assemble_bracket_passage(triples, sentences)
        logger.debug(f"STAGE 7-1 | STEP 1 | 최종으로 만들어진 지문\n{bracket_str}\n최종으로 만들어진 정답지 데이터\n{bracket_answers}")
        d["grammar_bracket_passage"] = bracket_str
        d["grammar_bracket_answers"] = bracket_answers
        d["grammar_bracket_count"] = len(bracket_answers)
        return d    

    data = _ai_call()

    # 🔒 8-2 grammar_error_passage 검증
    err_text = data.get("grammar_error_passage", "")
    if err_text and len(split_sentences(err_text)) != sent_count:
        _safe_print(f"  WARNING: grammar_error_passage 문장 수 {len(split_sentences(err_text))} != {sent_count}, retrying...")
        cache_path = passage_dir / "step5_grammar.json"
        if cache_path.exists():
            cache_path.unlink()
        data = _ai_call()

    orig_len = len(re.sub(r'\s+', '', passage))
    err_gen = data.get("grammar_error_passage", "")
    if err_gen:
        err_sents_list = split_sentences(err_gen)
        if len(err_sents_list) > sent_count:
            _safe_print(f"  WARNING: grammar_error_passage: {len(err_sents_list)} sentences > {sent_count}, trimming...")
            data["grammar_error_passage"] = " ".join(err_sents_list[:sent_count]).strip()
        elif len(err_sents_list) < sent_count:
            _safe_print(f"  WARNING: grammar_error_passage: {len(err_sents_list)} sentences < {sent_count}, using original")
            data["grammar_error_passage"] = passage

        # 길이 체크: 원문 대비 20% 초과면 내용 추가된 것
        err_clean_len = len(re.sub(r'\s+', '', data.get("grammar_error_passage", "")))
        if orig_len > 0 and err_clean_len > orig_len * 1.2:
            _safe_print(f"  WARNING: grammar_error_passage length {err_clean_len} >> {orig_len} (>20%), retrying...")
            cache_path = passage_dir / "step5_grammar.json"
            if cache_path.exists():
                cache_path.unlink()
            retry_data = _ai_call()
            retry_text = retry_data.get("grammar_error_passage", "")
            retry_clean_len = len(re.sub(r'\s+', '', retry_text))
            if retry_clean_len <= orig_len * 1.2:
                data["grammar_error_passage"] = retry_text
                data["grammar_error_answers"] = retry_data.get("grammar_error_answers", data.get("grammar_error_answers", []))
                logger.info(f"  OK: grammar_error_passage retry successful (length {retry_clean_len})")
            else:
                logger.warning(f"STAGE 7-1 | STEP 2 | grammar_error_passage retry still too long ({retry_clean_len}), keeping best version")
# ★ 서술형: 무의미 항목 제거 (error == original인 경우)
    raw_errors = data.get("grammar_error_answers", [])
    valid_errors = [a for a in raw_errors if isinstance(a, dict) and a.get("error", "").strip() != a.get("original", "").strip()]
    if len(valid_errors) != len(raw_errors):
        removed = len(raw_errors) - len(valid_errors)
        _safe_print(f"  ⚠️ 서술형: 무의미 항목 {removed}개 제거 (error==original)")
        for i, a in enumerate(valid_errors):
            a["num"] = i + 1
        data["grammar_error_answers"] = valid_errors

# ★ 서술형 error_count를 실제 answers 길이로 보정 (문제 텍스트와 일치시키기 위해 반드시 이후에 처리)
    actual_errors = data.get("grammar_error_answers", [])
    actual_error_count = len(actual_errors)
    data["grammar_error_count"] = actual_error_count

    # ★ 8-2 오류 0개면 재시도 (원문 그대로 나온 경우)
    if actual_error_count == 0:
        _safe_print("  ⚠️ 8-2 오류 0개! 원문 그대로 출력됨 → 재시도...")
        for _err_retry in range(3):
            data3 = _ai_call()
            errs3 = data3.get("grammar_error_answers", [])
            if len(errs3) >= 3:
                data["grammar_error_passage"] = data3.get("grammar_error_passage", data.get("grammar_error_passage", ""))
                data["grammar_error_answers"] = errs3
                data["grammar_error_count"] = len(errs3)
                _safe_print(f"  ✅ 8-2 재시도 성공: {len(errs3)}개 오류")
                break
            logger.debug(f"  ⚠ 8-2 재시도 {_err_retry+1} 실패 ({len(errs3)}개)")

    # ★ 8-1 괄호 검증(알고리즘): 둘 다 정답인 괄호를 올바른 출제로 교체
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
        
        _ing_base_verbs = {'sing','bring','ring','sting','string','cling','fling','swing','wring','spring','king','thing'}
        
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
            
            # 1.5. start/begin/continue/love/hate/like/prefer 뒤 to/ing → 둘다 정답이므로 괄호 제거
            dual_ok_verbs = ['start', 'begin', 'continue', 'love', 'hate', 'like', 'prefer',
                             'try', 'remember', 'forget', 'stop', 'regret']
            if any(v + ' ' in context or v + 's ' in context or v + 'ed ' in context for v in dual_ok_verbs):
                is_to_vs_ing = False
                if a.startswith('to ') and b.endswith('ing') and b.lower() not in _ing_base_verbs:
                    is_to_vs_ing = True
                elif b.startswith('to ') and a.endswith('ing') and a.lower() not in _ing_base_verbs:
                    is_to_vs_ing = True
                if is_to_vs_ing:
                    # 둘 다 정답 → 괄호 자체를 제거하고 원문 텍스트로 복원 (REMOVE 마커)
                    # 원문에서 해당 위치의 단어를 찾아서 복원
                    _safe_print(f"  🚫 8-1 괄호({num_str}) 제거: [{a_raw} / {b_raw}] (둘다가능동사 뒤 to/ing)")
                    # REMOVE 마커: 나중에 괄호를 원문 텍스트로 교체
                    # a_raw 또는 b_raw 중 원문에 있는 형태를 정답으로 남기고 괄호 제거
                    if a_raw.lower() in passage.lower():
                        fixed_nums[int(num_str)] = (a_raw, "__REMOVE__", "둘다가능동사 → 괄호제거")
                    elif b_raw.lower() in passage.lower():
                        fixed_nums[int(num_str)] = (b_raw, "__REMOVE__", "둘다가능동사 → 괄호제거")
                    else:
                        fixed_nums[int(num_str)] = (a_raw, "__REMOVE__", "둘다가능동사 → 괄호제거")

            # 2. 지각동사 뒤 원형/~ing → 정답: ~ing, 오답: to부정사
            elif (a.endswith('ing') or b.endswith('ing')):
                is_verb_pair = False
                # ing로 끝나는 동사원형 예외 (sing, bring, ring, sting, string, cling, fling, swing, wring, spring, king)
                _ing_base_verbs = {'sing','bring','ring','sting','string','cling','fling','swing','wring','spring','king','thing'}
                # 진짜 ~ing형인지 판별
                a_is_ing = a.endswith('ing') and a.lower() not in _ing_base_verbs
                b_is_ing = b.endswith('ing') and b.lower() not in _ing_base_verbs
                
                # ★ 진짜 ~ing형이 아니면 이 분기는 처리 대상 아님 (bring/brings 같은 경우)
                if not (a_is_ing or b_is_ing):
                    continue
                
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
                pat = re.compile(r'\(' + str(num) + r'\)\[[^\]]+\]')
                if wrong == "__REMOVE__":
                    # 괄호 자체를 제거하고 원문 텍스트만 남김
                    result_passage = pat.sub(correct, result_passage)
                    # answers에서도 제거
                    new_answers = [a for a in new_answers if a.get("num") != num]
                else:
                    # 지문에서 괄호 내용 교체
                    result_passage = pat.sub(f'({num})[{correct} / {wrong}]', result_passage)
                    # answers 업데이트
                    for ans in new_answers:
                        if ans.get("num") == num:
                            ans["answer"] = correct
                            ans["wrong"] = wrong
                            break
            
            data["grammar_bracket_passage"] = result_passage
            data["grammar_bracket_answers"] = new_answers
            # 괄호 제거 후 count 재보정
            actual_after = len(re.findall(r'\(\d+\)\[', result_passage))
            data["grammar_bracket_count"] = actual_after
            
            # 괄호 제거로 번호 빈칸 발생 시 재번호 매기기
            removed_any = any(w == "__REMOVE__" for _, w, _ in fixed_nums.values())
            if removed_any and actual_after > 0:
                # 현재 지문에 남아있는 괄호 번호 추출 (순서대로)
                remaining_nums = [int(m) for m in re.findall(r'\((\d+)\)\[', result_passage)]
                if remaining_nums != list(range(1, actual_after + 1)):
                    # 재번호 매기기
                    renumber_map = {old: new for new, old in enumerate(remaining_nums, 1)}
                    for old_num, new_num in renumber_map.items():
                        if old_num != new_num:
                            # 임시 마커로 교체 (충돌 방지)
                            result_passage = result_passage.replace(f'({old_num})[', f'(__TMP{new_num}__)[')
                    # 임시 마커를 최종 번호로
                    result_passage = re.sub(r'\(__TMP(\d+)__\)\[', lambda m: f'({m.group(1)})[', result_passage)
                    # answers도 재번호
                    for ans in new_answers:
                        old_n = ans.get("num", 0)
                        if old_n in renumber_map:
                            ans["num"] = renumber_map[old_n]
                    data["grammar_bracket_passage"] = result_passage
                    data["grammar_bracket_answers"] = new_answers
                    _safe_print(f"  🔢 괄호 재번호 완료: {list(renumber_map.items())}")
            
            _safe_print(f"  ✅ 둘 다 정답 괄호 {len(fixed_nums)}개 처리 완료 (남은 괄호: {actual_after}개)")
    

        # ★ grammar_bracket_passage / grammar_error_passage 중복 제거
    # API가 지문을 2번 붙여서 반환하는 경우 방어
    # 중복 제거: AI가 grammar_error_passage를 2번 붙여 반환하는 경우 방어
    # (8-1은 코드 조립이라 중복 불가능 — 검사 제외)
    err_val = data.get("grammar_error_passage", "")
    if err_val:
        half = len(err_val) // 2
        first_half = err_val[:half].strip()
        second_half = err_val[half:].strip()
        if first_half and second_half:
            overlap = sum(1 for a, b in zip(first_half[-200:], second_half[:200]) if a == b)
            similarity = overlap / min(200, len(first_half), len(second_half))
            if similarity > 0.7:
                _safe_print(f"  WARNING: grammar_error_passage appears duplicated, trimming...")
                data["grammar_error_passage"] = first_half

    # ★ 최종 지시대명사(those/these) 복원 — 모든 괄호 처리 후 마지막에 실행
    # AI가 those를 [that/where], [where/that], [those/that] 등으로 만든 뒤
    # do_swap_81이 추가 변환해도 여기서 최종 잡음
    final_bp = data.get("grammar_bracket_passage", "")
    if final_bp and ('those' in passage.lower() or 'these' in passage.lower()):
        all_final_br = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp)
        removed_demo = []
        for num_str, bracket_content in all_final_br:
            bracket_pos = final_bp.find(f'({num_str})[')
            close_pos = final_bp.find(']', bracket_pos)
            if bracket_pos < 0 or close_pos < 0:
                continue
            after_text = final_bp[close_pos+1:close_pos+25].strip().lower()
            # 괄호 바로 뒤에 "that " 또는 "who "가 오면 → those/these 자리일 가능성
            if after_text.startswith('that ') or after_text.startswith('who '):
                # 원문에 "those that" 또는 "those who" 패턴이 있는지 확인
                if 'those that' in passage.lower() or 'those who' in passage.lower():
                    final_bp = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', 'those', final_bp)
                    removed_demo.append(int(num_str))
                    _safe_print(f"  🚫 지시대명사 복원({num_str}): [{bracket_content}] → those")
                elif 'these that' in passage.lower() or 'these who' in passage.lower():
                    final_bp = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', 'these', final_bp)
                    removed_demo.append(int(num_str))
                    _safe_print(f"  🚫 지시대명사 복원({num_str}): [{bracket_content}] → these")
        if removed_demo:
            data["grammar_bracket_passage"] = final_bp
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_demo]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp))
            _safe_print(f"  ✅ 지시대명사 {len(removed_demo)}개 복원 완료")

   # ★ "바로 옆 자리" 금지 자동 제거 (대명사+동사, 조동사+동사, be+분사 등)
    final_bp_na = data.get("grammar_bracket_passage", "")
    if final_bp_na:
        all_br_na = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp_na)
        removed_na = []
        # 금지 단어 패턴: 괄호 바로 앞 단어가 이런 거면 "바로 옆" 자리
        forbidden_left = {
            # 인칭대명사 + 부정대명사
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'one',
            # 지시대명사
            'this', 'that', 'these', 'those',
            # 조동사
            'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must',
            # 완료시제
            'has', 'have', 'had',
            # 진행/수동
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            # do
            'do', 'does', 'did',
        }
        for num_str, content in all_br_na:
            bracket_pos = final_bp_na.find(f'({num_str})[')
            if bracket_pos < 1:
                continue
            # 괄호 바로 앞 텍스트 추출 (괄호 직전 5단어)
            before_text = final_bp_na[max(0, bracket_pos-50):bracket_pos].rstrip()
            words_before = before_text.split()
            if not words_before:
                continue
            # 마지막 단어 (구두점 제거, 소문자)
            last_word = re.sub(r'[^\w]', '', words_before[-1]).lower()
            # 부사 1개만 끼어있는 경우도 체크: "they also" → also 무시하고 they 봄
            common_adverbs = {'also', 'always', 'often', 'never', 'still', 'just', 
                             'only', 'really', 'quite', 'very', 'rather', 'now',
                             'then', 'too', 'so', 'even', 'already', 'yet',
                             'usually', 'sometimes', 'generally', 'typically',
                             'simply', 'merely', 'truly', 'actually'}
            check_word = last_word
            if last_word in common_adverbs and len(words_before) >= 2:
                # 부사면 그 앞 단어 체크
                check_word = re.sub(r'[^\w]', '', words_before[-2]).lower()
            
            if check_word in forbidden_left:
                # "바로 옆" 자리 → 괄호 제거
                # 정답 단어로 복원 (answers에서 찾기)
                correct_word = ""
                for ans in data.get("grammar_bracket_answers", []):
                    if ans.get("num") == int(num_str):
                        correct_word = ans.get("answer", "")
                        break
                if not correct_word:
                    # answers에 없으면 첫 번째 선지를 정답으로 사용
                    parts = [p.strip() for p in content.split('/')]
                    correct_word = parts[0] if parts else ""
                
                if correct_word:
                    final_bp_na = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp_na)
                    removed_na.append(int(num_str))
                    _safe_print(f"  🚫 '바로 옆' 자리 괄호({num_str}) 제거: 앞 단어='{check_word}' → '{correct_word}'")
        
        if removed_na:
            data["grammar_bracket_passage"] = final_bp_na
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_na]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_na))
            _safe_print(f"  ✅ '바로 옆' 자리 {len(removed_na)}개 제거 완료")

  # ★ "바로 옆 자리" 금지 자동 제거 (인칭대명사+동사, 조동사+동사, be+분사 등)
    final_bp_na = data.get("grammar_bracket_passage", "")
    if final_bp_na:
        all_br_na = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp_na)
        removed_na = []
        forbidden_left = {
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'one',
            'this', 'that', 'these', 'those',
            'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must',
            'has', 'have', 'had',
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'do', 'does', 'did',
        }
        common_adverbs = {'also', 'always', 'often', 'never', 'still', 'just', 
                         'only', 'really', 'quite', 'very', 'rather', 'now',
                         'then', 'too', 'so', 'even', 'already', 'yet',
                         'usually', 'sometimes', 'generally', 'typically',
                         'simply', 'merely', 'truly', 'actually', 'not'}
        for num_str, content in all_br_na:
            bracket_pos = final_bp_na.find(f'({num_str})[')
            if bracket_pos < 1:
                continue
            before_text = final_bp_na[max(0, bracket_pos-50):bracket_pos].rstrip()
            words_before = before_text.split()
            if not words_before:
                continue
            last_word = re.sub(r'[^\w]', '', words_before[-1]).lower()
            check_word = last_word
            if last_word in common_adverbs and len(words_before) >= 2:
                check_word = re.sub(r'[^\w]', '', words_before[-2]).lower()
            
            if check_word in forbidden_left:
                correct_word = ""
                for ans in data.get("grammar_bracket_answers", []):
                    if ans.get("num") == int(num_str):
                        correct_word = ans.get("answer", "")
                        break
                if not correct_word:
                    parts = [p.strip() for p in content.split('/')]
                    correct_word = parts[0] if parts else ""
                
                if correct_word:
                    final_bp_na = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp_na)
                    removed_na.append(int(num_str))
                    _safe_print(f"  🚫 '바로 옆' 자리 괄호({num_str}) 제거: 앞 단어='{check_word}' → '{correct_word}'")
        
        if removed_na:
            data["grammar_bracket_passage"] = final_bp_na
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_na]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_na))
            _safe_print(f"  ✅ '바로 옆' 자리 {len(removed_na)}개 제거 완료")

    # ★ 의미 차이 쌍(부정/긍정 조동사 등) 자동 제거 — 어법이 아닌 의미 문제
    final_bp_neg = data.get("grammar_bracket_passage", "")
    if final_bp_neg:
        all_br_neg = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp_neg)
        removed_neg = []
        # 부정/긍정 쌍: 의미가 반대인 단어들
        negation_pairs = [
            # 조동사 부정/긍정
            ('could', "couldn't"), ('could', 'could not'),
            ('can', "can't"), ('can', 'cannot'),
            ('will', "won't"), ('will', 'will not'),
            ('would', "wouldn't"), ('would', 'would not'),
            ('should', "shouldn't"), ('should', 'should not'),
            ('may', "may not"), ('might', "mightn't"), ('might', 'might not'),
            ('must', "mustn't"), ('must', 'must not'),
            # be동사 부정/긍정
            ('is', "isn't"), ('are', "aren't"), ('was', "wasn't"), ('were', "weren't"),
            ('is', 'is not'), ('are', 'are not'),
            # 일반 조동사 부정/긍정
            ('has', "hasn't"), ('have', "haven't"), ('had', "hadn't"),
            ('do', "don't"), ('does', "doesn't"), ('did', "didn't"),
        ]
        # 추상명사 + s 패턴
        uncountable_nouns = {
            'well-being', 'advice', 'information', 'knowledge', 'equipment',
            'furniture', 'research', 'evidence', 'progress', 'news', 'feedback',
            'homework', 'baggage', 'luggage', 'machinery', 'staff', 'traffic',
        }
        for num_str, content in all_br_neg:
            parts = [p.strip().lower() for p in content.split('/')]
            if len(parts) != 2:
                continue
            a, b = parts[0], parts[1]
            should_remove = False
            reason = ""
            # 1. 부정/긍정 쌍 체크
            for pos, neg in negation_pairs:
                if (a == pos and b == neg) or (a == neg and b == pos):
                    should_remove = True
                    reason = f"의미 차이 (부정/긍정): {a}/{b}"
                    break
            # 2. 추상명사 + s 체크
            if not should_remove:
                for noun in uncountable_nouns:
                    if (a == noun and b == noun + 's') or (a == noun + 's' and b == noun):
                        should_remove = True
                        reason = f"추상명사 복수형: {a}/{b}"
                        break
            if should_remove:
                # 정답 단어로 복원
                correct_word = ""
                for ans in data.get("grammar_bracket_answers", []):
                    if ans.get("num") == int(num_str):
                        correct_word = ans.get("answer", "")
                        break
                if not correct_word:
                    correct_word = content.split('/')[0].strip()
                if correct_word:
                    final_bp_neg = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp_neg)
                    removed_neg.append(int(num_str))
                    _safe_print(f"  🚫 의미 차이 괄호({num_str}) 제거: {reason} → '{correct_word}'")
        if removed_neg:
            data["grammar_bracket_passage"] = final_bp_neg
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_neg]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_neg))
            _safe_print(f"  ✅ 의미 차이 괄호 {len(removed_neg)}개 제거 완료")

    # ★ 추가 자동 제거: 시제 차이 / 주어자리 동명사·to부정사 / 관계대명사 that-which 쌍 / 진행시제
    final_bp_extra = data.get("grammar_bracket_passage", "")
    if final_bp_extra:
        all_br_extra = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp_extra)
        removed_extra = []
        for num_str, content in all_br_extra:
            parts = [p.strip().lower() for p in content.split('/')]
            if len(parts) != 2:
                continue
            a, b = parts[0], parts[1]
            should_remove = False
            reason = ""

            # 1. 시제 차이 페어 (의미 차이일 뿐, 어법 X)
            tense_pairs_set = [
                {'is', 'was'}, {'are', 'were'}, {'am', 'was'},
                {'has', 'had'}, {'have', 'had'},
                {'do', 'did'}, {'does', 'did'},
                {'go', 'went'}, {'goes', 'went'},
                {'come', 'came'}, {'comes', 'came'},
                {'see', 'saw'}, {'sees', 'saw'},
                {'know', 'knew'}, {'knows', 'knew'},
            ]
            if {a, b} in tense_pairs_set:
                should_remove = True
                reason = f"시제 차이 (의미 차이): {a}/{b}"

            # ★ 1-2. 진행시제 (be + V-ing) 페어 차단
            # 한쪽이 "is/are/was/were/am + V-ing" 형태 + 다른 쪽이 단순 동사 → 진행 vs 단순 시제 차이
            # 예: "was taking" / "took", "is studying" / "studies"
            if not should_remove:
                def _has_be_ing(s):
                    words = s.strip().split()
                    return (len(words) == 2 and
                            words[0] in {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being'} and
                            words[1].endswith('ing'))
                if _has_be_ing(a) or _has_be_ing(b):
                    should_remove = True
                    reason = f"진행시제 포함 (어법 X, 의미 차이): {a}/{b}"

            # 2. 주어자리 동명사 vs to부정사 (X-ing / to X) — 어근 비교 없이 패턴만 체크
            if not should_remove:
                is_ing_to_pair = False
                if a.endswith('ing') and b.startswith('to ') and len(b.split()) == 2:
                    is_ing_to_pair = True
                elif b.endswith('ing') and a.startswith('to ') and len(a.split()) == 2:
                    is_ing_to_pair = True
                if is_ing_to_pair:
                    # 위치 확인: 문장 시작 부근이면 주어 자리
                    bracket_pos = final_bp_extra.find(f'({num_str})[')
                    if bracket_pos >= 0:
                        last_punct = max(
                            final_bp_extra.rfind('.', 0, bracket_pos),
                            final_bp_extra.rfind('!', 0, bracket_pos),
                            final_bp_extra.rfind('?', 0, bracket_pos),
                        )
                        words_before = final_bp_extra[max(0, last_punct + 1):bracket_pos].strip().split()
                        # 문장 시작 ~ 3단어 이내면 주어 자리로 간주
                        if len(words_before) <= 3:
                            should_remove = True
                            reason = f"주어자리 동명사 vs to부정사 (둘 다 가능): {a}/{b}"

            # 3. 관계대명사 that vs which 쌍 (사용자 명시 금지 자리)
            if not should_remove:
                if {a, b} == {'that', 'which'}:
                    should_remove = True
                    reason = f"관계대명사 자리 that vs which (둘 다 가능): {a}/{b}"

            if should_remove:
                # 정답 단어로 복원
                correct_word = ""
                for ans in data.get("grammar_bracket_answers", []):
                    if ans.get("num") == int(num_str):
                        correct_word = ans.get("answer", "")
                        break
                if not correct_word:
                    correct_word = content.split('/')[0].strip()
                if correct_word:
                    final_bp_extra = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp_extra)
                    removed_extra.append(int(num_str))
                    _safe_print(f"  🚫 추가 차단 괄호({num_str}) 제거: {reason} → '{correct_word}'")
        if removed_extra:
            data["grammar_bracket_passage"] = final_bp_extra
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_extra]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_extra))
            _safe_print(f"  ✅ 추가 차단 괄호 {len(removed_extra)}개 제거 완료")

    # ★ "뒤에 by" 능동/수동 자동 제거 — 뒤에 by 행위자 있으면 수동태 1초 컷
    final_bp_by = data.get("grammar_bracket_passage", "")
    if final_bp_by:
        all_br_by = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp_by)
        removed_by = []
        for num_str, content in all_br_by:
            # 능/수동 쌍인지 확인 (한쪽이 p.p. 형태이거나 are/was/is/were 포함)
            parts = [p.strip().lower() for p in content.split('/')]
            if len(parts) != 2:
                continue
            is_voice_pair = False
            # 패턴 1: be동사 포함 (are replaced / replace)
            for p in parts:
                if 'are ' in p or 'is ' in p or 'was ' in p or 'were ' in p or 'be ' in p or 'been ' in p:
                    is_voice_pair = True
                    break
            # 패턴 2: ed/en vs ing 쌍 (caused / causing)
            if not is_voice_pair:
                ends = [p.split()[-1] if p.split() else '' for p in parts]
                has_ed = any(e.endswith('ed') or e.endswith('en') for e in ends)
                has_ing = any(e.endswith('ing') for e in ends)
                if has_ed and has_ing:
                    is_voice_pair = True
            if not is_voice_pair:
                continue
            # 괄호 뒤 텍스트 50자 안에 "by 단어"가 있는지
            bracket_pos = final_bp_by.find(f'({num_str})[')
            close_pos = final_bp_by.find(']', bracket_pos)
            if close_pos < 0:
                continue
            after_text = final_bp_by[close_pos+1:close_pos+50].lower().strip()
            if re.match(r'^\s*by\s+\w', after_text):
                # 뒤에 by 행위자 → 수동태 1초 컷 → 괄호 제거
                correct_word = ""
                for ans in data.get("grammar_bracket_answers", []):
                    if ans.get("num") == int(num_str):
                        correct_word = ans.get("answer", "")
                        break
                if not correct_word:
                    correct_word = content.split('/')[0].strip()
                if correct_word:
                    final_bp_by = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', correct_word, final_bp_by)
                    removed_by.append(int(num_str))
                    _safe_print(f"  🚫 '뒤에 by' 능/수동 괄호({num_str}) 제거: 뒤='{after_text[:30]}' → '{correct_word}'")
        if removed_by:
            data["grammar_bracket_passage"] = final_bp_by
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_by]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_by))
            _safe_print(f"  ✅ '뒤에 by' 능/수동 {len(removed_by)}개 제거 완료")

    # ★ whom/who 둘다정답 괄호 제거
    final_bp3 = data.get("grammar_bracket_passage", "")
    if final_bp3:
        all_br3 = re.findall(r'\((\d+)\)\[([^\]]+)\]', final_bp3)
        removed_ww = []
        for num_str, bracket_content in all_br3:
            parts = [p.strip().lower() for p in bracket_content.split('/')]
            if len(parts) == 2:
                pair = set(parts)
                if pair == {'whom', 'who'}:
                    # 원문에서 해당 단어 확인
                    if 'whom' in passage.lower():
                        final_bp3 = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', 'whom', final_bp3)
                    else:
                        final_bp3 = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', 'who', final_bp3)
                    removed_ww.append(int(num_str))
                    _safe_print(f"  🚫 whom/who 괄호 제거({num_str})")
        if removed_ww:
            data["grammar_bracket_passage"] = final_bp3
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_ww]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp3))

    # 최종 번호 재정의: (1)부터 시작하도록 번호 정리
    final_bp_renum = data.get("grammar_bracket_passage", "")
    if final_bp_renum:
        remaining_nums = [int(m) for m in re.findall(r'\((\d+)\)\[', final_bp_renum)]
        expected_nums = list(range(1, len(remaining_nums) + 1))
        if remaining_nums and remaining_nums != expected_nums:
            renumber_map = dict(zip(remaining_nums, expected_nums))
            new_passage = final_bp_renum
            # 1단계: 임시 마커 선언
            for old_num in remaining_nums:
                new_num = renumber_map[old_num]
                if old_num != new_num:
                    new_passage = new_passage.replace(f'({old_num})[', f'(__TMP{new_num}__)[', 1)
            # 2단계: 임시 마커 -> 최종 번호
            new_passage = re.sub(r'\(__TMP(\d+)__\)\[', lambda m: f'({m.group(1)})[', new_passage)
            # answers도 재번호
            new_answers = []
            for ans in data.get("grammar_bracket_answers", []):
                old_n = ans.get("num", 0)
                if old_n in renumber_map:
                    ans["num"] = renumber_map[old_n]
                    new_answers.append(ans)
            data["grammar_bracket_passage"] = new_passage
            data["grammar_bracket_answers"] = new_answers
            data["grammar_bracket_count"] = len(new_answers)
            logger.info(f"INFO | STAGE 8 | 최종 번호 매기기 완료: {dict(list(renumber_map.items())[:5])}...")


    # ★★ 최종 안전장치: 항상 핵심 차단 한 번 더 (캐시되기 전)
    # data = _apply_critical_grammar_filters(data) 테스트1: 안전장치 제거.

    save_step(passage_dir, "step5_grammar", data)
    return data

# ============================================================
# 내용일치 part: 셔플 후 데이터 매칭 (kr/en 공용)
# ============================================================
def _shuffle_content_match(items, answers, ai_wrong, ai_wrong_trans):
    """
    content_match_{kr|en}을 셔플하면서 answer/wrong/wrong_trans 라벨을 동시 동기화.

    Returns:
        (new_items, new_answer, new_wrong, new_wrong_trans)
    """
    if not items:
        return list(items), list(answers), list(ai_wrong), list(ai_wrong_trans)

    label_strip = re.compile(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*')
    answers_set = set(answers)

    original_texts = [label_strip.sub('', it).strip() for it in items]
    correct_flags = [_CIRCLE_NUMS[i] in answers_set for i in range(len(original_texts))]

    # 셔플
    pairs = list(zip(original_texts, correct_flags))
    random.shuffle(pairs)
    shuffled = [p[0] for p in pairs]

    # {새 라벨: 오답 본문} dict
    trans_by_new = {}
    for old_label, trans in zip(ai_wrong, ai_wrong_trans):
        if old_label not in _CIRCLE_NUMS:
            continue
        old_idx = _CIRCLE_NUMS.index(old_label)
        if not (0 <= old_idx < len(original_texts)):
            continue
        text = original_texts[old_idx]
        new_idx = shuffled.index(text)
        body = label_strip.sub('', trans).strip()
        trans_by_new[_CIRCLE_NUMS[new_idx]] = body

    new_wrong = [_CIRCLE_NUMS[i] for i in range(len(pairs)) if not pairs[i][1]]

    return (
        [f"{_CIRCLE_NUMS[i]} {shuffled[i]}" for i in range(len(pairs))],
        [_CIRCLE_NUMS[i] for i in range(len(pairs)) if pairs[i][1]],
        new_wrong,
        [f"{lbl} {trans_by_new[lbl]}" for lbl in new_wrong if lbl in trans_by_new],
    )


# ============================================================
# STEP 6: Lv.9 어휘심화 + 내용일치
# => (변경) Stage 8: 어휘 심화 | Stage 9: 내용 일치
# ============================================================
def step6_vocab_content(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step6_vocab_content")
    if cached:
        _safe_print("  step6: using cache")
        return cached

    logger.info("INFO | STAGE 6 | generating Lv.9 vocab...")
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
- ★불일치(오답) 원칙: 오답의 소재·단어는 지문 안에서 가져올 것 (지문에 없는 새 소재 금지 — 너무 쉬움)
  · 1순위 방식: 지문의 실제 한 문장을 골라 사실 한 곳만 정반대로 뒤집기
    (긍정↔부정 / 지문 속 A를 지문 속 B로 / 주체·대상·장소·횟수 바꿔치기)
  · 이렇게 하면 원래 옳은 문장이 그대로 남아 wrong_trans가 정확해짐
  · 금지: 단어 하나만 살짝 비틀어 헷갈리게 / 지문 근거로 우길 수 있는 부분적 사실
  · 불일치 선지는 지문 한 문장과 1:1 대조 시 "명백한 거짓"이어야 함


[content_match_kr_wrong_trans, content_match_en_wrong_trans에 대한 지침]
- 각 content_match_kr_wrong_trans과 content_match_en_wrong_trans은 오답의 번호에 대해 올바른 원래 정답인 string값들을 가진 list이다.
- 오답으로 출제된 문제에 대해 원래 정답인 내용을 말한다.(한글에는 한글 내용, 영어에는 영어 내용)
    - 설명(content_match_kr_wrong_trans)
        - 예시 데이터 형식
            - content_match_kr_wrong: ["②", "③", "⑤", ...]
            - content_match_kr_wrong_trans: ["① 원문의 한글 해석에서 ①번에 해당하는 오답이 아닌 원래 정답의 문장", "④ 원문의 한글 해석에서 ④번에 해당하는 오답이 아닌 원래 정답의 문장", "⑥ 원문의 한글 해석에서 ⑥번에 해당하는 오답이 아닌 원래 정답의 문장", ...]
        - content_match_kr_wrong_trans의 실제 데이터 예시
            - 문제로 작성된 내용: "① 화산 진흙이나 남극의 얼음 아래에서는 생물체를 찾을 수 없다."
            - content_match_kr_wrong_trans에 들어갈 내용: "① 화산 진흙이나 남극의 얼음 아래에서는 생물체를 발견될 수 있다." (실제 옳은 해석)
    - 설명(content_match_en_wrong_trans)
        - 예시 데이터 형식
            - content_match_en_wrong: ["②", "④", ...]
            - content_match_en_wrong_trans: ["① 원문(영어)에서 ①번에 해당하는 오답이 아닌 원래 정답의 문장", "③ 원문(영어)에서 ③번에 해당하는 오답이 아닌 원래 정답의 문장", ...]
        - content_match_en_wrong_trans의 실제 데이터 예시
            - 문제로 작성된 내용: "④ Guests must select 'Travel Packages' in the reservation form."
            - content_match_en_wrong_trans에 들어갈 내용: "④ Select "Vacation Packages" in the reservation form." (문제가 되는 내용을 만들기 위해 사용한 원래 문장)

[content_match_kr, content_match_en 생성 지침]
- 단순 이중 부정한 내용을 정답이라고 제출하지 말 것.(지문의 '부정+부정'의 문장을 '긍정+긍정'으로 바꾸어서 문제로 출제하여 정답에 포함시키지 말 것, 원문 명제를 단순히 이중 부정, 완곡한 부정, 부정어 결합, 또는 ‘~와 무관하지 않다’, ‘~하지 않는 것은 아니다’, ‘~라고 보기 어렵다’ 등의 표현으로 바꾸어 원문과 사실상 같은 의미가 되거나 같은 정답으로 인정될 수 있는 단순 문장은 생성하지 않는다.)

[JSON 형식]
{{
  "vocab_advanced_passage": "괄호 포함 지문",
  "vocab_parta_answers": [{{"num":1, "answer":"regarded", "wrong":"overlooked", "reason":"~로 여겨지다 vs 간과하다"}}, ...],
  "vocab_partb": [{{"word":"regarded", "choices":"considered / perceived / overlooked / neglected / dismissed"}}, ...],
  "vocab_partb_answers": [{{"num":1, "correct":["considered", "perceived"], "wrong":["overlooked", "neglected", "dismissed"]}}, ...],
  "content_match_kr": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ...", "⑥ ...", "⑦ ...", "⑧ ...", "⑨ ...", "⑩ ..."],
  "content_match_kr_answer": ["②", "③", "⑤", ...],
  "content_match_kr_wrong": ["①", "④", "⑥", ...],
  "content_match_kr_wrong_trans": ["① ...", "④ ...", "⑥ ...", ...],
  "content_match_en": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ...", "⑥ ...", "⑦ ...", "⑧ ...", "⑨ ...", "⑩ ..."],
  "content_match_en_answer": ["②", "④", ...],
  "content_match_en_wrong": ["①", "③", ...],
  "content_match_en_wrong_trans": ["① ...", "③ ...", ...]
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
    (data["content_match_kr"],
     data["content_match_kr_answer"],
     data["content_match_kr_wrong"],
     data["content_match_kr_wrong_trans"]) = _shuffle_content_match(
        data.get("content_match_kr", []),
        data.get("content_match_kr_answer", []),
        data.get("content_match_kr_wrong", []),
        data.get("content_match_kr_wrong_trans", []),
    )

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

    # 내용일치 영어 선지 셔플 (번호 재부여 + answer/wrong/wrong_trans 동기화)
    (data["content_match_en"],
     data["content_match_en_answer"],
     data["content_match_en_wrong"],
     data["content_match_en_wrong_trans"]) = _shuffle_content_match(
        data.get("content_match_en", []),
        data.get("content_match_en_answer", []),
        data.get("content_match_en_wrong", []),
        data.get("content_match_en_wrong_trans", []),
    )

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
    va_passage = data.get("vocab_advanced_passage", "")
    va_answers = data.get("vocab_parta_answers", [])
    if va_passage and va_answers:
        result_va = va_passage
        for ans in va_answers:
            num = ans.get("num", "")
            if random.random() < 0.5:  # 50% 확률로 각각 독립적으로 swap
                pat = re.compile(r'\(' + str(num) + r'\)\[([^\]]+)\]')
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
def step7_writing(sentences: list, passage_dir: Path, sentence_translations: list = None) -> dict:
    cached = load_step(passage_dir, "step7_writing")
    if cached:
        _safe_print("  Stage 10 영작: using cache")
        return cached

    _safe_print("  Stage 10 영작 | Lv.10 writing...")
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

    # Lv.4(구 Stage 7의 정답: 정답 번호들+오답의 해석)
    stage4_data = all_data.get("step4") or {}
    correct_number = ', '.join(stage4_data.get("topic_correct") or [])
    wrong_list = stage4_data.get("topic_wrong_translation") or []
    wrong_items = []
    for i in wrong_list:
        wrong_items.append(f'<p>{i}</p>')
    
    wrong_translation = ''.join(wrong_items)

    logger.debug(f"DEBUG | STAGE 4 |오답 선지 해석 DATA CHECK\n{wrong_translation}\n")

    blocks.append(
        '<div class="ablock">'
        '<p class="ast">Stage 4 수업 직후 정리</p>'
        f'<p>[STEP 1 - 주제문 직접 쓰기]<br>정답: {correct_number}</p>'
        '<p>[오답 선지 해석]</p>'
        f'{wrong_translation}'
        '</div>'
    )

    # Lv.5
    s2 = all_data.get("step2", {})
    blocks.append(f'<div class="ablock">'
                  '<p class="ast">Stage 5 순서 배열</p>'
                  f'<p>[A. 순서 배열]<br>정답: {s2.get("order_answer","")}</p>'
                  f'<p>[B. 문장 삽입]<br>정답: {s2.get("insert_answer","")}</p>'
                  f'<p>[심화 - 문장 순서 배열]<br>정답: {s2.get("full_order_answer","")}</p>'
                  '</div>'
    )

    # Lv.6
    s3 = all_data.get("step3") or {}
    correct = ', '.join(s3.get("blank_correct") or [])
    wrong_list6 = s3.get("blank_wrong_translation") or []

    logger.debug(f"DEBUG | STAGE 6 | DATA가 있는가? CHECK\n{wrong_list6}\n")

    wrong_items6 = []
    for i in wrong_list6:
        wrong_items6.append(f'<p>{i}</p>')
    
    stage6_wrong = ''.join(wrong_items6)
    
    logger.debug(f"DEBUG | STAGE 6 |오답 선지 해석 DATA CHECK\n{stage6_wrong}\n")

    blocks.append(f'<div class="ablock">'
                  '<p class="ast">Stage 6 빈칸 추론</p>'
                   f'<p>[STEP 2]<br>정답: {correct}</p>'
                   '<p>[오답 선지 해석]</p>'
                   f'{stage6_wrong}'
                   '</div>')

    # Lv.7-1 (step1 / step2 정답 3단/2단 나열)
    stage7_data = all_data.get("step5") or {}
    stage7_step1_correct_list = stage7_data.get("grammar_bracket_answers") or []
    stage7_step2_error_list = stage7_data.get("grammar_error_answers") or []

    step1_items = []
    for i in range(0, len(stage7_step1_correct_list), 3):
        chunk = stage7_step1_correct_list[i:i+3]
        line = ' '.join(f'({c.get("num")}) {c.get("answer")}' for c in chunk)
        step1_items.append(f'<p>{line}</p>')
    
    stage7_step1_answer = ''.join(step1_items)
    
    step2_items = []
    for i in range(0, len(stage7_step2_error_list), 2):
        chunk = stage7_step2_error_list[i:i+2]
        line = ' '.join(
            f'({c.get("num")}) {c.get("error")} → {c.get("original")}' for c in chunk
        )
        step2_items.append(f'<p>{line}</p>')
    
    stage7_step2_answer = ''.join(step2_items)
    
    blocks.append(f'<div class="ablock"><p class="ast">Stage 7-1 어법</p>'
                  '<p>[STEP 1]</p>'
                  f'<ul>{stage7_step1_answer}</ul>'
                  '<p>[STEP 2]</p>'
                  f'<ul>{stage7_step2_answer}</ul>'
                  '</div>'
                  )
    
    # Stage 8 어휘 (Part A / Part B 정답만: 3단/2단 구성의 정답지)
    stage8_data = all_data.get("step6")
    stage8_partA_list = stage8_data.get("vocab_parta_answers")
    stage8_partB_list = stage8_data.get("vocab_partb_answers")

    partA_items = []
    for i in range(0, len(stage8_partA_list), 3):
        chunk = stage8_partA_list[i:i+3]
        stage8_partA_answers = ' '.join(f'({c.get("num")}) {c.get("answer")}' for c in chunk)
        partA_items.append(f'<p>{stage8_partA_answers}</p>')
    
    insert_data8_A = ''.join(partA_items)

    partB_items = []
    for i in range(0, len(stage8_partB_list), 2):
        chunck = stage8_partB_list[i:i+2]
        stage8_partB_answers = ' '.join(f'({c.get("num")}) {', '.join(c.get("correct"))}' for c in chunck)
        partB_items.append(f'<p>{stage8_partB_answers}</p>')
    
    insert_data_8_B = ''.join(partB_items)

    blocks.append('<div class="ablock"><p class="ast">Stage 8 어휘</p>'
                  '<p>[Part A]</p>'
                  f'<ul>{insert_data8_A}</ul>'
                  '<p>[Part B]</p>'
                  f'<ul>{insert_data_8_B}</ul>'
                  '</div>'
                )

    # Lv.9 - step2 정답 (정답: 번호만 / 오답: 번호+한글해석문장)
    stage9_data = all_data.get("step6") or {}
    correct_numbers_kor = ', '.join(stage9_data.get("content_match_kr_answer") or [])
    wrong_kr_list = stage9_data.get("content_match_kr_wrong_trans") or []
    correct_numbers_eng = ', '.join(stage9_data.get("content_match_en_answer") or [])
    wrong_eng_list = stage9_data.get("content_match_en_wrong_trans") or []

    kor_items = []
    for i in wrong_kr_list:
        kor_items.append(f'<p>{i}</p>')
    
    stage9_wrong_kor = ''.join(kor_items)

    eng_items = []
    for i in wrong_eng_list:
        eng_items.append(f'<p>{i}</p>')

    stage9_wrong_eng = ''.join(eng_items)

    logger.debug(f"DEBUG | STAGE 9 |오답 선지 해석 KOR/ENG DATA CHECK\nKOR> {stage6_wrong}\nENG> {stage9_wrong_eng}\n")

    blocks.append(
          '<div class="ablock"><p class="ast">Stage 9 내용 일치</p>'
          '<p>[STEP 2 - Part A. 한국어]</p>'
          f'<p>정답: {correct_numbers_kor}</p>'
          '<p>[오답 선지 해설 및 정답]</p>'
          f'{stage9_wrong_kor}'
          '<p>[STEP 2 - Part B. English]</p>'
          f'<p>정답: {correct_numbers_eng}</p>'
          '<p>[오답 선지 해설 및 정답]</p>'
          f'{stage9_wrong_eng}'
          '</div>'
      )

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
        # "order_choices": s2.get("order_choices", []),
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
        "writing_chunks": _split_sentences_chunks(s7.get("writing_items", []), max_per_page=8),
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

    # QA 자동 검증
    qa_issues = []
    if _QA_AVAILABLE:
        html, qa_issues = fix_single_html(html)

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

    return qa_issues

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

    # ★ passage_text 안에 ###해석### 이 포함되어 있으면 분리
    if '###해석###' in passage:
        parts = passage.split('###해석###', 1)
        passage = parts[0].strip()
        kr_text = parts[1].strip()
        kr_lines = [l.strip() for l in kr_text.split('\n') if l.strip()]
        if 'user_translations' not in meta:
            meta['user_translations'] = kr_lines
            _safe_print(f"  passage_text에서 해석 {len(kr_lines)}줄 추출")

    logger.debug(f"DEBUG | process_passage | 한국어 해석 후처리 CHECK\n1. 문장분리 풀번역\n> {kr_lines}\n2. meta data check\n> {meta['user_translations']}")

    _safe_print(f"\n{'='*50}")
    _safe_print(f"Processing: {passage_id} ({meta.get('challenge_title','')})")
    _safe_print(f"{'='*50}")

    sentences = split_sentences(passage)
    # 대화문이면 짧은 문장 병합 (6단어 이하, 같은 화자만)
    sentences = _merge_short_dialogue(sentences)
    all_steps = {}

    # Step 1: 기본 분석xx
    user_tr = meta.get("user_translations")
    all_steps["step1"] = step1_basic_analysis(passage, passage_dir, full_translation=kr_text, user_translations=kr_lines)

    # 지문을 한문장씩 나눈 list -> sentences_from_api
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
    # ★ 사용자 해석이 있으면 step7 캐시 무효화 (한국어 문장이 바뀌므로)
    if user_tr:
        step7_cache = passage_dir / "step7_writing.json"
        if step7_cache.exists():
            step7_cache.unlink()
            _safe_print("  step7: 사용자 해석 변경 → 로컬 캐시 삭제")
        # Supabase 캐시도 삭제
        try:
            import supa
            if supa._enabled():
                _run_async(supa.delete_step(passage_dir.name, "step7_writing"))
                _safe_print("  step7: Supabase 캐시도 삭제")
        except:
            pass
    translation = all_steps["step1"].get("translation", "")
    sentence_translations = all_steps["step1"].get("sentence_translations", [])
    
    logger.debug(f"DEBUG | translation와 sentence_translations 비교\ntranslation> {translation}\nsentence_translations> {sentence_translations}")

    all_steps["step7"] = step7_writing(sentences=sentences_from_api, passage_dir=passage_dir, sentence_translations=sentence_translations)

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
    qa_issues = render_pdf(template_data, pdf_path, levels=levels)

    # QA 재생성: 괄호 누락 등 심각한 이슈 감지 시 step5 캐시 삭제 후 재생성
    if qa_issues and "bracket_missing" in qa_issues:
        _safe_print("  QA: 8-1 bracket missing → step5+step8 cache delete (local+Supabase)...")
        # 로컬 삭제
        cache5 = passage_dir / "step5_grammar.json"
        cache8 = passage_dir / "step8_answers.json"
        if cache5.exists(): cache5.unlink()
        if cache8.exists(): cache8.unlink()
        # Supabase도 삭제
        try:
            import supa
            if supa._enabled():
                cache_key = passage_dir.name
                _run_async(supa.delete_step(cache_key, "step5_grammar"))
                _run_async(supa.delete_step(cache_key, "step8_answers"))
                _safe_print(f"  Supabase cache deleted: {cache_key}/step5_grammar, step8_answers")
        except Exception as e:
            _safe_print(f"  [supa] delete error: {str(e)[:80]}")
        # 재생성
        all_steps["step5"] = step5_grammar(passage, passage_dir)
        all_steps["step8"] = step8_answers(all_steps, passage_dir)
        template_data = merge_to_template_data(passage, meta, all_steps)
        pdf_path2 = _unique_path(OUTPUT_DIR, base_name, ".pdf")
        qa_issues2 = render_pdf(template_data, pdf_path2, levels=levels)
        if "bracket_missing" in qa_issues2:
            _safe_print("  QA: still missing after retry — manual check needed")
        else:
            _safe_print("  QA: regeneration success")
            old_html = pdf_path.with_suffix('.html')
            if old_html.exists(): old_html.unlink()
            pdf_path = pdf_path2

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
        
        # ###해석### → 이전 지문에 한국어 해석 연결
        if title == '해석' and passages:
            kr_lines = [l.strip() for l in text.split('\n') if l.strip()]
            passages[-1]["meta"]["user_translations"] = kr_lines
            _safe_print(f"  해석 {len(kr_lines)}줄 → {passages[-1]['meta']['challenge_title']}")
            continue
        
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
    
    # 첫 번째 파일에서 자바스크립트 추출 (한 번만 넣기 위해)
    extracted_script = ""
    if html_files:
        first_html = html_files[0].read_text(encoding='utf-8')
        script_match = _re.search(r'<script>.*?</script>', first_html, _re.DOTALL)
        if script_match:
            extracted_script = script_match.group(0)
    
   # 각 파일에서 <body> 내용만 추출 (단, <script>...</script> 제거)
    all_bodies = []
    for idx, hf in enumerate(html_files):
        html = hf.read_text(encoding='utf-8')
        body_match = _re.search(r'<body[^>]*>(.*?)</body>', html, _re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
            # ⚠️ <script>...</script> 제거: 합본 시 자바스크립트가 여러 번 들어가면 paginate가 중복 실행되어 layout 깨짐
            body_content = _re.sub(r'<script>.*?</script>', '', body_content, flags=_re.DOTALL)
            # ⚠️ page-break-before div 제거: .page의 page-break-after: always 와 이중으로 작용해 빈 페이지 생성
            # 자연스러운 .page 분할에만 의존
            all_bodies.append(body_content)
    
    # 합친 HTML 생성 (자바스크립트는 마지막에 한 번만)
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
{extracted_script}
</body>
</html>"""
    
    # QA 자동 검증 (합본)
    if _QA_AVAILABLE:
        merged = fix_merged_html(merged)

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


# ============================================================
# ★ 비밀노트 (Secret Note) - 수정된 코드
# ============================================================

SYS_SECRET_NOTE_A = """You are an expert English teacher preparing Korean high school students for their FINAL EXAM.
Analyze the passage and return structured study materials.

=== CRITICAL RULES ===
1. VOCABULARY: Select 10 key words. Each must have:
   - exactly 5 synonyms (middle/high school level, 중고등 수준)
   - 3 antonyms with Korean meaning in parentheses
   
2. TOPICS (주제): 3 different topic statements with Korean translation
   - Express what the passage is ABOUT
   - LONG complete phrases (15-25 words), like 수능 format
   - Example: "The essential role of accumulated agricultural knowledge in transforming solar energy into consumable food"

3. TITLES (제목): 3 creative titles - 수능 24번 STYLE with Korean translation
   - MUST BE LONG with subtitle (콜론 사용)
   - Example: "From Sunlight to Supper: The Knowledge Revolution in Food Production"
   - NOT short titles like "Food and Knowledge" - TOO SHORT!

4. MAIN_POINTS (요지): 3 statements in KOREAN only
   - State the key message/lesson of the passage
   - LONG complete sentences in natural Korean (40-60 characters)
   - Must be detailed and comprehensive

5. ONE_SENTENCE_SUMMARY: A SINGLE English sentence that captures the entire passage
   - MUST have at least 2 clauses (절이 2개 이상)
   - MUST be 30-40 words (not too short!)
   - MUST include ALL key content of the passage
   - Example: "While information itself cannot be eaten, food represents solar energy transformed through millennia of accumulated agricultural knowledge, enabling human flourishing."

6. SUMMARY_KR: Korean translation of the one_sentence_summary
   - Natural Korean, not translation-style
   - Must convey the same meaning as the English summary

7. PARAPHRASE: Rewrite the ENTIRE passage in different words
   - SAME number of sentences as original
   - SIMILAR word count as original
   - COMPLETELY different vocabulary and sentence structures

=== JSON FORMAT ===
Return ONLY valid JSON:
{
  "vocabulary": [
    {
      "word": "flourish",
      "synonyms": ["thrive", "prosper", "bloom", "succeed", "grow"],
      "antonyms": ["wither (시들다)", "decline (쇠퇴하다)", "perish (소멸하다)"]
    }
  ],
  "topics": [
    {"en": "The essential role of accumulated agricultural knowledge in transforming solar energy into consumable food", "kr": "축적된 농업 지식이 태양 에너지를 먹을 수 있는 음식으로 변환하는 데 있어서의 핵심적인 역할"},
    {"en": "Topic 2...", "kr": "주제 2 번역"},
    {"en": "Topic 3...", "kr": "주제 3 번역"}
  ],
  "titles": [
    {"en": "From Sunlight to Supper: The Knowledge Revolution in Food Production", "kr": "햇빛에서 저녁 식사까지: 식량 생산의 지식 혁명"},
    {"en": "Title 2...", "kr": "제목 2 번역"},
    {"en": "Title 3...", "kr": "제목 3 번역"}
  ],
  "main_points": [
    "요지 1 - 40-60자의 완전한 한국어 문장",
    "요지 2 - 40-60자의 완전한 한국어 문장",
    "요지 3 - 40-60자의 완전한 한국어 문장"
  ],
  "one_sentence_summary": "A single sentence of 30-40 words with at least 2 clauses that captures all key content...",
  "summary_kr": "위 영어 요약의 자연스러운 한국어 번역...",
  "paraphrase": "Full passage rewritten in different words..."
}

IMPORTANT:
- vocabulary: exactly 10 words, each with exactly 5 synonyms and 3 antonyms (with Korean meaning)
- topics: exactly 3 items (LONG phrases, 15-25 words each)
- titles: exactly 3 items (수능 24번 style with colon, 10-15 words)
- main_points: exactly 3 items (Korean only, 40-60 characters each)
- one_sentence_summary: ONE sentence, 2+ clauses, 30-40 words
- summary_kr: Korean translation of the summary
- paraphrase: FULL passage rewritten
- Return ONLY valid JSON. No markdown, no backticks."""


SYS_SECRET_NOTE_B = """You are an expert English teacher preparing Korean high school students for their FINAL EXAM in ONE WEEK.
Your analysis must be precise, academic, and exam-ready.

=== CRITICAL: TITLE RULES ===
The titles are THE MOST IMPORTANT part. Students will memorize these as potential exam answers.
Each title MUST be a COMPLETE THESIS STATEMENT that:
1. States WHAT the author ARGUES or CLAIMS (not just the topic)
2. Includes the MECHANISM or REASONING (how/why)
3. Could appear as the CORRECT answer in a 주제/요지/제목 찾기 question

❌ TERRIBLE (instant fail - these are just topics, not arguments):
- "Microclimate Control in Outdoor Gardening"
- "Vancouver's Weather Advantage"
- "The Power of Music"
- "Benefits of Reading"
- "Importance of Communication"
- "Local Temperature Management"

✓ EXCELLENT (these state actual claims with reasoning):
- "Strategic manipulation of garden orientation and thermal mass materials can override regional climate limitations for successful cultivation"
- "The deliberate creation of localized growing conditions enables year-round agriculture even in regions with harsh winters"
- "Human intervention in microclimate factors demonstrates that effective gardening depends more on technique than on geographic location"

Notice the difference:
- BAD titles are 3-6 word topic labels
- GOOD titles are 15-25 word thesis statements that make a CLAIM

=== JSON FORMAT ===
Return ONLY valid JSON with this exact structure:
{
  "titles": [
    {"en": "Full thesis statement 15-25 words stating the author's actual argument", "kr": "저자의 주장을 담은 완전한 한국어 문장"},
    {"en": "Same thesis from a different angle", "kr": "한국어 번역"},
    {"en": "Third perspective on the same core argument", "kr": "한국어 번역"}
  ],
  "summary_en": "Comprehensive summary 50-80 words covering: main argument + key evidence + conclusion",
  "summary_kr": "위 영어 요약의 자연스러운 한국어 번역 (번역투 금지, 자연스러운 한국어로)",
  "key_arguments": [
    {"en": "First key point as a complete sentence", "kr": "한국어 번역"},
    {"en": "Second key point", "kr": "한국어 번역"},
    {"en": "Third key point", "kr": "한국어 번역"},
    {"en": "Fourth key point", "kr": "한국어 번역"},
    {"en": "Fifth key point", "kr": "한국어 번역"}
  ],
  "vocabulary": [
    {"word": "word1", "easy": ["syn1","syn2","syn3","syn4","syn5"], "hard": ["h1","h2","h3","h4","h5"]},
    {"word": "word2", "easy": ["syn1","syn2","syn3","syn4","syn5"], "hard": ["h1","h2","h3","h4","h5"]}
  ]
}

=== VOCABULARY RULES ===
- Select exactly 10 important words from the passage
- "easy": 5 synonyms at middle/high school level (중고등 수준)
- "hard": 5 synonyms at 수능~편입 level (고급 어휘)

=== KEY ARGUMENTS RULES ===
- Extract 5 distinct logical points from the passage
- Each must be a complete, standalone sentence
- Together they should cover the entire argument of the passage

Return ONLY valid JSON. No markdown, no backticks, no explanation."""


def generate_secret_note_a(passage: str, translation: str, passage_dir: Path) -> dict:
    """유형 A 비밀노트 (한국어 종합 분석)"""
    cached = load_step(passage_dir, "secret_note_a")
    if cached:
        _safe_print("  ✅ secret_note_a 캐시 사용")
        return cached
    _safe_print("  🔐 secret_note_a 생성 중...")
    prompt = f"Analyze this passage:\n\n{passage}"
    if translation:
        prompt += f"\n\nKorean translation reference:\n{translation}"
    data = call_claude_json(SYS_SECRET_NOTE_A, prompt, max_tokens=5000)
    save_step(passage_dir, "secret_note_a", data)
    return data


def generate_secret_note_b(passage: str, passage_dir: Path, translation: str = "") -> dict:
    """유형 B 비밀노트 (영어 중심 분석) - translation은 선택적"""
    cached = load_step(passage_dir, "secret_note_b")
    if cached:
        _safe_print("  ✅ secret_note_b 캐시 사용")
        return cached
    _safe_print("  🔐 secret_note_b 생성 중...")
    
    prompt = f"Analyze this passage for Korean high school exam preparation:\n\n{passage}"
    
    # translation이 있으면 참고용으로 제공 (번역 품질 향상)
    if translation:
        prompt += f"\n\n[Reference Korean translation - use this to ensure accurate Korean translations]:\n{translation}"
    
    data = call_claude_json(SYS_SECRET_NOTE_B, prompt, max_tokens=4000)
    save_step(passage_dir, "secret_note_b", data)
    return data


def render_secret_note(passages_data: list, note_type: str, school_name: str, teacher_name: str = "") -> str:
    """비밀노트 HTML 렌더링 (Jinja2)"""
    from jinja2 import Environment, FileSystemLoader
    template_file = f"secret_note_type_{note_type.lower()}.html"
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    tmpl = env.get_template(template_file)
    return tmpl.render(passages=passages_data, school_name=school_name, teacher_name=teacher_name)


# ============================================================
# ★ 비밀노트 Type C (어휘+주제/제목/요지+요약) - 1페이지 버전
# ============================================================

SYS_SECRET_NOTE_C = """You are an expert English teacher preparing Korean high school students for their FINAL EXAM.
Analyze the passage and return structured study materials.

=== CRITICAL RULES ===
1. VOCABULARY: Select 10 key words. Each must have:
   - 6-7 synonyms (middle/high school level, 중고등 수준)
   - 4-5 hard_synonyms (편입/고급 수준) with Korean meaning in parentheses
   
2. TOPICS (주제): 3 different topic statements with Korean translation
   - Express what the passage is ABOUT
   - LONG complete phrases (15-25 words), like 수능 format
   - Example: "The essential role of accumulated agricultural knowledge in transforming solar energy into consumable food"

3. TITLES (제목): 3 creative titles - 수능 24번 STYLE with Korean translation
   - MUST BE LONG with subtitle (콜론 사용)
   - Example: "From Sunlight to Supper: The Knowledge Revolution in Food Production"
   - Example: "Harvesting Wisdom: How Centuries of Agricultural Knowledge Feed Humanity"
   - NOT short titles like "Food and Knowledge" - TOO SHORT!

4. MAIN_POINTS (요지): 3 statements in KOREAN only
   - State the key message/lesson of the passage
   - LONG complete sentences in natural Korean (40-60 characters, 2배 길이)
   - Must be detailed and comprehensive, not short summaries

5. PARAPHRASE: Rewrite the ENTIRE passage in different words
   - SAME number of sentences as original
   - SIMILAR word count as original
   - COMPLETELY different vocabulary and sentence structures
   - SAME meaning, different expression
   - This is for students to see the same content expressed differently

=== JSON FORMAT ===
Return ONLY valid JSON:
{
  "vocabulary": [
    {
      "word": "flourish",
      "synonyms": ["thrive", "prosper", "bloom", "succeed", "grow", "blossom", "expand"],
      "hard_synonyms": ["burgeon (번성하다)", "proliferate (확산하다)", "effloresce (개화하다)", "fructify (결실을 맺다)", "luxuriate (무성하게 자라다)"]
    }
  ],
  "topics": [
    {"en": "The essential role of accumulated agricultural knowledge in transforming solar energy into consumable food", "kr": "축적된 농업 지식이 태양 에너지를 먹을 수 있는 음식으로 변환하는 데 있어서의 핵심적인 역할"},
    {"en": "How human civilization has flourished by developing sophisticated methods to capture solar energy", "kr": "인류 문명이 태양 에너지를 포착하는 정교한 방법을 개발하여 번성한 방식"},
    {"en": "The relationship between information accumulation and converting natural resources into sustenance", "kr": "정보 축적과 자연 자원을 식량으로 전환하는 능력 간의 관계"}
  ],
  "titles": [
    {"en": "From Sunlight to Supper: The Knowledge Revolution in Food Production", "kr": "햇빛에서 저녁 식사까지: 식량 생산의 지식 혁명"},
    {"en": "Harvesting Wisdom: How Centuries of Agricultural Knowledge Feed Humanity", "kr": "지혜를 수확하다: 수세기의 농업 지식이 인류를 먹여 살리는 방법"},
    {"en": "The Solar Energy Diet: Thousands of Years of Knowledge in Every Bite", "kr": "태양 에너지 식단: 한 입에 담긴 수천 년의 지식"}
  ],
  "main_points": [
    "우리가 오늘날 먹는 음식은 단순한 칼로리가 아니라, 수천 년에 걸쳐 축적된 농업 지식과 기술의 결정체이다.",
    "인류가 번성할 수 있었던 것은 태양 에너지를 음식으로 변환하는 방법에 대한 지식이 점점 더 정교해졌기 때문이다.",
    "지식의 발전으로 우리가 활용할 수 있는 에너지의 양이 증가했지만, 여전히 태양 에너지의 극히 일부에 불과하다."
  ],
  "paraphrase": "People often claim that information cannot be consumed..."
}

IMPORTANT:
- vocabulary: exactly 10 words, each with 6-7 synonyms and 4-5 hard_synonyms
- topics: exactly 3 items (LONG phrases, 15-25 words each)
- titles: exactly 3 items (수능 24번 style with colon, 10-15 words)
- main_points: exactly 3 items (Korean only, LONG 40-60 characters each)
- paraphrase: FULL passage rewritten, same length as original
- Return ONLY valid JSON. No markdown, no backticks."""


def generate_secret_note_c(passage: str, passage_dir: Path, translation: str = "") -> dict:
    """유형 C 비밀노트 (어휘+주제/제목/요지+요약) - 1페이지"""
    cached = load_step(passage_dir, "secret_note_c")
    if cached:
        _safe_print("  ✅ secret_note_c 캐시 사용")
        return cached
    _safe_print("  🔐 secret_note_c 생성 중...")
    
    prompt = f"Analyze this English passage:\n\n{passage}"
    
    if translation:
        prompt += f"\n\n[Korean translation for reference]:\n{translation}"
    
    data = call_claude_json(SYS_SECRET_NOTE_C, prompt, max_tokens=5000)
    save_step(passage_dir, "secret_note_c", data)
    return data


# ════════════════════════════════════════════════════════════════════
# 0회독 — 수업 전 분석 (4페이지 완전 분석)
# ────────────────────────────────────────────────────────────────────
# 선생님 본인 수업 준비용. 비밀노트 A/B/C와 별개 기능.
# - 시스템 프롬프트: SYS_PRECLASS_ANALYSIS (v4)
# - 생성 함수:      generate_preclass_analysis()
# - 렌더 함수:      render_preclass_analysis() (preclass_analysis.html)
# - 후처리:        _rebalance_grammar_boxes(), _passage_marked_to_html()
# ════════════════════════════════════════════════════════════════════

SYS_PRECLASS_ANALYSIS = """You are an expert Korean high-school English teacher producing exam-preparation analysis at the quality of Pyeongga-won (수능·평가원) standard.

Return ONE JSON object — no markdown, no backticks, no commentary.

═════════════════════════════════════════════════
## ABSOLUTE RULES (8 STRICT RULES)
═════════════════════════════════════════════════

### RULE 1 — PASSAGE IS SACRED (원문 절대 보존)
`passage_marked` reproduces the input passage VERBATIM. After stripping all `[[...]]` markers, the cleaned text must equal the input CHARACTER-BY-CHARACTER.
- NO adding sentences/phrases/words not in input.
- NO paraphrasing, summarizing, or "completing" the passage.

### RULE 2 — MARKER SCOPE = 1~3 WORDS ONLY
Every `[[GRAMMAR]]`, `[[VOCAB]]`, `[[IMPL]]` marker wraps ONLY 1~3 words.

**MARKER FORMAT (CRITICAL):**
- GRAMMAR: use circled numerals — `[[GRAMMAR:n=①]]...[[/GRAMMAR]]`, `[[GRAMMAR:n=②]]`, etc.
  (You may also use plain numbers: `[[GRAMMAR:n=1]]`, `[[GRAMMAR:n=2]]` — both work.)
- VOCAB: use circled letters — `[[VOCAB:l=ⓐ]]...[[/VOCAB]]`, `[[VOCAB:l=ⓑ]]`, etc.
  (You may also use plain letters: `[[VOCAB:l=a]]`, `[[VOCAB:l=b]]` — both work.)
- IMPL: no numbering — `[[IMPL]]...[[/IMPL]]`

Examples of CORRECT narrow marking:
- 관계대명사 what → just "what"
- not A but B → just "not...but rather" or "but rather"
- such as → "such as" (2 words)
- 동명사 부정 → "not taking" (2 words)
- 가주어 it → "It's" (1 word)
- 관계대명사 that → just "that"
- 계속적 which → ", which"
- 동명사 주어 → "Being moderate"

### RULE 3 — BOX MERGING ABSOLUTE MAX = 2
Each box's `num` field: "①" or at most "②③" (2 merged max). NEVER 3+.
If you'd merge 3+, create SEPARATE boxes instead.

### RULE 4 — ALL 4 BOX-SLOTS MUST BE FILLED
`grammar_p1.left`, `grammar_p1.right`, `grammar_p2.left`, `grammar_p2.right` — ALL FOUR must have 3~4 boxes each. Total 12~14 boxes.

Distribution for 12 grammar_notes: 3+3+3+3 (p1.L=①②③ / p1.R=④⑤⑥ / p2.L=⑦⑧⑨ / p2.R=⑩⑪⑫)
Distribution for 13: 3+4+3+3 or 4+3+3+3

★ 절대 규칙 — 어법 박스 묶기 금지 ★
**각 어법 박스는 단 하나의 번호만 가진다.** 다음과 같이 두 개 이상 합치는 것 절대 금지:
- ❌ `"num": "④⑤⑥"` (3개 합침) — FAIL
- ❌ `"num": "⑦⑧"` (2개 합침) — FAIL
- ❌ `"num": "①②"` (2개 합침) — FAIL
- ✅ `"num": "④"`, `"num": "⑤"`, `"num": "⑥"` (각각 별도 박스)

grammar_notes가 12개면 박스도 정확히 12개 (각각 ①, ②, ③ ... ⑫). 14개면 박스 14개.

### RULE 4.5 — 함축(IMPL) 마커 길이 제한
**[[IMPL]]...[[/IMPL]] 안의 내용은 5단어 이내로 한정.** 너무 길게 잡으면 어휘·어법 마커와 겹쳐서 표시가 깨진다.
- ❌ `[[IMPL]]Such real-world clues offer the[[/IMPL]]` (7단어 — FAIL)
- ❌ `[[IMPL]]And the more absorbed in a story a reader is[[/IMPL]]` (10단어 — FAIL)
- ✅ `[[IMPL]]real-world clues[[/IMPL]]` (2단어)
- ✅ `[[IMPL]]explore the human condition[[/IMPL]]` (4단어)

핵심 비유·역접 신호어·주제 응축 표현만 짧게 잡아라.

### RULE 5 — MARKER COUNT IN PASSAGE = NOTES COUNT (★ABSOLUTE — NO EXCEPTIONS★)

**3가지 마커 모두 반드시 본문에 표시해야 한다. 어법(빨간)만 마킹하고 어휘(파랑)·함축(검정) 마커를 빼먹는 것은 절대 금지.**

In `passage_marked`, ALL THREE marker types must appear in the body:

| 마커 종류 | 본문 내 개수 | 대응 항목 |
|---|---|---|
| `[[GRAMMAR:n=N]]...[[/GRAMMAR]]` | = len(grammar_notes) | 보통 12~14개 |
| `[[VOCAB:l=L]]...[[/VOCAB]]` | **= len(vocab_notes) = 정확히 10개** | 어휘 노트와 1:1 매칭 |
| `[[IMPL]]...[[/IMPL]]` | **= len(implication_box.expressions) = 6~7개** | implication 표현 1:1 매칭 |

★ ENFORCEMENT ★
- 본문에 VOCAB 마커가 10개 미만이면 SELF-CHECK 실패 → 다시 본문 스캔해서 어휘 위치 찾기
- 본문에 IMPL 마커가 6개 미만이면 SELF-CHECK 실패 → implication_box.expressions 각 표현을 본문에서 찾아 마킹
- vocab_notes·implication_box를 만들었다면, 그 단어/표현이 본문에 반드시 있다 → 본문에서 위치 찾아 마커로 감싸라

**IMPL 마커 가이드 (★검은 밑줄 = 함축 빈칸 흐름★)**
검은 밑줄은 다음 4가지 중 본문에 실제 등장하는 6~7개를 마킹:
1. 핵심 비유·은유·구체 표현 (예: "compare apples to oranges", "yesterday-you")
2. 전환점 신호어 직후의 핵심구 (예: "Instead, ~", "However, ~")
3. 주제 응축 표현 / 필자 주장 (예: "the only fair comparison is ~", "it's you")
4. 빈칸 출제 후보 구문 (논리 단계의 결정적 어구)

implication_box.expressions의 각 expr 필드는 본문에서 정확히 그 표현으로 마킹된 부분이어야 함.

### RULE 6 — GRAMMAR_NOTES COUNT = STRICT 12~14 (절대 7~10개 안됨)
**MINIMUM 12, ABSOLUTE FLOOR. If you find fewer than 12, you MUST re-scan the passage.**

You MUST systematically scan the passage for ALL these categories:

**카테고리 A — 동사 관련 (Verbs)**
- 수일치 (S-V agreement, 부분표현+V, 동명사 주어 등)
- 시제 (과거/현재완료/대과거/미래완료)
- 능동/수동 (수동태 다양한 형태, by 생략, 진행수동 등)
- 사역동사 (make/have/let + O + V/p.p.)
- 지각동사 (see/hear + O + V/V-ing/p.p.)
- 가정법 (현재/과거/혼합/도치)

**카테고리 B — 준동사 (Verbals)**
- to부정사 (명사적·형용사적·부사적; "S+V+to-V" 5형식 동사 포함)
- 동명사 (주어/목적어/보어/전치사 뒤)
- 분사 (현재/과거; 명사 수식; 보어)
- 분사구문 (능동·수동·완료·접속사+분사)

**카테고리 C — 관계사·접속사**
- 관계대명사 (who/which/that, 제한적 vs 계속적, 소유격 whose)
- 관계부사 (when/where/why/how)
- what (관계대명사 vs 의문사)
- 접속사 that vs 동격 that vs 관계대명사 that
- 양보·조건 부사절 (no matter how, even if, unless 등)

**카테고리 D — 특수구문**
- 가주어 it ~ 진주어 to-V/that
- **가목적어 it ~ 진목적어 to-V/that** ★ABSOLUTELY DO NOT MISS★
  - 5형식 동사(make/find/think/consider/believe/feel) + it + 형용사/명사 + to-V (or that S+V)
  - 본문 예: "makes it impossible to recycle", "find it difficult to V", "consider it necessary to V"
  - **빨간 밑줄 대상: "it" 부분과 "to-V" 부분 모두 (또는 it만)**
- It-cleft 강조구문 (It is X that ~)
- so~that / such~that
- 도치 (부정어 도치, only 도치, so/neither 도치)
- 동격 (쉼표/of/that/to-V)
- 비교 (원급/비교급/최상급, 비교급 강조, the 비교급 the 비교급)

**카테고리 E — 자주 출제되는 어구 구문 ★ABSOLUTELY DO NOT MISS★**
- **prevent/keep/stop/prohibit/discourage A from V-ing** ← 매번 빼먹지 말 것
- **be expected to V / be likely to V / be supposed to V**
- **make/find/consider/think/believe + it + 형/명 + to-V** ← ★가목적어 it ★ 매번 빼먹지 말 것
- **make/find/consider + O + OC(형용사/명사)** ← 일반 5형식
- **부분표현 + of + N + V** (most/all/some/half/the rest of)
- **It takes 사람 시간 to-V**
- **the way (how) S+V / the reason why**
- **as if / as though**

**카테고리 F — 어휘구문·전치사**
- such as / including (예시)
- not A but B / not only A but also B (대조·병렬)
- 전치사 + 동명사 (be used to V-ing, look forward to V-ing 등)
- by V-ing / despite + N · in spite of

**SCAN EVERY SENTENCE.** For each sentence, identify at least 1-2 grammar points if possible. Aim for 12-14 distinct points.

### RULE 7 — TOPICS / TITLES (★ABSOLUTE — NO EXCEPTIONS★)

**topics와 titles는 동일한 크기·무게여야 한다. 한 번에 시각적으로 보아 비슷한 길이여야 한다.**

#### HARD LIMITS (위반 시 self-check 실패 → 재작성)
- **topics 영어: 6~10 단어. ABSOLUTE MAX 10 단어. 100자 이내. (이전 12 → 10으로 더 축소)**
- **titles 영어: 6~10 단어. ABSOLUTE MAX 10 단어.**
- **topics_kr: 한국어 15자 이내**
- **titles_kr: 한국어 15자 이내**
- 한 줄 분량 (절대 두 줄로 넘어가지 않아야 함)

#### topics/titles 작성 원칙
- NEVER "When X happens, Y" / "How X affects Y" / "The importance of X"
- 추상명사 or 동명사로 시작 (Self-comparison, Harnessing, Beyond Petroleum 등)
- topics와 titles **둘 다 콜론(:) 사용 가능**, **둘 다 동일 길이·무게**
- topics가 titles보다 길면 안 됨 — 한 번에 봐서 비슷해야 함

#### [O] 좋은 예시 (이 정도 길이만 허용)
- topics: "Self-comparison as the only meaningful growth metric" (7 단어, 49자)
- topics: "Knowledge as the engine of solar energy conversion" (8 단어, 49자)
- titles: "Beyond Petroleum: The Rise of Plant-Based Alternatives" (7 단어, 53자)
- titles_kr: "햇빛에서 저녁까지: 지식의 혁명" (15자)

#### [X] 절대 금지 예시 (이렇게 길면 즉시 재작성)
- "Corporate adoption of biodegradable alternatives to conventional synthetic materials reflects growing consumer environmental consciousness and regulatory pressure" (25 단어 ❌ — 즉시 압축)
- "Revolutionary sustainable packaging materials derived from marine organisms are replacing traditional petroleum-based products across multiple industries" (18 단어 ❌)
- "How human civilization has flourished by developing sophisticated methods to capture solar energy" (14 단어 ❌)

#### SELF-CHECK (출력 직전 반드시 확인)
□ topics 3개 모두 ≤ 10 단어인가? (단어 수 세어보기)
□ titles 3개 모두 ≤ 10 단어인가?
□ topics_kr / titles_kr 모두 ≤ 15자인가?
□ topics와 titles의 길이가 한 눈에 비슷한가? (topics가 titles 2배 이상이면 FAIL)
→ **하나라도 위반이면 그 항목을 압축해서 다시 작성**

#### ★ 생성 절차 (반드시 이 순서로) ★
1. 먼저 titles 3개를 만든다 (각 6~10 단어).
2. **topics는 각 title과 1:1로 짝지어, 같은 길이(±2 단어)로 만든다.**
3. topics[0]를 만들 때 titles[0]를 보고 "이것과 비슷한 길이인가?" 확인.
4. topics가 길어지려 하면, 핵심 명사구만 남기고 부연(분사구문·관계절·콤마 뒤)을 모두 제거.
→ [X] "Corporate adoption of biodegradable alternatives reflects growing consumer consciousness and regulatory pressure"
→ [O] "Corporate adoption of biodegradable material alternatives" (핵심만, 6단어)

### RULE 8 — NO EMOJIS
Use [O], [X], ?, ■, ●, ★ instead of ✅, ❌, ❓.

═════════════════════════════════════════════════
## VOCABULARY FORMAT
═════════════════════════════════════════════════

### `vocab_notes` (PAGE A, 10 items) — 수능 기본~중급

★★★ 어휘 선정 절대 규칙 ★★★
**다음 단어는 절대 vocab_notes로 선정하지 마라 (선정 시 FAIL):**
- 기능어: that, which, who, whom, whose, where, when, why, how, what
- 관사·전치사: the, a, an, of, in, on, at, by, for, with, from, to, into
- 대명사: it, they, this, those, these, he, she, we, you, I
- 접속사: and, or, but, so, because, although, while, if
- be동사·조동사: is, are, was, were, will, would, can, could, should, must, have, has, had, do, does, did
- 빈출 평이 단어: thing, way, place, time, day, year, people, world, life, work, make, go, come, get, take, put

**vocab_notes에 선정해야 할 단어 (반드시 이런 단어를 골라라):**
1. **글의 주장·핵심 메시지와 직결된 어휘** (예: empathy, perspective, autonomy)
2. **수능·내신 빈출 고난도 어휘** (수능 3등급 이상 학생 기준 어려운 단어)
3. **다의어 중 본문에서 특정 의미로 쓰인 단어** (예: "address"가 "다루다"로 쓰일 때)
4. **학술적 추상명사·분사** (예: implementation, sustainability, fragmented)
5. **본문에서 빈도가 낮지만 의미 핵심을 담는 명사·동사·형용사**

선정 우선순위:
- 명사 > 동사 > 형용사 > 부사 (이 순서로 선호)
- **고유명사·지명·인명은 제외** (Mr. Smith, New York 같은 건 NO)
- **숫자·연도·통계 수치 제외** (2023, 1,000명 같은 건 NO)

[O] 좋은 vocab_notes 예시:
{"letter": "ⓐ", "word": "moderation",
 "syns": [["temperance","절제"], ["balance","균형"]],
 "ants": [["excess","과잉"]]}
{"letter": "ⓑ", "word": "empathy",
 "syns": [["compassion","연민"], ["understanding","이해"]],
 "ants": [["indifference","무관심"]]}

[X] 절대 금지 예시:
{"letter": "ⓐ", "word": "that"}          ❌ 기능어
{"letter": "ⓐ", "word": "the"}           ❌ 관사
{"letter": "ⓐ", "word": "Voice Assistants"} ❌ 고유명사·복합어
{"letter": "ⓐ", "word": "general public"}   ❌ 너무 평이한 표현
{"letter": "ⓐ", "word": "make"}           ❌ 빈출 평이 동사

### `vocab_detail` (PAGE C, 6 items) — ★편입영어/GRE 수준 고난도★
Each item:
- letter, word, ko
- **syns: 4개 — 전부 고급 어휘. 학생이 사전 찾아야 하는 수준.**
  - **syns[0]: 수능 상위~편입 입문** (예: difficult→demanding)
  - **syns[1]: 편입 중급** (예: difficult→onerous)
  - **syns[2]: 편입 고급~GRE** (예: difficult→arduous)
  - **syns[3]: GRE/희귀어휘 또는 2번째 뜻** (예: difficult→intractable, formidable)

★★ 절대 규칙 — 위반 시 FAIL ★★
1. **쉬운 단어(중학~수능 기본) 절대 금지**: hard, big, good, easy, fast, large, small, main, important 같은 건 syns에 넣지 마라. 이런 건 학생이 이미 안다 → 추가 자료로 줄 가치 없음.
2. **편입영어·GRE 수준 또는 다의어의 잘 안 쓰이는 2·3번째 뜻을 활용하라.**
   - 예: "interest" → syns에 "stake(이해관계)", "dividend(배당)" 같은 2번뜻
   - 예: "address" → "tackle(다루다)", "broach(꺼내다)" 같은 고급 의미
3. 학생이 "어? 이 단어가 이런 뜻도 있어?" 하고 놀랄 만한 어휘여야 한다.

[O] 좋은 예시 (전부 고난도):
{"word": "increase", "ko": "증가하다",
 "syns": [["augment","증대시키다"], ["escalate","고조되다"], ["proliferate","급증하다"], ["burgeon","급성장하다"]]}
{"word": "important", "ko": "중요한",
 "syns": [["pivotal","중추적인"], ["paramount","가장 중요한"], ["salient","두드러진"], ["cardinal","기본적인"]]}

[X] 절대 금지 (너무 쉬움):
{"word": "alternative", "syns": [["substitute","대체물"], ["replacement","교체물"], ["option","선택"], ["choice","선택"]]}
 → option, choice는 중학생도 안다 ❌❌❌ 이런 거 주지 마라

- **`def_en`**: English 영영풀이, 8~15 words. (메인 단어 바로 아래 배치됨)
- **`ex_short`**: 짧은 예문, 8~12 words.

### `theme_vocab` (PAGE C 하단, 6 items)
동일하게 편입/GRE 수준. 쉬운 동의어 금지.

═════════════════════════════════════════════════
## JSON SCHEMA (return EXACTLY this structure)
═════════════════════════════════════════════════
{
  "passage_marked": "...",
  "topics": [3 SHORT phrases, **6~10 words each, MAX 10 words, ≤ 100 chars**. Same length/weight as titles. Abstract noun-phrase or compressed proposition. NEVER long explanatory propositions],
  "titles": [3 titles, **6~10 words each, MAX 10 words**. Colon style allowed],
  "topics_kr": [3 natural Korean, **≤ 15 자 each**, same length as titles_kr],
  "titles_kr": [3 natural Korean titles, **≤ 15 자 each**],
  "grammar_notes": [{"num":"①","tag":"...","desc":"..."}],
  "vocab_notes": [{"letter":"ⓐ","word":"...","syns":[...],"ants":[...]}],
  "grammar_p1": {
    "left": [{"num":"①","title":"...","lines":[
       {"text":"전체 어법 핵심 원리를 한 줄로 설명 (예: 'to부정사가 동사 뒤에서 명사 역할')","is_ex":false},
       {"text":"[패턴] 구체적 공식 (예: 'V + to-V = ~할 것을 V하다')","is_ex":false},
       {"text":"본문 적용 또는 [O] 예문 (영어 문장)","is_ex":true},
       {"text":"★ 교체(+) 변형/대체 표현","is_ex":false},
       {"text":"cf. 혼동되는 비교 포인트","is_ex":false}
    ]}],
    "right": [3~4 boxes]
  },
  "grammar_p2": {"left":[3~4 boxes],"right":[3~4 boxes]},
  "implication_box": {
    "flow_4panel": [
      {"stage": "도입", "label_en": "Setup", "text_kr": "글이 어떤 상황·문제로 시작되는지. **1~2문장, 50~80자 (절대 80자 초과 금지)**, ~해요체. 박스에 들어가야 하므로 짧고 강렬하게.", "key_phrase": "본문의 이 단계 핵심 영어 (3~6단어)"},
      {"stage": "전개", "label_en": "Development", "text_kr": "어떤 논리·예시로 발전. **1~2문장, 50~80자**, ~해요체", "key_phrase": "본문 핵심 (3~6단어)"},
      {"stage": "전환", "label_en": "Turn", "text_kr": "어디서 방향이 바뀌는지 — 역접·대조·강조. **1~2문장, 50~80자**, ~해요체", "key_phrase": "전환점 영어 표현 (Instead/However 등 포함, 3~6단어)"},
      {"stage": "결론", "label_en": "Conclusion", "text_kr": "글쓴이의 최종 입장·주장. **1~2문장, 50~80자**, ~해요체", "key_phrase": "결론부 핵심 (3~6단어)"}
    ],
    "blank_candidates": [
      "★ 정확히 3개만 (절대 3개 초과 금지) ★",
      {"sentence": "본문에서 빈칸으로 출제하기 좋은 중요 문장 (원문 그대로, 영어)", "paraphrase": "이 문장이 빈칸/변형으로 출제될 때 paraphrasing 될 수 있는 형태 (영어)", "why": "왜 이 문장이 빈칸 출제 포인트인지 한 줄 (한국어)"},
      {"sentence": "두 번째 중요 문장", "paraphrase": "패러프레이즈", "why": "이유"},
      {"sentence": "세 번째 중요 문장", "paraphrase": "패러프레이즈", "why": "이유"}
    ],
    "implicit_meanings": [
      "★ 정확히 3개만 (절대 3개 초과 금지) ★",
      {"expr": "본문의 비유적·함축적 표현 (영어 원문)", "literal": "표면적 의미 (한국어)", "implied": "글에서 빗댄/함축한 진짜 의미 (한국어)", "implied_en": "그 함축 의미를 영어로 다시 표현 (paraphrase, 영어 한 문장)"},
      {"expr": "두 번째 함축 표현", "literal": "표면 의미", "implied": "함축 의미", "implied_en": "영어 paraphrase"},
      {"expr": "세 번째 함축 표현", "literal": "표면 의미", "implied": "함축 의미", "implied_en": "영어 paraphrase"}
    ],
    "summary": {
      "en": "본문 전체를 정확히 40 단어(35~45 허용)로 요약한 영어 한 단락. 주제+핵심논거+결론 포함. 학술적이되 명료하게.",
      "kr": "위 영어 요약의 자연스러운 한국어 번역 (한 단락)"
    },
    "theme_summary": "주제를 응축한 2~3문장 (이전 1문장 → 2배로 확대). 글의 핵심 주장 + 그것이 왜 중요한지 + 대비/함의까지. 80~120자 한국어."
  },
  "vocab_detail": {
    "left": [{"letter":"ⓐ","word":"...","ko":"...","syns":[4],"ants":[3],"def_en":"...","ex_short":"..."}],
    "right": [3 items]
  },
  "theme_vocab": {
    "left": [{"word":"...","ko":"...","syns":[2~3],"def_en":"...","ex_short":"..."}],
    "right": [3 items]
  }
}

═════════════════════════════════════════════════
## SELF-CHECK BEFORE RETURNING
═════════════════════════════════════════════════
□ passage_marked stripped of [[...]] = input character-by-character?
□ Every [[...]] marker wraps 1~3 words only?
□ Every box's num is "X" or "XY" (max 2 merged)?
□ grammar_p1.left/right ≥ 3 AND grammar_p2.left/right ≥ 3?
□ Count of [[GRAMMAR]] = len(grammar_notes)?
□ **Count of [[VOCAB]] in passage_marked = len(vocab_notes) = 10?** ★MOST MISSED CHECK★
□ **Count of [[IMPL]] in passage_marked = len(implication_box.expressions) = 6~7?** ★MOST MISSED CHECK★
□ Every vocab_detail/theme_vocab has def_en (8~15w) AND ex_short (8~12w)?
□ **All 3 topics ≤ 10 words AND ≤ 100 chars?** ★ABSOLUTE — RE-WRITE IF VIOLATED★
□ **All 3 topics_kr ≤ 15 chars (Korean)?** ★ABSOLUTE — RE-WRITE IF VIOLATED★
□ **topics와 titles 길이가 한 눈에 비슷한가?** (한쪽이 다른쪽 1.5배 이상이면 FAIL)

★★★ MARKER COMPLETENESS — DO NOT SKIP ★★★
파란 밑줄(VOCAB)과 검은 밑줄(IMPL)은 빨간 밑줄(GRAMMAR)만큼 중요하다.
vocab_notes 10개를 만들었으면 본문에 VOCAB 마커도 정확히 10개 있어야 한다.
implication_box.expressions 6~7개를 만들었으면 본문에 IMPL 마커도 정확히 6~7개 있어야 한다.
**만약 본문에 마커가 부족하면 → 단어/표현을 본문에서 다시 찾아서 마커 추가 → 재검토.**

★★★ FINAL CHECK — GRAMMAR_NOTES COUNT ★★★
□ len(grammar_notes) >= 12?
   If < 12: RE-SCAN. Common missed points:
   • prevent/keep/stop A from V-ing
   • to-V 부사적 용법 (목적: "in order to V")
   • 현재완료 수동 (have been + p.p.)
   • 형용사적 to-V (N + to-V)
   • be expected to V / be likely to V
   • 분사구문 (V-ing... = 접속사 + S + V)
   • 부분표현 + V 수일치 (most/all of N + V)
   • such as (예시 표현)
   • 관계대명사 계속적 용법 (, which)
   • 동격 of (the city of Seoul, the idea of V-ing)

If still < 12 after re-scan, BREAK COMPOUND structures into separate notes:
- "have been successfully turned into" → 1) 현재완료 수동 + 2) turn A into B
- "preventing food from sticking to paper" → 1) 분사구문 + 2) prevent A from V-ing

Return ONLY the JSON object.
"""


def generate_preclass_analysis(passage: str, passage_dir: Path, translation: str = "") -> dict:
    """0회독 — 수업 전 4페이지 완전 분석 (선생님 본인 수업 준비용)."""
    # v21: 캐시 로드 후 어휘 마커 보정 마이그레이션 추가
    cached = load_step(passage_dir, "preclass_analysis_v24")
    if cached:
        _safe_print("  ✅ preclass_analysis_v24 캐시 사용")
        # ★ v24 마이그레이션 — 캐시도 항상 지문↔상세 동기화 실행
        # 옛 캐시에 지문 마커 3개 / 상세 12개 같은 불일치 있을 수 있음
        try:
            pm = cached.get("passage_marked", "") or ""
            vn = cached.get("vocab_notes", []) or []
            vocab_count_in_passage = pm.count("[[VOCAB:l=")
            expected_count = len([v for v in vn if isinstance(v, dict) and v.get("word")])
            if expected_count > 0 and vocab_count_in_passage < expected_count:
                _safe_print(f"  🔵 캐시 어휘 마커 부족 ({vocab_count_in_passage}/{expected_count}) — 후처리 재실행")
                cached = _filter_bad_vocab(cached)
                cached["passage_marked"] = _ensure_vocab_markers(
                    cached.get("passage_marked", ""),
                    cached.get("vocab_notes", []),
                )
            
            # ★ v24 핵심 — 지문↔상세 동기화 (항상 실행, 옛 캐시도 보정됨)
            cached = _sync_grammar_boxes_with_passage(cached)
            cached = _sync_vocab_with_passage(cached)
            
            cached["passage_html"] = _passage_marked_to_html(
                cached.get("passage_marked", ""), passage
            )
            # 마이그레이션 결과 다시 저장
            save_step(passage_dir, "preclass_analysis_v24", cached)
            _safe_print(f"  ✅ 캐시 마이그레이션 완료")
        except Exception as e:
            import traceback as _tb_mig
            _safe_print(f"  ⚠️ 캐시 마이그레이션 실패: {type(e).__name__}: {e}")
            _safe_print(f"     {_tb_mig.format_exc()}")
        return cached
    _safe_print("  📕 preclass_analysis 생성 중...")

    prompt = f"Analyze this passage for Korean high-school exam preparation (Level-MeUp 4-page pre-class note):\n\n{passage}"
    if translation:
        prompt += f"\n\n[Reference Korean translation - use this to ensure accurate Korean translations]:\n{translation}"

    # v4: Supabase grammar_points 자동 fetch → 시스템 프롬프트 동적 주입
    # 한국 수능·내신 빈출 어법 + 김대균 영문법 정제 데이터를 매 호출마다 적용
    sys_prompt = SYS_PRECLASS_ANALYSIS
    grammar_addendum = _load_grammar_points_for_prompt()
    if grammar_addendum:
        sys_prompt = sys_prompt + "\n\n" + grammar_addendum
        _safe_print(f"  📚 grammar_points 주입됨 ({len(grammar_addendum):,} chars)")

    # 출력이 큰 스키마 — max_tokens 넉넉히
    data = call_claude_json(sys_prompt, prompt, max_tokens=16000)
    
    # ★ 후처리 안전 실행 헬퍼 — 에러 시 함수명·에러내용 로깅하고 다음 단계로
    def _safe_post(step_name, fn):
        try:
            return fn()
        except Exception as e:
            import traceback as _tb
            _safe_print(f"  ⚠️ [후처리 실패] {step_name}: {type(e).__name__}: {str(e)[:200]}")
            _safe_print(f"     {_tb.format_exc().splitlines()[-3] if _tb.format_exc().splitlines() else ''}")
            return None
    
    # 후처리 1 — 박스 균형 자동 보정
    _r = _safe_post("_rebalance_grammar_boxes", lambda: _rebalance_grammar_boxes(data))
    if _r is not None: data = _r
    
    # ★ v14 후처리 1.5 — 빈칸/함축 정확히 3개로 제한
    _r = _safe_post("_enforce_count_3", lambda: _enforce_count_3(data))
    if _r is not None: data = _r
    
    # ★ v6 후처리 2 — 가주어/가목적어/It-cleft 패턴
    _r = _safe_post("_inject_dummy_it_notes", lambda: _inject_dummy_it_notes(data, passage))
    if _r is not None: data = _r
    
    # ★ v6 후처리 3 — IMPL 마커 본문 자동 보정
    try:
        impl_box = data.get("implication_box", {}) or {}
        impl_sources = []
        for im in impl_box.get("implicit_meanings", []) or []:
            if isinstance(im, dict) and im.get("expr"):
                impl_sources.append({"expr": im["expr"]})
        for bc in impl_box.get("blank_candidates", []) or []:
            if isinstance(bc, dict) and bc.get("sentence"):
                impl_sources.append({"expr": bc["sentence"]})
        data["passage_marked"] = _ensure_implication_markers(
            data.get("passage_marked", ""),
            impl_sources,
        )
    except Exception as e:
        _safe_print(f"  ⚠️ [후처리 실패] _ensure_implication_markers: {type(e).__name__}: {str(e)[:200]}")

    # ★ v16 후처리 3.5 — vocab_notes 부적절 어휘 제거
    _r = _safe_post("_filter_bad_vocab", lambda: _filter_bad_vocab(data))
    if _r is not None: data = _r

    # ★ v6 후처리 4 — VOCAB 마커 본문 자동 보정
    try:
        data["passage_marked"] = _ensure_vocab_markers(
            data.get("passage_marked", ""),
            data.get("vocab_notes", []),
        )
    except Exception as e:
        _safe_print(f"  ⚠️ [후처리 실패] _ensure_vocab_markers: {type(e).__name__}: {str(e)[:200]}")

    # ★ v24 후처리 4.5 — 지문 마커 ↔ 상세 박스 동기화
    # 선생님 지적: 지문 어법 ③번까지인데 상세 박스 ⑫번까지 나오는 문제
    # 지문에 실제 마킹된 번호만 살리고 그 외 상세 박스 제거
    _r = _safe_post("_sync_grammar_boxes_with_passage", lambda: _sync_grammar_boxes_with_passage(data))
    if _r is not None: data = _r
    _r = _safe_post("_sync_vocab_with_passage", lambda: _sync_vocab_with_passage(data))
    if _r is not None: data = _r

    # 후처리 5 — passage_marked → passage_html 변환
    try:
        data["passage_html"] = _passage_marked_to_html(data.get("passage_marked", ""), passage)
    except Exception as e:
        _safe_print(f"  ⚠️ [후처리 실패] _passage_marked_to_html: {type(e).__name__}: {str(e)[:200]}")
        # 최소한 plain text라도
        import html as _html_fallback
        import re as _re_fallback
        pm = data.get("passage_marked", "") or ""
        plain = _re_fallback.sub(r'\[\[/?(?:GRAMMAR|VOCAB|IMPL)[^\]]*\]\]', '', pm)
        data["passage_html"] = _html_fallback.escape(plain)

    # ★ v9 후처리 6 — topics 길이 검증
    _r = _safe_post("_check_topics_length", lambda: _check_topics_length(data))
    if _r is not None: data = _r

    # ★ v13 후처리 7 — 요약문 40단어 강제
    _r = _safe_post("_enforce_summary_length", lambda: _enforce_summary_length(data, max_words=40))
    if _r is not None: data = _r

    save_step(passage_dir, "preclass_analysis_v24", data)
    return data


def _load_grammar_points_for_prompt() -> str:
    """Supabase grammar_points 테이블에서 활성 어법 포인트를 fetch하여
    시스템 프롬프트에 부착할 텍스트로 변환.
    
    - 우선순위 5(★★★★★)부터 3(★★★)까지 모두 포함
    - 카테고리별로 그룹화
    - 실패 시 빈 문자열 반환 (안전)
    """
    try:
        import httpx
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            return ""
        
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        # priority DESC, category 순으로 가져오기. priority >= 3 만
        endpoint = (
            f"{url}/rest/v1/grammar_points"
            "?active=eq.true&priority=gte.3"
            "&select=category,subcategory,priority,name,pattern,example_good,example_bad,why_important,trap_warning,prohibited_analysis,trigger_keywords,notes"
            "&order=priority.desc,category.asc"
        )
        with httpx.Client(timeout=10) as client:
            resp = client.get(endpoint, headers=headers)
        if resp.status_code != 200:
            return ""
        rows = resp.json()
        if not isinstance(rows, list) or len(rows) == 0:
            return ""
    except Exception as e:
        _safe_print(f"  ⚠️ grammar_points fetch 실패: {str(e)[:100]}")
        return ""
    
    # 카테고리별로 그룹화
    by_cat: dict = {}
    for r in rows:
        cat = r.get("category", "기타")
        by_cat.setdefault(cat, []).append(r)
    
    # 시스템 프롬프트 부착용 텍스트 생성
    lines = [
        "═" * 60,
        "## 📚 한국 수능·내신 빈출 어법 함정 (반드시 우선 적용)",
        "═" * 60,
        "",
        "본문 분석 시 다음 어법 포인트를 우선적으로 발견하라.",
        "특히 '★ trap_warning'과 '🚫 prohibited_analysis'는 절대 위반 금지.",
        "지문에 trigger_keywords가 보이면 즉시 해당 어법으로 분석할 것.",
        "",
    ]
    star_map = {5: "★★★★★", 4: "★★★★", 3: "★★★", 2: "★★", 1: "★"}
    
    for cat, items in by_cat.items():
        lines.append(f"### [{cat}]")
        for it in items:
            star = star_map.get(it.get("priority", 3), "★★★")
            name = it.get("name", "")
            pattern = it.get("pattern") or ""
            ex_good = it.get("example_good") or ""
            ex_bad = it.get("example_bad") or ""
            trap = it.get("trap_warning") or ""
            prohibited = it.get("prohibited_analysis") or ""
            keywords = it.get("trigger_keywords") or []
            notes = it.get("notes") or ""
            
            lines.append(f"- **{star} {name}**")
            if pattern:
                lines.append(f"  - 패턴: {pattern}")
            if ex_good:
                lines.append(f"  - [O] {ex_good}")
            if ex_bad:
                lines.append(f"  - [X] {ex_bad}")
            if trap:
                lines.append(f"  - ⚠️ {trap}")
            if prohibited:
                lines.append(f"  - 🚫 금지: {prohibited}")
            if keywords:
                kw_str = ", ".join(keywords[:8])
                lines.append(f"  - 🔑 trigger: {kw_str}")
            if notes:
                lines.append(f"  - 💡 {notes}")
        lines.append("")
    
    lines.extend([
        "═" * 60,
        "## 적용 규칙 (RECAP)",
        "═" * 60,
        "1. 위 어법 중 본문 trigger_keywords에 해당하는 것은 무조건 grammar_notes에 포함",
        "2. prohibited_analysis 위반 절대 금지 (잘못된 해설 방지)",
        "3. 생략구문 분석 시 생략된 부분을 [I was], [it is], [have] 같이 대괄호로 명시",
        "4. 의미상 주어 + 동명사 패턴 (someone V-ing)은 단순 분사가 아닌 동명사로 분석",
        "5. 한국 수능·내신 빈출 어법 (도치/의미상주어/대동사/조동사 등) 우선 포함",
    ])
    
    return "\n".join(lines)


def _split_sentences_for_ordering(passage: str) -> list:
    """지문을 자연스러운 문장 단위로 분리.
    
    영어 마침표/물음표/느낌표 + 따옴표 처리. 약어(Mr. Dr. e.g. 등)는 분리 안 함.
    너무 짧은 조각(<= 6단어)은 직전 문장과 합침.
    """
    import re as _re
    text = (passage or "").strip()
    if not text:
        return []
    
    # 약어 보호 (마침표를 임시 토큰으로)
    ABBR = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "e.g.", "i.e.", "etc.", "vs.", "U.S.", "U.K.", "U.S.A.",
        "Inc.", "Ltd.", "Co.", "Corp.", "No.", "Vol.",
    ]
    placeholder = "§"
    protected = text
    for ab in ABBR:
        protected = protected.replace(ab, ab.replace(".", placeholder))
    
    # 문장 경계: 마침표/물음표/느낌표 + 공백 + 대문자
    parts = _re.split(r'(?<=[.!?])(?=[\s"\'])\s*(?=["\']?[A-Z])', protected)
    parts = [p.replace(placeholder, ".").strip() for p in parts if p.strip()]
    
    # 너무 짧은 단편만 앞 문장과 합치기
    # - 3단어 이하 AND 마침표로 끝나지 않으면 (불완전 단편) → 병합
    # - 5단어짜리 완결 문장은 독립 문장으로 인정 (병합 안 함)
    merged = []
    for p in parts:
        wc = len(p.split())
        is_fragment = (wc <= 3) and (not p.rstrip().endswith(('.', '!', '?', '"', "'")))
        if merged and is_fragment:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def generate_sentence_order(passage: str, passage_id: str = "") -> dict:
    """지문을 문장 단위로 분리·랜덤 섞기 → 순서배열 문제 데이터 생성.
    
    Returns:
        {"shuffled":[{"label":"(A)","text":"..."}], "answer":["(B)","(D)",...], "n":6, "skipped":False}
    
    안전 보장:
    1. 셔플 결과는 절대 원본 순서와 동일하지 않음 (최후엔 강제 스왑)
    2. 첫 문장이 (A)인 평이한 결과도 회피 (학생이 의심하기 좋은 패턴)
    3. passage_id + 본문 해시로 시드 견고화 → 본문 다르면 시드 무조건 다름
    """
    import random as _rand
    import hashlib as _hash
    
    sents = _split_sentences_for_ordering(passage)
    n = len(sents)
    if n < 3:
        return {"shuffled": [], "answer": [], "n": 0, "skipped": True}
    if n > 10:
        sents = sents[:10]
        n = 10
    
    # ★ 시드 견고화: passage_id가 같아도 본문이 다르면 다른 시드
    # passage_id만 쓰면 빈 문자열/같은 문자열로 인한 충돌 가능
    body_hash = _hash.md5(passage.encode("utf-8")).hexdigest()[:12]
    seed_src = f"{passage_id}::{body_hash}"
    rng = _rand.Random(seed_src)
    
    LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]
    original = list(range(n))
    
    # ★ 셔플 시도 최대 20회 — 다음 조건 모두 통과해야 채택:
    #   조건1: 원본 순서와 동일하지 않을 것 [0,1,2,3,4,5] 금지
    #   조건2: 첫 문장이 원본 첫 문장이 아닐 것 (학생이 의심하기 쉬운 패턴)
    #   조건3: 최소 절반 이상이 원래 위치가 아닐 것 (너무 약한 셔플 회피)
    shuffled_indices = None
    for attempt in range(20):
        candidate = original.copy()
        rng.shuffle(candidate)
        if candidate == original:
            continue
        # 조건2: 첫 위치가 원본 0번이면 약한 셔플 — 회피
        if candidate[0] == 0:
            continue
        # 조건3: 원래 위치와 일치하는 문장 수가 절반 초과면 약함
        same_count = sum(1 for i in range(n) if candidate[i] == i)
        if same_count > n // 2:
            continue
        shuffled_indices = candidate
        break
    
    # ★ 최후 안전망: 위 조건을 만족 못 했으면 인위적으로 만든다
    if shuffled_indices is None:
        # 원본 reverse → 이건 100% 원본과 다르고 첫 위치도 0이 아님
        shuffled_indices = list(reversed(original))
        # 그 다음 rng로 한 번 더 추가 셔플 (단조로움 방지) — 단 원본과 다른지 재검증
        for _ in range(10):
            candidate = shuffled_indices.copy()
            rng.shuffle(candidate)
            if candidate != original and candidate[0] != 0:
                shuffled_indices = candidate
                break
        # 그래도 안 되면(극희박) 인접 두 위치 강제 스왑
        if shuffled_indices == original or shuffled_indices[0] == 0:
            shuffled_indices = original.copy()
            shuffled_indices[0], shuffled_indices[-1] = shuffled_indices[-1], shuffled_indices[0]
    
    shuffled = [
        {"label": LABELS[i], "text": sents[shuffled_indices[i]]}
        for i in range(n)
    ]
    
    answer = []
    for orig_idx in range(n):
        pos = shuffled_indices.index(orig_idx)
        answer.append(LABELS[pos])
    
    return {"shuffled": shuffled, "answer": answer, "n": n, "skipped": False}


def render_preclass_analysis(passages_data: list, school_name: str,
                              include_sentence_order: bool = False,
                              sentence_order_only: bool = False) -> str:
    """0회독 HTML 렌더링 (preclass_analysis.html 템플릿 사용).
    
    Args:
        passages_data: 본문 데이터 리스트 (각 item에 label, data 필드)
        school_name: 학교명
        include_sentence_order: True면 0회독 자료 끝에 순서배열 문제·정답지 추가
        sentence_order_only: True면 0회독 자료 생략하고 순서배열만 출력
                              (자동으로 include_sentence_order=True 처리)
    
    호출 패턴:
        # 0회독만
        render_preclass_analysis(data, school)
        
        # 0회독 + 순서배열 (한 파일에 합쳐서)
        render_preclass_analysis(data, school, include_sentence_order=True)
        
        # 순서배열만 (별도 파일)
        render_preclass_analysis(data, school, sentence_order_only=True)
    """
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    tmpl = env.get_template("preclass_analysis.html")
    
    # sentence_order_only면 include_sentence_order도 자동 True
    if sentence_order_only:
        include_sentence_order = True
    
    # 순서배열 옵션이 켜져 있으면 각 본문에 sentence_order 데이터 부착
    if include_sentence_order:
        import re as _re_so
        # ★ 정답 중복 회피용 추적
        used_answers = []  # 이미 사용된 answer 튜플 모음
        
        for idx, item in enumerate(passages_data):
            data = item.get("data", {}) or {}
            
            # ★ v23 핵심 수정: Supabase 원문(item["passage"])을 최우선 사용
            # 이유: 0회독 자료(data.passage_marked)는 AI가 마커 단 텍스트라 변형 가능성 있음
            #      - 일부 문장이 누락되거나
            #      - 키워드 메모가 섞이거나  
            #      - 원문에 없는 문장이 추가되는 사고 발생
            # main.py에서 Supabase passages 테이블의 원본을 item["passage"]에 그대로 저장하므로
            # 이걸 직접 쓰는 게 100% 안전
            passage_raw = (item.get("passage") or "").strip()
            
            # 백업1: item["passage"]가 비어있을 때만 data.passage / passage_text
            if not passage_raw:
                passage_raw = (data.get("passage") or data.get("passage_text") or "").strip()
            
            # 백업2: 둘 다 없으면 마지막 수단으로 passage_marked에서 마커 제거
            if not passage_raw:
                marked = data.get("passage_marked", "") or ""
                if marked:
                    passage_raw = _re_so.sub(r'\[\[/?(?:GRAMMAR|VOCAB|IMPL)[^\]]*\]\]', '', marked)
                    passage_raw = _re_so.sub(r'\s+', ' ', passage_raw).strip()
            
            # 학생 보기용 — 위첨자(①~⑳, ⓐ~ⓩ) 제거 + HTML 엔티티 정리
            passage_clean = _re_so.sub(r'[\u2460-\u2473\u24B6-\u24E9]', '', passage_raw)
            passage_clean = passage_clean.replace('&#x27;', "'").replace('&quot;', '"').replace('&amp;', '&')
            # ###해석### 안전망 — 혹시 섞여있으면 영어만
            if "###해석###" in passage_clean:
                passage_clean = passage_clean.split("###해석###", 1)[0]
            passage_clean = _re_so.sub(r'\s+', ' ', passage_clean).strip()
            if not passage_clean:
                continue
            
            base_pid = item.get("label", "") + ":" + (item.get("passage_id","") or "")
            
            # ★ 정답 중복이면 시드 변경하면서 재생성 (최대 10회)
            so_result = None
            for retry in range(10):
                pid = f"{base_pid}::r{retry}" if retry > 0 else base_pid
                so_result = generate_sentence_order(passage_clean, passage_id=pid)
                # skipped거나 본문 너무 짧으면 그대로 사용
                if so_result.get("skipped") or so_result.get("n", 0) < 3:
                    break
                ans_tuple = tuple(so_result.get("answer", []))
                if ans_tuple not in used_answers:
                    used_answers.append(ans_tuple)
                    break
                # 중복이면 다른 시드로 재시도
            
            item["sentence_order"] = so_result
    
    return tmpl.render(
        passages=passages_data,
        school_name=school_name,
        include_sentence_order=include_sentence_order,
        sentence_order_only=sentence_order_only,
    )


def _sync_grammar_boxes_with_passage(data: dict) -> dict:
    """지문(passage_marked)의 어법 마커 개수에 맞춰 grammar_p1/p2 상세 박스를 잘라낸다.
    
    문제: AI가 지문에는 ①②③만 마킹하고, grammar_p1/p2에는 ①~⑫까지 12개 박스를 만드는 경우.
    결과: 지문 마커는 3개인데 상세 설명은 12개 — 학생 혼란.
    
    해결: 지문에 실제 존재하는 마커 번호만 살리고, 그 외 박스는 제거.
    """
    if not isinstance(data, dict):
        return data
    
    import re as _re
    
    passage_marked = data.get("passage_marked", "") or ""
    if not passage_marked:
        return data
    
    # 지문에 실제 존재하는 어법 번호 추출
    # [[GRAMMAR:n=①]] 또는 [[GRAMMAR:n=1]] 또는 [[GRAMMAR:n=①,split=1]]
    grammar_nums_in_passage = set()
    for m in _re.finditer(r'\[\[GRAMMAR:n=([^,\]]+)', passage_marked):
        raw_num = m.group(1).strip()
        # 숫자면 원숫자로 변환 (예외 안전)
        try:
            if raw_num.isdigit():
                n_int = int(raw_num)
                if 1 <= n_int <= 20:
                    if n_int <= 10:
                        raw_num = chr(0x2460 + n_int - 1)
                    elif n_int <= 15:
                        raw_num = "⑪⑫⑬⑭⑮"[n_int - 11]
                    else:
                        raw_num = "⑯⑰⑱⑲⑳"[n_int - 16]
        except (ValueError, IndexError):
            pass
        grammar_nums_in_passage.add(raw_num)
    
    if not grammar_nums_in_passage:
        return data
    
    CIRCLE_ORDER = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    
    # grammar_p1, grammar_p2의 모든 박스 모으기
    p1 = data.get("grammar_p1", {}) or {}
    p2 = data.get("grammar_p2", {}) or {}
    all_boxes = (
        (p1.get("left", []) or []) + (p1.get("right", []) or [])
        + (p2.get("left", []) or []) + (p2.get("right", []) or [])
    )
    
    # 지문에 있는 번호의 박스만 남김 + 번호 순으로 정렬
    filtered = []
    seen_nums = set()
    for box in all_boxes:
        if not isinstance(box, dict):
            continue
        num = (box.get("num") or "").strip()
        # 박스 num에 포함된 원숫자 중 지문에 있는 것만
        box_nums = [c for c in num if c in CIRCLE_ORDER]
        if not box_nums:
            continue
        # 첫 번호가 지문에 있어야 박스 채택
        first_num = box_nums[0]
        if first_num in grammar_nums_in_passage and first_num not in seen_nums:
            filtered.append(box)
            seen_nums.add(first_num)
    
    # 번호 순으로 정렬
    def _sort_key(box):
        num = (box.get("num") or "").strip()
        for i, c in enumerate(CIRCLE_ORDER):
            if c in num:
                return i
        return 999
    filtered.sort(key=_sort_key)
    
    removed_count = len(all_boxes) - len(filtered)
    if removed_count > 0:
        _safe_print(f"  📕 지문 마커 동기화: {len(all_boxes)}개 → {len(filtered)}개 (불일치 {removed_count}개 제거)")
    
    # 그리고 grammar_notes도 같이 정리
    grammar_notes = data.get("grammar_notes", []) or []
    filtered_notes = []
    for note in grammar_notes:
        if not isinstance(note, dict):
            continue
        num = (note.get("num") or "").strip()
        note_nums = [c for c in num if c in CIRCLE_ORDER]
        if not note_nums:
            continue
        if note_nums[0] in grammar_nums_in_passage:
            filtered_notes.append(note)
    filtered_notes.sort(key=_sort_key)
    data["grammar_notes"] = filtered_notes
    
    # 박스 4분면 재분배
    n = len(filtered)
    if n == 0:
        data["grammar_p1"] = {"left": [], "right": []}
        data["grammar_p2"] = {"left": [], "right": []}
        return data
    
    q = n // 4
    r = n % 4
    slots = [q + (1 if i < r else 0) for i in range(4)]
    idx = 0
    new_p1_l = filtered[idx:idx+slots[0]]; idx += slots[0]
    new_p1_r = filtered[idx:idx+slots[1]]; idx += slots[1]
    new_p2_l = filtered[idx:idx+slots[2]]; idx += slots[2]
    new_p2_r = filtered[idx:idx+slots[3]]
    data["grammar_p1"] = {"left": new_p1_l, "right": new_p1_r}
    data["grammar_p2"] = {"left": new_p2_l, "right": new_p2_r}
    
    return data


def _sync_vocab_with_passage(data: dict) -> dict:
    """지문(passage_marked)의 어휘 마커 letter에 맞춰 vocab_notes를 잘라낸다.
    
    같은 원리 — 지문에 ⓐⓑⓒ만 있는데 vocab_notes에 ⓐ~ⓙ까지 10개면 ⓓⓔ... 제거.
    """
    if not isinstance(data, dict):
        return data
    
    import re as _re
    
    passage_marked = data.get("passage_marked", "") or ""
    if not passage_marked:
        return data
    
    CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"
    
    # 지문에 실제 존재하는 어휘 letter 추출
    vocab_letters_in_passage = set()
    for m in _re.finditer(r'\[\[VOCAB:l=([^\]]+)\]\]', passage_marked):
        raw_letter = m.group(1).strip()
        # 알파벳이면 원알파벳으로 변환
        if len(raw_letter) == 1 and raw_letter.isalpha():
            idx_l = ord(raw_letter.lower()) - ord('a')
            if 0 <= idx_l < len(CIRCLED_ALPHA):
                raw_letter = CIRCLED_ALPHA[idx_l]
        vocab_letters_in_passage.add(raw_letter)
    
    if not vocab_letters_in_passage:
        return data
    
    # vocab_notes에서 지문에 있는 letter만 남김
    vocab_notes = data.get("vocab_notes", []) or []
    filtered = []
    for v in vocab_notes:
        if not isinstance(v, dict):
            continue
        letter = (v.get("letter") or "").strip()
        if letter and letter[0] in vocab_letters_in_passage:
            filtered.append(v)
    
    removed_count = len(vocab_notes) - len(filtered)
    if removed_count > 0:
        _safe_print(f"  🔵 지문 어휘 마커 동기화: {len(vocab_notes)}개 → {len(filtered)}개 (불일치 {removed_count}개 제거)")
    
    data["vocab_notes"] = filtered
    return data


def _rebalance_grammar_boxes(data: dict) -> dict:
    """grammar_p1/p2 박스가 4분면에 균등 분배되도록 보정 + num 순서로 정렬.
    
    ★ v18 추가: AI가 박스를 묶은 경우(예: num="④⑤⑥") 자동으로 분리.
    """
    try:
        p1 = data.get("grammar_p1", {}) or {}
        p2 = data.get("grammar_p2", {}) or {}
        all_boxes = (
            (p1.get("left", []) or []) + (p1.get("right", []) or [])
            + (p2.get("left", []) or []) + (p2.get("right", []) or [])
        )
        
        # ★ 묶인 박스 자동 분리 (num="④⑤⑥" → 박스 3개로 쪼개기)
        import re as _re
        CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
        unrolled = []
        split_count = 0
        for box in all_boxes:
            if not isinstance(box, dict):
                unrolled.append(box)
                continue
            num = (box.get("num") or "").strip()
            # 원숫자 모두 추출
            nums_found = [c for c in num if c in CIRCLED_NUMS]
            if len(nums_found) >= 2:
                # 여러 번호 묶임 → 각각의 박스로 복제
                for nf in nums_found:
                    new_box = dict(box)
                    new_box["num"] = nf
                    unrolled.append(new_box)
                split_count += len(nums_found) - 1
            else:
                unrolled.append(box)
        
        if split_count > 0:
            _safe_print(f"  📕 묶인 어법 박스 자동 분리: {split_count}개 추가")
        
        all_boxes = unrolled
        n = len(all_boxes)
        if n < 4:
            return data

        # num 기준 정렬 — "①②"처럼 합쳐진 경우 첫 번째 마크 기준
        CIRCLE_ORDER = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
        def _sort_key(box):
            num = (box.get("num") or "").strip()
            for i, c in enumerate(CIRCLE_ORDER):
                if c in num:
                    return i
            return 999
        all_boxes_sorted = sorted(all_boxes, key=_sort_key)

        # 12개 이상이면 균등 분배 (3+3+3+3 또는 3+4+3+3)
        if n >= 12:
            q = n // 4
            r = n % 4
            slots = [q + (1 if i < r else 0) for i in range(4)]
        else:
            # 12 미만이면 기존 분배 유지하되 num 순서로만 재정렬
            slots = [
                len(p1.get("left", [])) or 1,
                len(p1.get("right", [])) or 1,
                len(p2.get("left", [])) or 1,
                len(p2.get("right", [])) or 1,
            ]
            total = sum(slots)
            if total != n:
                q = n // 4; r = n % 4
                slots = [q + (1 if i < r else 0) for i in range(4)]

        idx = 0
        new_p1_l = all_boxes_sorted[idx:idx+slots[0]]; idx += slots[0]
        new_p1_r = all_boxes_sorted[idx:idx+slots[1]]; idx += slots[1]
        new_p2_l = all_boxes_sorted[idx:idx+slots[2]]; idx += slots[2]
        new_p2_r = all_boxes_sorted[idx:idx+slots[3]]
        data["grammar_p1"] = {"left": new_p1_l, "right": new_p1_r}
        data["grammar_p2"] = {"left": new_p2_l, "right": new_p2_r}
    except Exception as e:
        _safe_print(f"  ⚠️ rebalance 실패: {e}")
    return data


# ═══════════════════════════════════════════════════════════════════
# v6 후처리 — AI가 빠뜨린 VOCAB/IMPL 마커와 가주어/가목적어 패턴 자동 보정
# ═══════════════════════════════════════════════════════════════════

def _find_in_marked(marked: str, search: str, word_boundary: bool = False) -> tuple:
    """marked 텍스트에서 [[...]] 마커를 무시하고 search 찾기.
    매치 영역이 어떤 마커 영역(GRAMMAR/VOCAB/IMPL) 안이거나, 매치 영역 안에 
    마커가 들어가 있으면 다음 매치 시도. 모두 충돌하면 None.
    
    중첩 마커는 HTML 변환이 지원 안 되므로 회피.
    
    Args:
        word_boundary: True면 단어 경계로만 매치 (\\b...\\b) — 짧은 단어가 다른 단어 안에 
                       substring으로 들어가는 것 방지 (예: 'imagine'이 'imagines' 안에 매치되지 않게)
    
    Returns: (start_in_marked, end_in_marked, matched_text) or None
    """
    import re as _re
    
    # plain text 추출 + 위치 매핑 + 마커 안인지 flag
    plain_chars = []
    pos_map = []  # plain[i] -> marked의 위치
    in_marker = []  # plain[i]가 마커 영역(GRAMMAR/VOCAB/IMPL) 안에 있는지
    
    depth = 0
    i = 0
    while i < len(marked):
        if marked[i:i+2] == "[[":
            end = marked.find("]]", i)
            if end > 0:
                tag = marked[i:end+2]
                if tag.startswith("[[/"):
                    depth = max(0, depth - 1)
                else:
                    depth += 1
                i = end + 2
                continue
        plain_chars.append(marked[i])
        pos_map.append(i)
        in_marker.append(depth > 0)
        i += 1
    
    plain = "".join(plain_chars)
    if word_boundary:
        pattern = _re.compile(r'\b' + _re.escape(search) + r'\b', _re.IGNORECASE)
    else:
        pattern = _re.compile(_re.escape(search), _re.IGNORECASE)
    
    # 모든 매치 시도 — 첫 번째 충돌 없는 매치 사용
    for sm in pattern.finditer(plain):
        if sm.end() > len(pos_map):
            continue
        # 매치 영역이 마커 안에 있으면 패스 (중첩 회피)
        if any(in_marker[sm.start():sm.end()]):
            continue
        s_start = pos_map[sm.start()]
        s_end = pos_map[sm.end() - 1] + 1
        region = marked[s_start:s_end]
        # 매치 영역 내에 마커 시작/끝 있는지 체크 (있으면 충돌)
        if "[[" in region or "]]" in region:
            continue
        return (s_start, s_end, region)
    
    return None


def _filter_bad_vocab(data: dict) -> dict:
    """vocab_notes에서 부적절한 어휘(기능어/대명사/관사/너무 평이한 단어) 제거.
    
    선생님 지적: AI가 'that', 'the', 'Voice Assistants', 'general public' 같은 
    기능어·고유명사·평이한 표현을 vocab_notes로 골라버리는 문제 해결.
    
    제거된 vocab은 passage_marked의 VOCAB 마커도 함께 제거 (일관성).
    
    ★ 안정성: 모든 비정상 입력에도 견디게 — data가 dict 아니거나 필드 누락 시 그대로 반환.
    """
    # 입력 검증 — 비정상이면 그대로 반환
    if not isinstance(data, dict):
        return data
    
    import re as _re
    
    # 절대 vocab 후보가 될 수 없는 단어 (소문자 기준)
    FORBIDDEN = {
        # 기능어·의문사
        'that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'what',
        # 관사·전치사
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to',
        'into', 'onto', 'upon', 'over', 'under', 'about', 'between', 'among', 'through',
        # 대명사
        'it', 'its', 'they', 'them', 'their', 'this', 'these', 'those',
        'he', 'him', 'his', 'she', 'her', 'we', 'us', 'our', 'you', 'your', 'i', 'my',
        # 접속사
        'and', 'or', 'but', 'so', 'yet', 'nor', 'because', 'although', 'though',
        'while', 'whereas', 'if', 'unless', 'since', 'as',
        # be동사·조동사
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
        'will', 'would', 'can', 'could', 'shall', 'should', 'must', 'may', 'might',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'done',
        # 빈출 평이 단어
        'thing', 'things', 'way', 'ways', 'place', 'places', 'time', 'times',
        'day', 'days', 'year', 'years', 'people', 'person',
        'world', 'life', 'lives', 'work', 'works',
        'make', 'makes', 'made', 'making',
        'go', 'goes', 'went', 'going',
        'come', 'comes', 'came', 'coming',
        'get', 'gets', 'got', 'getting',
        'take', 'takes', 'took', 'taking',
        'put', 'puts', 'putting',
        'see', 'sees', 'saw', 'seen', 'seeing',
        'say', 'says', 'said', 'saying',
        'know', 'knows', 'knew', 'known',
        'good', 'bad', 'big', 'small', 'large', 'little',
        'new', 'old', 'high', 'low', 'long', 'short',
        # 기타 너무 평이
        'general public', 'voice assistants', 'computer vision', 
        'natural language processing', 'text recognition', 'ride-hailing apps',
    }
    
    CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"
    
    vocab_notes = data.get("vocab_notes", []) or []
    vocab_detail_raw = data.get("vocab_detail", []) or []
    passage_marked = data.get("passage_marked", "") or ""
    
    # ★ v19 버그 픽스: vocab_detail이 dict({left:[], right:[]}) 형태로 올 수도 있고
    # flat list로 올 수도 있음. 필터링은 flat list로 하지만 원래 구조 보존.
    vocab_detail_is_dict = isinstance(vocab_detail_raw, dict)
    if vocab_detail_is_dict:
        vocab_detail = (vocab_detail_raw.get("left", []) or []) + (vocab_detail_raw.get("right", []) or [])
    else:
        vocab_detail = vocab_detail_raw if isinstance(vocab_detail_raw, list) else []
    
    # ★ v17 강화: 정확 매칭 외에 5가지 패턴으로 부적절 판정
    def _is_bad_vocab(w: str) -> tuple:
        """부적절한 어휘인지 판정. (True/False, 사유) 반환."""
        wl = w.strip().lower()
        if not wl:
            return True, "빈 단어"
        if len(wl) <= 1:
            return True, "1글자"
        if wl.isdigit() or all(c.isdigit() or c in '.,%-' for c in wl):
            return True, "숫자/수치"
        if wl in FORBIDDEN:
            return True, "기능어/평이단어"
        # 관사·전치사로 시작 ("the general population", "a new approach")
        first_word = wl.split()[0]
        if first_word in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                          'his', 'her', 'its', 'their', 'our', 'my', 'your'}:
            return True, "관사·대명사로 시작"
        # 모두 대문자로 시작하는 복합 고유명사 ("Voice Assistants", "Ride-Hailing Apps")
        original_words = w.strip().split()
        if len(original_words) >= 2:
            caps = sum(1 for ow in original_words if ow and ow[0].isupper())
            if caps >= 2:
                return True, "복합 고유명사"
        # 1단어인데 첫 글자 대문자 (Apple, Google, Korea 등 고유명사 가능성)
        if len(original_words) == 1 and w.strip() and w.strip()[0].isupper():
            # 일반 명사도 첫 단어면 대문자일 수 있으니, 모두 대문자거나 특정 패턴만 거름
            if w.strip().isupper():  # 전부 대문자 (NASA, USA, AI 같은 약어)
                return True, "대문자 약어"
        # 5단어 이상의 긴 구문 (구절은 어휘로 부적합)
        if len(original_words) >= 5:
            return True, "긴 구문(어휘 부적합)"
        return False, ""
    
    # vocab_notes 필터링
    removed = []
    cleaned_notes = []
    for idx, v in enumerate(vocab_notes):
        if not isinstance(v, dict):
            cleaned_notes.append(v)
            continue
        word_orig = (v.get("word") or "").strip()
        is_bad, reason = _is_bad_vocab(word_orig)
        # 빈 단어, 1글자, 숫자만, 기능어/평이단어 제거
        if is_bad:
            letter = (v.get("letter") or "").strip()
            removed.append((letter, v.get("word", ""), reason))
            # 본문에서도 해당 VOCAB 마커 제거
            if letter and letter[0] in CIRCLED_ALPHA:
                l = letter[0]
                # [[VOCAB:l=ⓐ]]word[[/VOCAB]] 패턴 제거 (단어만 남김)
                pat1 = f"[[VOCAB:l={l}]]"
                pat2 = "[[/VOCAB]]"
                # 가장 가까운 [[/VOCAB]] 짝 찾아서 제거
                start = passage_marked.find(pat1)
                if start >= 0:
                    end = passage_marked.find(pat2, start)
                    if end >= 0:
                        inner = passage_marked[start+len(pat1):end]
                        passage_marked = passage_marked[:start] + inner + passage_marked[end+len(pat2):]
            continue
        cleaned_notes.append(v)
    
    # vocab_detail도 동일 필터
    cleaned_detail = []
    for v in vocab_detail:
        if not isinstance(v, dict):
            cleaned_detail.append(v)
            continue
        word_orig = (v.get("word") or "").strip()
        is_bad, reason = _is_bad_vocab(word_orig)
        if is_bad:
            removed.append((v.get("letter",""), v.get("word",""), f"detail/{reason}"))
            continue
        cleaned_detail.append(v)
    
    if removed:
        _safe_print(f"  ⚠️ 부적절한 vocab 제거: {removed}")
        # letter 재할당 (제거 후 연속된 ⓐⓑⓒ로) — 본문 마커도 동기화
        # 1) 현재 남은 vocab의 letter → 새 letter 매핑
        notes_remap = {}  # old_letter -> new_letter
        for new_idx, v in enumerate(cleaned_notes):
            if isinstance(v, dict) and new_idx < len(CIRCLED_ALPHA):
                old = (v.get("letter") or "").strip()
                if old:
                    old = old[0]
                new_letter = CIRCLED_ALPHA[new_idx]
                if old and old != new_letter:
                    notes_remap[old] = new_letter
                v["letter"] = new_letter
        
        for new_idx, v in enumerate(cleaned_detail):
            if isinstance(v, dict) and new_idx < len(CIRCLED_ALPHA):
                v["letter"] = CIRCLED_ALPHA[new_idx]
        
        # 2) 본문 VOCAB 마커도 동일하게 매핑 — 임시 placeholder 거쳐서 안전하게
        if notes_remap:
            tmp = passage_marked
            for old_l, new_l in notes_remap.items():
                tmp = tmp.replace(f"[[VOCAB:l={old_l}]]", f"[[VOCAB:l=__{old_l}__]]")
            for old_l, new_l in notes_remap.items():
                tmp = tmp.replace(f"[[VOCAB:l=__{old_l}__]]", f"[[VOCAB:l={new_l}]]")
            passage_marked = tmp
    
    data["vocab_notes"] = cleaned_notes
    # ★ v19 버그 픽스: vocab_detail이 원래 dict 구조였으면 left/right로 복원
    if vocab_detail_is_dict:
        # 좌우 균등 분배 (총 6개 기준 3+3)
        n = len(cleaned_detail)
        mid = (n + 1) // 2
        data["vocab_detail"] = {
            "left": cleaned_detail[:mid],
            "right": cleaned_detail[mid:],
        }
    else:
        data["vocab_detail"] = cleaned_detail
    data["passage_marked"] = passage_marked
    return data


def _find_in_marker_and_split(passage_marked: str, word: str, vocab_letter: str) -> str:
    """단어가 IMPL/GRAMMAR 마커 안에 갇혀있을 때, 그 마커를 단어 위치에서 쪼개고
    중간에 [[VOCAB:l=X]]단어[[/VOCAB]] 삽입.
    
    예: 'Such real-world [[IMPL]]clues offer the[[/IMPL]] visitor'에서 'clues'를 어휘로 마킹하면:
        → 'Such real-world [[IMPL]]'+ [[VOCAB:l=ⓐ]]clues[[/VOCAB]] + '[[IMPL]] offer the[[/IMPL]] visitor'
        실제로는: 'Such real-world [[IMPL]][[/IMPL]][[VOCAB:l=ⓐ]]clues[[/VOCAB]][[IMPL]] offer the[[/IMPL]]'
        간단히: 마커 내부 텍스트만 분할
    
    실패 시 None 반환.
    """
    import re as _re
    
    # 마커 토큰화 (마커 시작·종료·내부 텍스트 분리)
    # 마커 패턴: [[TAG:k=v]]...[[/TAG]] 또는 [[TAG]]...[[/TAG]]
    token_re = _re.compile(
        r'(\[\[(?:GRAMMAR|VOCAB|IMPL)(?::[^\]]+)?\]\])'  # 시작 마커
        r'([^\[]*?)'                                       # 내부 텍스트 (간단히 - 비-중첩 가정)
        r'(\[\[/(?:GRAMMAR|VOCAB|IMPL)\]\])'              # 종료 마커
    )
    
    # IMPL과 GRAMMAR 마커만 대상 (VOCAB은 이미 마킹된 거니까)
    target_re = _re.compile(
        r'(\[\[(?:GRAMMAR|IMPL)(?::[^\]]+)?\]\])([^\[]*?)(\[\[/(?:GRAMMAR|IMPL)\]\])'
    )
    
    word_pat = _re.compile(r'\b' + _re.escape(word) + r'\b', _re.IGNORECASE)
    # 굴절형도 같이
    word_variants = [word, word + 's', word + 'ed', word + 'ing', word + 'es']
    if word.endswith('e'):
        word_variants.append(word + 'd')
        word_variants.append(word[:-1] + 'ing')
    
    for m in target_re.finditer(passage_marked):
        open_tag = m.group(1)
        inner = m.group(2)
        close_tag = m.group(3)
        
        # inner에 단어가 있는지 (단어 경계 적용)
        for variant in word_variants:
            inner_pat = _re.compile(r'\b' + _re.escape(variant) + r'\b', _re.IGNORECASE)
            wm = inner_pat.search(inner)
            if not wm:
                continue
            
            # 마커를 분할: 시작마커 + 앞부분 + 종료마커 + VOCAB마커 + 시작마커 + 뒷부분 + 종료마커
            before_word = inner[:wm.start()]
            matched = inner[wm.start():wm.end()]
            after_word = inner[wm.end():]
            
            # 새 텍스트 구성
            parts = []
            # 마커 종류 판단 (GRAMMAR? IMPL?)
            is_grammar = open_tag.startswith("[[GRAMMAR")
            
            if before_word.strip():
                parts.append(f"{open_tag}{before_word}{close_tag}")
            # 어휘 마커
            parts.append(f"[[VOCAB:l={vocab_letter}]]{matched}[[/VOCAB]]")
            if after_word.strip():
                # ★ GRAMMAR 마커가 분할되는 경우 — 두 번째는 위첨자 숨김(split=1)
                # IMPL은 위첨자 없으니 그대로
                if is_grammar and before_word.strip():
                    # 시작 태그에 split=1 추가 (예: [[GRAMMAR:n=⑤]] → [[GRAMMAR:n=⑤,split=1]])
                    second_open = open_tag.replace("]]", ",split=1]]")
                    parts.append(f"{second_open}{after_word}{close_tag}")
                else:
                    parts.append(f"{open_tag}{after_word}{close_tag}")
            
            replacement = "".join(parts)
            # 원본 마커 영역만 교체
            new_passage = passage_marked[:m.start()] + replacement + passage_marked[m.end():]
            return new_passage
    
    return None


def _ensure_vocab_markers(passage_marked: str, vocab_notes: list) -> str:
    """vocab_notes의 각 단어가 passage_marked에 [[VOCAB]] 마커로 감싸져 있는지 확인하고,
    없으면 본문에서 단어 첫 등장 위치를 찾아 자동으로 [[VOCAB:l=L]]word[[/VOCAB]] 삽입.
    
    AI가 vocab_notes는 만들었지만 본문에 마커를 빠뜨리는 문제 해결.
    
    ★ v15 강화: 단어가 그대로 안 매치되면 단복수·과거형·-ing 형태도 자동 시도.
    """
    if not passage_marked or not vocab_notes:
        return passage_marked
    
    CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"
    added = 0
    not_found = []
    
    for idx, v in enumerate(vocab_notes):
        if not isinstance(v, dict):
            continue
        word = (v.get("word") or "").strip()
        if not word:
            continue
        letter = (v.get("letter") or "").strip()
        # letter가 원알파벳 아니면 idx로 보정
        if not letter or letter[0] not in CIRCLED_ALPHA:
            if idx < len(CIRCLED_ALPHA):
                letter = CIRCLED_ALPHA[idx]
            else:
                continue
        else:
            letter = letter[0]
        
        # 이미 마커가 있는지 체크 (l=letter 또는 plain letter 형태 모두)
        l_idx = CIRCLED_ALPHA.index(letter)
        plain_letter = chr(ord('a') + l_idx)
        if (f"[[VOCAB:l={letter}]]" in passage_marked or 
            f"[[VOCAB:l={plain_letter}]]" in passage_marked):
            continue  # 이미 마커 있음
        
        # ★ 시도 1: 원형 그대로 검색 (단어 경계 적용 — 'imagine'이 'imagines' 안에 매치되지 않게)
        result = _find_in_marked(passage_marked, word, word_boundary=True)
        
        # ★ 시도 2: 굴절형 (단복수, -ing, -ed, -s) — 단어 경계 적용
        if not result:
            candidates = []
            if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
                # study -> studies, studied
                candidates.append(word[:-1] + 'ies')
                candidates.append(word[:-1] + 'ied')
            if word.endswith('e'):
                # use -> used, using
                candidates.append(word + 'd')
                candidates.append(word[:-1] + 'ing')
            else:
                # walk -> walked, walking
                candidates.append(word + 'ed')
                candidates.append(word + 'ing')
            candidates.append(word + 's')
            candidates.append(word + 'es')
            
            for cand in candidates:
                result = _find_in_marked(passage_marked, cand, word_boundary=True)
                if result:
                    break
        
        # ★ 시도 3: 단어 경계 없이 substring 매치 (multi-word phrase / 하이픈 단어 등)
        if not result:
            result = _find_in_marked(passage_marked, word, word_boundary=False)
        
        # ★ v18 시도 4: 다른 마커(IMPL/GRAMMAR) 영역 안에 갇혀있는 단어 — 마커를 쪼개서 어휘 삽입
        # AI가 'Such real-world clues offer the' 같은 긴 함축을 잡으면 'clues' 같은 어휘가 갇힘
        if not result:
            result = _find_in_marker_and_split(passage_marked, word, letter)
            if result:
                # 이 경로는 본문 자체를 재구성하므로 직접 반환된 새 passage_marked 사용
                passage_marked = result
                added += 1
                continue
        
        if not result:
            not_found.append(word)
            continue
        
        s_start, s_end, matched_text = result
        wrapped = f"[[VOCAB:l={letter}]]{matched_text}[[/VOCAB]]"
        passage_marked = passage_marked[:s_start] + wrapped + passage_marked[s_end:]
        added += 1
    
    if added > 0:
        _safe_print(f"  🔵 VOCAB 마커 자동 보정: {added}개 추가")
    if not_found:
        _safe_print(f"  ⚠️ VOCAB 마커 추가 실패 (본문에 없거나 충돌): {not_found}")
    return passage_marked


def _ensure_implication_markers(passage_marked: str, expressions: list) -> str:
    """implication_box.expressions의 각 표현이 passage_marked에 [[IMPL]] 마커로 감싸져
    있는지 확인하고, 없으면 본문에서 자동으로 [[IMPL]]...[[/IMPL]] 삽입.
    
    검색은 마커 영역을 무시하고 plain text로 진행 (마커와 충돌 시 다음 매치 시도).
    """
    import re as _re
    if not passage_marked or not expressions:
        return passage_marked
    
    added = 0
    
    for e in expressions:
        if not isinstance(e, dict):
            continue
        expr = (e.get("expr") or "").strip()
        if not expr or len(expr) < 3:
            continue
        # 앞쪽 기호 제거 (■ 같은 거)
        expr = _re.sub(r'^[■●★◆□○·\s]+', '', expr).strip()
        if not expr:
            continue
        
        # 이미 마커 있는지 (정확히 같은 텍스트를 IMPL로 감싼 게 있는지)
        if f"[[IMPL]]{expr}[[/IMPL]]" in passage_marked:
            continue
        # 대소문자 무관도 체크
        if _re.search(r'\[\[IMPL\]\]' + _re.escape(expr) + r'\[\[/IMPL\]\]', 
                      passage_marked, _re.IGNORECASE):
            continue
        
        # 본문에서 정확히 그 표현 찾기 (마커 영역 회피)
        # 너무 긴 표현은 핵심 부분만으로도 시도
        words = expr.split()
        candidates = [expr]
        if len(words) > 5:
            candidates.append(' '.join(words[:5]))
            candidates.append(' '.join(words[:4]))
            candidates.append(' '.join(words[:3]))
        elif len(words) > 3:
            candidates.append(' '.join(words[:3]))
        
        for cand in candidates:
            result = _find_in_marked(passage_marked, cand)
            if result:
                s_start, s_end, matched_text = result
                wrapped = f"[[IMPL]]{matched_text}[[/IMPL]]"
                passage_marked = passage_marked[:s_start] + wrapped + passage_marked[s_end:]
                added += 1
                break
    
    if added > 0:
        _safe_print(f"  ⚫ IMPL 마커 자동 보정: {added}개 추가")
    return passage_marked


def _check_it_cleft_and_dummy(passage: str, grammar_notes: list, passage_marked: str) -> list:
    """본문에서 가주어/가목적어/It-cleft 패턴을 자동 감지.
    grammar_notes에 해당 패턴이 없으면 추가 후보를 반환.
    
    감지 패턴:
    1. 가주어 it ~ to-V/that:  It is/was + 형용사/명사 + (for X) + to-V/that S+V
    2. 가목적어 it ~ to-V:  V + it + 형용사/명사 + to-V  (make/find/think/consider + it)
    3. It-cleft 강조구문:  It is/was + X + that/who + 동사
    """
    import re as _re
    
    # 이미 grammar_notes에 가주어/가목적어/cleft 관련이 있는지 검사
    existing_tags = []
    for n in grammar_notes or []:
        if isinstance(n, dict):
            tag = (n.get("tag") or "") + " " + (n.get("desc") or "")
            existing_tags.append(tag)
    existing_text = " ".join(existing_tags).lower()
    
    has_dummy_subj = any(k in existing_text for k in ["가주어", "dummy subject", "it ~ to-v"])
    has_dummy_obj = any(k in existing_text for k in ["가목적어", "dummy object"])
    has_cleft = any(k in existing_text for k in ["it-cleft", "it cleft", "분열문", "강조구문"])
    
    detected = []
    
    # 패턴 1: 가주어 it
    p1 = _re.compile(
        r"\bIt\s+(?:is|was|seems|appears|becomes|became|remained|remains)\s+"
        r"(?:not\s+)?(?:\w+\s+){1,3}(?:to\s+\w+|that\s+\w+)",
        _re.IGNORECASE
    )
    for m in p1.finditer(passage):
        if not has_dummy_subj and not has_cleft:
            detected.append({
                "type": "dummy_subj",
                "tag": "가주어 it ~ 진주어 to-V/that",
                "match": m.group(0)[:60],
                "pos": m.start(),
            })
    
    # 패턴 2: 가목적어 it
    p2 = _re.compile(
        r"\b(?:makes?|made|making|finds?|found|finding|thinks?|thought|considers?|considered|"
        r"believes?|believed|deems?|deemed|feels?|felt)\s+it\s+"
        r"(?:\w+\s+){0,3}(?:to\s+\w+|that\s+\w+)",
        _re.IGNORECASE
    )
    for m in p2.finditer(passage):
        if not has_dummy_obj:
            detected.append({
                "type": "dummy_obj",
                "tag": "가목적어 it ~ 진목적어 to-V",
                "match": m.group(0)[:60],
                "pos": m.start(),
            })
    
    # 패턴 3: It-cleft (it is X that + V)
    p3 = _re.compile(
        r"\bIt\s+(?:is|was)\s+\w+(?:\s+\w+){0,5}\s+that\s+(?:\w+\s+)?(?:do|does|did|is|was|are|were|has|have|had|can|could|will|would|may|might|\w+ed|\w+s)\b",
        _re.IGNORECASE
    )
    for m in p3.finditer(passage):
        if not has_cleft:
            # 가주어와 겹칠 수 있으니 위치로 dedup
            already = any(d["pos"] == m.start() for d in detected)
            if not already:
                detected.append({
                    "type": "cleft",
                    "tag": "It-cleft 강조구문 (It is X that ~)",
                    "match": m.group(0)[:60],
                    "pos": m.start(),
                })
    
    return detected


def _inject_dummy_it_notes(data: dict, passage: str) -> dict:
    """가주어/가목적어/It-cleft가 본문에 있는데 grammar_notes에 없으면 강제 추가."""
    grammar_notes = data.get("grammar_notes", [])
    passage_marked = data.get("passage_marked", "")
    detected = _check_it_cleft_and_dummy(passage, grammar_notes, passage_marked)
    
    if not detected:
        return data
    
    CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
    added = 0
    
    # dedup by type
    seen_types = set()
    for d in detected:
        if d["type"] in seen_types:
            continue
        seen_types.add(d["type"])
        
        # 번호 할당 (이미 있는 마지막 번호 + 1)
        next_idx = len(grammar_notes)
        if next_idx >= len(CIRCLED_NUMS):
            break  # 15개 초과 방지
        num = CIRCLED_NUMS[next_idx]
        
        # desc 작성
        if d["type"] == "dummy_subj":
            desc = f'It+be+형/명+to-V/that: it=가주어, to-V/that절=진주어 (본문: "{d["match"]}...")'
        elif d["type"] == "dummy_obj":
            desc = f'V+it+형/명+to-V: it=가목적어, to-V=진목적어 (본문: "{d["match"]}...")'
        else:  # cleft
            desc = f'It is/was + X + that ~: X 강조 (분열문). 제거해도 문장 성립 (본문: "{d["match"]}...")'
        
        grammar_notes.append({
            "num": num,
            "tag": d["tag"],
            "desc": desc,
        })
        added += 1
    
    if added > 0:
        _safe_print(f"  🔴 가주어/가목적어/It-cleft 패턴 자동 감지: {added}개 추가")
        data["grammar_notes"] = grammar_notes
    
    return data


def _shorten_one_topic_en(text: str, target_words: int) -> str:
    """영어 topic을 target_words 근처로 압축. 자연 절단점 우선."""
    import re as _re
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text.split()) <= target_words + 2:
        return text
    # 1. 콜론 앞 (제목형이면 핵심은 콜론 앞)
    if ':' in text:
        before = text.split(':', 1)[0].strip()
        if 3 <= len(before.split()) <= target_words + 3:
            return before
    # 2. 분사구문/관계절/콤마 신호어 앞에서 절단
    for marker in [
        r',\s+(?:demonstrating|reflecting|indicating|representing|suggesting|revealing|while|which|that|where|whereby)\b',
        r'\s+(?:demonstrating|reflecting|indicating|representing|suggesting|revealing)\b',
        r',\s+',
    ]:
        m = _re.search(marker, text)
        if m:
            cand = text[:m.start()].strip().rstrip(',;:')
            if 3 <= len(cand.split()) <= target_words + 3:
                return cand
    # 3. 전치사구 앞 (across/throughout/for 등 부연)에서 절단
    for prep in [' across ', ' throughout ', ' for ', ' in order ', ' so as ']:
        idx = text.find(prep)
        if idx > 0:
            cand = text[:idx].strip()
            if 3 <= len(cand.split()) <= target_words + 3:
                return cand
    # 4. 어절 강제 절단 (target_words개)
    cut = ' '.join(text.split()[:target_words]).rstrip(',;:')
    # 끝이 전치사·관사·접속사로 끝나면 그 단어 제거 (어색함 방지)
    _TRAIL = {'of','to','from','for','in','on','at','by','with','and','or',
              'the','a','an','that','which','as','into','across','through'}
    parts = cut.split()
    while parts and parts[-1].lower() in _TRAIL:
        parts.pop()
    return ' '.join(parts) if parts else cut


def _shorten_one_topic_kr(text: str, target_chars: int) -> str:
    """한국어 topic을 target_chars 근처로 압축. 어절 경계 + 연결어미."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text) <= target_chars + 3:
        return text
    # 연결어미·구두점 뒤에서 절단
    for marker in ['하며 ', '되며 ', '이며 ', '하고 ', '되고 ', '으로 ', '에서 ', ', ', '이는 ', '하는 ', '되는 ']:
        idx = text.find(marker)
        if 6 <= idx <= target_chars + 4:
            return text[:idx + len(marker)].strip()
    # 어절 경계로 target_chars까지
    words = text.split(' ')
    out = ''
    for w in words:
        if len(out) + len(w) + 1 > target_chars:
            break
        out = (out + ' ' + w).strip()
    return out if len(out) >= 5 else text[:target_chars].strip()


def _check_topics_length(data: dict) -> dict:
    """★ topics를 titles 길이에 맞춰 강제 압축 (제목의 1/3 수준).
    
    선생님 반복 요청: 주제문 3문장이 너무 김 → 제목과 비슷한 짧은 길이로.
    시스템 프롬프트로 안 되므로 코드에서 확실히 자른다.
    """
    topics = data.get("topics", [])
    titles = data.get("titles", [])
    topics_kr = data.get("topics_kr", [])
    titles_kr = data.get("titles_kr", [])

    # 목표 길이 = 대응하는 title 길이 (없으면 8단어)
    fixed_en, fixed_kr = 0, 0

    new_topics = []
    for i, t in enumerate(topics):
        if not isinstance(t, str):
            new_topics.append(t); continue
        # 대응 title 단어 수 (기준)
        if i < len(titles) and isinstance(titles[i], str):
            target = max(6, len(titles[i].split()))
        else:
            target = 8
        before_len = len(t.split())
        shortened = _shorten_one_topic_en(t, target)
        if len(shortened.split()) < before_len:
            fixed_en += 1
        new_topics.append(shortened)
    if new_topics:
        data["topics"] = new_topics

    new_kr = []
    for i, t in enumerate(topics_kr):
        if not isinstance(t, str):
            new_kr.append(t); continue
        if i < len(titles_kr) and isinstance(titles_kr[i], str):
            target = max(12, len(titles_kr[i]))
        else:
            target = 18
        before = len(t)
        shortened = _shorten_one_topic_kr(t, target)
        if len(shortened) < before:
            fixed_kr += 1
        new_kr.append(shortened)
    if new_kr:
        data["topics_kr"] = new_kr

    if fixed_en or fixed_kr:
        _safe_print(f"  ✂️ topics 자동 압축: 영어 {fixed_en}건, 한글 {fixed_kr}건 (제목 길이에 맞춤)")
    return data


def _enforce_summary_length(data: dict, max_words: int = 40) -> dict:
    """요약문(summary.en)을 40단어 이내로 강제 절단.
    
    선생님 요청: 요약이 너무 김 → 항상 40단어 이내. 시스템 프롬프트가 안 지켜져서 코드 강제.
    문장 경계를 존중하며 40단어 넘는 마지막 문장은 통째로 제거.
    """
    import re as _re
    impl = data.get("implication_box", {})
    if not isinstance(impl, dict):
        return data
    summary = impl.get("summary", {})
    if not isinstance(summary, dict):
        return data
    en = summary.get("en", "")
    if not isinstance(en, str) or not en.strip():
        return data

    words = en.split()
    if len(words) <= max_words:
        return data

    # 문장 단위로 누적하다가 40단어 넘으면 직전까지만 (문장 완결성 유지)
    sentences = _re.split(r'(?<=[.!?])\s+', en.strip())
    kept, count = [], 0
    for s in sentences:
        w = len(s.split())
        if count + w <= max_words:
            kept.append(s)
            count += w
        else:
            break
    if kept:
        result = ' '.join(kept).strip()
    else:
        # 첫 문장조차 40단어 초과 → 어절로 강제 절단 후 마침표
        result = ' '.join(words[:max_words]).rstrip(',;:') + '.'

    summary["en"] = result
    impl["summary"] = summary
    data["implication_box"] = impl
    _safe_print(f"  ✂️ 요약문 압축: {len(words)}단어 → {len(result.split())}단어 (40 이내 강제)")
    return data


def _enforce_count_3(data: dict) -> dict:
    """blank_candidates와 implicit_meanings를 정확히 3개로 강제.
    
    선생님 요청: 함축이 3~5개로 들쭉날쭉 → 빈칸·함축 둘 다 정확히 3개씩만.
    AI가 3개 초과 생성하면 앞 3개만 남기고 자른다.
    문자열 등 dict가 아닌 항목(스키마 설명 잔재)은 제거.
    """
    impl = data.get("implication_box", {})
    if not isinstance(impl, dict):
        return data
    for key in ("blank_candidates", "implicit_meanings"):
        items = impl.get(key, [])
        if isinstance(items, list):
            # dict 항목만 남기고 (스키마 설명 문자열 제거)
            dicts = [x for x in items if isinstance(x, dict)]
            if len(dicts) != len(items) or len(dicts) > 3:
                _safe_print(f"  ✂️ {key}: {len(items)}개 → {min(len(dicts),3)}개로 정리")
            impl[key] = dicts[:3]
    data["implication_box"] = impl
    return data


def _passage_marked_to_html(marked: str, original: str = "") -> str:
    """passage_marked의 [[GRAMMAR/VOCAB/IMPL]] 마커를 HTML span으로 변환.

    Claude가 n=①(원숫자) 대신 n=1(일반숫자), l=ⓐ 대신 l=a로 출력해도 호환되도록
    정규식이 두 형태 모두 매칭. 출력 시 원숫자/원알파벳으로 자동 변환.

    원본 길이를 초과하는 부분은 잘라낸다 (Claude가 끝에 추가 문장을 붙였을 때 대비).
    """
    import re as _re
    import html as _html

    orig_len = len(original.strip()) if original else None

    # 원숫자/원알파벳 변환 테이블
    CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
    CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"

    def _to_circled_num(s: str) -> str:
        """'1' → '①', '①' → '①' (이미 원숫자면 그대로)."""
        s = s.strip()
        if s and s[0] in CIRCLED_NUMS:
            return s
        try:
            n = int(s)
            if 1 <= n <= 15:
                return CIRCLED_NUMS[n - 1]
        except ValueError:
            pass
        return s  # 못 변환하면 원본 유지

    def _to_circled_alpha(s: str) -> str:
        """'a' → 'ⓐ', 'ⓐ' → 'ⓐ'."""
        s = s.strip()
        if s and s[0] in CIRCLED_ALPHA:
            return s
        if len(s) == 1 and 'a' <= s.lower() <= 'o':
            return CIRCLED_ALPHA[ord(s.lower()) - ord('a')]
        return s

    out = []
    plain_len = 0
    # 일반 숫자/알파벳 + 유니코드 원숫자/원알파벳 모두 허용
    # GRAMMAR는 ,split=1 같은 추가 attribute가 붙을 수 있음
    pattern = _re.compile(
        r'\[\[(GRAMMAR:n=([0-9]+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+)(?:,split=(\d+))?'
        r'|VOCAB:l=([a-oA-O]|[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ])'
        r'|IMPL)\]\]'
        r'([\s\S]*?)'
        r'\[\[/(GRAMMAR|VOCAB|IMPL)\]\]'
    )
    last_end = 0
    for m in pattern.finditer(marked):
        if m.start() > last_end:
            plain = marked[last_end:m.start()]
            if orig_len and plain_len + len(plain) > orig_len:
                plain = plain[:max(0, orig_len - plain_len)]
                out.append(_html.escape(plain))
                plain_len += len(plain)
                return "".join(out)
            out.append(_html.escape(plain))
            plain_len += len(plain)

        kind = m.group(1)
        inner = m.group(5)  # ★ inner 그룹 번호 변경 (split 그룹 추가됨)
        if kind.startswith("GRAMMAR"):
            num = _to_circled_num(m.group(2))
            split_marker = m.group(3)  # split=1이면 위첨자 숨김
            if split_marker:
                # 분할된 두 번째 조각 — 밑줄만, 위첨자 없음
                out.append(f'<span class="gr">{_html.escape(inner)}</span>')
            else:
                out.append(f'<span class="gr">{_html.escape(inner)}</span><sup class="sup-r">{num}</sup>')
        elif kind.startswith("VOCAB"):
            letter = _to_circled_alpha(m.group(4))  # ★ group 번호 변경
            out.append(f'<span class="vc">{_html.escape(inner)}</span><sup class="sup-b">{letter}</sup>')
        else:  # IMPL
            out.append(f'<span class="im">{_html.escape(inner)}</span>')
        plain_len += len(inner)
        last_end = m.end()

    if last_end < len(marked):
        tail = marked[last_end:]
        if orig_len and plain_len + len(tail) > orig_len:
            tail = tail[:max(0, orig_len - plain_len)]
        out.append(_html.escape(tail))

    return "".join(out)
