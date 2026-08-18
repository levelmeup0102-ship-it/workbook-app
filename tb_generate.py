# -*- coding: utf-8 -*-
"""
수업배경자료 자동 생성 (web_search 사용).
topic_background.py 에서 import 해서 씀. pipeline 의 call_claude 인프라(curl)를 그대로 활용.

[패치 v2] _try_parse 강화 — HTML 포함 JSON의 이스케이프 문제 해결
"""
import os, re, json, time, subprocess, tempfile, html as _html

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-6"

# 생성 로직을 고칠 때마다 올린다. 이 값이 다르면 예전 캐시를 무시하고 다시 만든다.
GEN_VERSION = "v6"   # v6: sanitize_svg 그라디언트 태그 버그 수정(+폰트 임베드 렌더러)

# 이미지 허용 도메인 (위키 계열만) — v5부터 사진은 쓰지 않지만 세니타이즈용으로 남겨둠
_IMG_ALLOW = ("upload.wikimedia.org",)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_EXAM_CTX_CACHE = None


def _supa_get(endpoint: str, timeout: int = 15):
    """Supabase REST GET (동기). tb_generate 는 curl 기반이라 supa.py(async)를 쓰지 않는다."""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    try:
        r = subprocess.run(
            ['curl', '-s', '--max-time', str(timeout),
             f"{SUPABASE_URL}/rest/v1/{endpoint}",
             '-H', f'apikey: {SUPABASE_KEY}',
             '-H', f'Authorization: Bearer {SUPABASE_KEY}'],
            capture_output=True, timeout=timeout + 5)
        out = r.stdout.decode('utf-8', 'replace').strip()
        return json.loads(out) if out else None
    except Exception as e:
        print(f"[exam_ctx] supabase 조회 실패: {str(e)[:120]}", flush=True)
        return None


def load_exam_context() -> str:
    """학원의 실제 출제 체계를 프롬프트에 주입한다.
    seosul_types = 서술형 5유형(SA~SE) + 어법 출제 전역 원칙(GRAMMAR-RULES)
    grammar_points = 내신·수능 빈출 어법 화이트리스트
    (seosul_types 주석: "어느 프로젝트에서든 서술형/어법 출제 시 grammar_points와 함께 참조한다")
    조회 실패 시 빈 문자열 → 프롬프트는 기존대로 동작한다."""
    global _EXAM_CTX_CACHE
    if _EXAM_CTX_CACHE is not None:
        return _EXAM_CTX_CACHE

    parts = []
    types = _supa_get("seosul_types?select=code,name,instruction,given_to_student,"
                      "student_writes,source_schools,notes&active=eq.true&order=id")
    if isinstance(types, list) and types:
        lines = ["[우리 학원 서술형 출제 유형 — 시험 포인트 카드에 반드시 반영]"]
        for t in types:
            if t.get("code") == "GRAMMAR-RULES":
                continue
            lines.append(
                f"- {t.get('code')} {t.get('name')}: {(t.get('given_to_student') or '')[:90]}"
                f" → 학생이 쓰는 것: {(t.get('student_writes') or '')[:60]}"
                f" (출처교: {t.get('source_schools') or '-'})")
        rules = next((t for t in types if t.get("code") == "GRAMMAR-RULES"), None)
        if rules and rules.get("notes"):
            lines.append(f"[어법 출제 원칙] {rules['notes'][:1200]}")
        parts.append("\n".join(lines))

    gp = _supa_get("grammar_points?select=category,subcategory,name,trap_warning"
                   "&active=eq.true&exam_frequency=gte.4&order=exam_frequency.desc&limit=40")
    if isinstance(gp, list) and gp:
        lines = ["[내신 빈출 어법 포인트 — 어법 언급은 이 목록 안에서만]"]
        for g in gp:
            trap = (g.get("trap_warning") or "").strip().replace("\n", " ")
            lines.append(f"- {g.get('category')}·{g.get('subcategory')}: {g.get('name')}"
                         + (f" / 함정: {trap[:70]}" if trap else ""))
        parts.append("\n".join(lines))

    _EXAM_CTX_CACHE = ("\n\n" + "\n\n".join(parts)) if parts else ""
    print(f"[exam_ctx] 출제 체계 {len(_EXAM_CTX_CACHE)}자 로드", flush=True)
    return _EXAM_CTX_CACHE

def _curl_messages(body: dict, timeout=180) -> dict:
    """단일 /v1/messages 호출 (curl). web_search tool 포함 가능."""
    if not API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    body_json = json.dumps(body, ensure_ascii=False)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', delete=False) as tmp:
            tmp.write(body_json); tmp_path = tmp.name
        r = subprocess.run(
            ['curl','-s','-X','POST',API_URL,
             '-H',f'x-api-key: {API_KEY}',
             '-H','anthropic-version: 2023-06-01',
             '-H','content-type: application/json; charset=utf-8',
             '-d',f'@{tmp_path}'],
            capture_output=True, timeout=timeout)
        if r.returncode != 0:
            raise Exception(f"curl error: {r.stderr.decode('utf-8','replace')[:200]}")
        data = json.loads(r.stdout.decode('utf-8'))
        if 'error' in data:
            raise Exception(f"API error: {json.dumps(data['error'])[:300]}")
        return data
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

def call_claude_with_search(system_prompt: str, user_prompt: str,
                            max_uses=5, max_tokens=16000, max_turns=6) -> str:
    tools = [{"type":"web_search_20250305","name":"web_search","max_uses":max_uses,
              "user_location":{"type":"approximate","country":"KR"}}]
    messages = [{"role":"user","content":user_prompt}]
    for _turn in range(max_turns):
        body = {"model":MODEL,"max_tokens":max_tokens,"temperature":0.3,
                "system":system_prompt,"messages":messages,"tools":tools}
        data = _curl_messages(body)
        content = data.get("content", [])
        messages.append({"role":"assistant","content":content})
        stop = data.get("stop_reason")
        # 서버측 web_search 가 여러 턴에 걸치면 API 는 pause_turn 을 돌려준다.
        # 이때 아무 말도 덧붙이지 말고 그대로 이어 호출해야 검색이 계속된다.
        # (예전에는 pause_turn 을 처리하지 않아 검색 도중 끊기고, 불완전한 텍스트가
        #  반환돼 JSON 파싱 실패 → '검색 없이 재시도' 폴백으로 새는 원인이었다.)
        if stop == "pause_turn":
            continue
        if stop == "tool_use":
            messages.append({"role":"user","content":"이어서 진행하세요."})
            continue
        if stop == "max_tokens":
            print(f"[topic_background] ⚠ max_tokens 도달 — 출력이 잘렸을 수 있음", flush=True)
        texts = [b.get("text","") for b in content if b.get("type")=="text"]
        return "\n".join(t for t in texts if t).strip()
    texts = [b.get("text","") for b in messages[-1]["content"] if isinstance(b,dict) and b.get("type")=="text"]
    return "\n".join(texts).strip()

_WIKI_IMG_RE = re.compile(r'^https://upload\.wikimedia\.org/wikipedia/[^\s"\'<>]+\.(?:jpg|jpeg|png|svg|gif|webp)(?:\?[^\s"\'<>]*)?$', re.I)

def validate_image_url(url: str, timeout=8) -> bool:
    if not url or not _WIKI_IMG_RE.match(url.strip()):
        return False
    try:
        r = subprocess.run(['curl','-s','-I','-L','--max-time',str(timeout),url],
                           capture_output=True, timeout=timeout+3)
        head = r.stdout.decode('utf-8','replace').lower()
        # 사내/서버 프록시가 HEAD를 막는 특수 케이스만 통과시킨다.
        if ("host_not_allowed" in head) or ("x-deny-reason" in head):
            return True
        if not head.strip():
            return False          # ★ 응답 자체가 없으면 '살아있다'고 믿지 않는다 (깨진 이미지 원인)
        first = head.split("\n")[0] if head else ""
        if any(code in first for code in ("404", "410", "403", "500", "502", "503")):
            return False
        return True
    except Exception:
        return False              # ★ 확인 실패 = 넣지 않는다

def _curl_json(url: str, timeout=10):
    try:
        r = subprocess.run(['curl','-s','-L','--max-time',str(timeout),
                            '-H','User-Agent: LevelMeUp-Workbook/1.0 (education)',
                            url], capture_output=True, timeout=timeout+3)
        out = r.stdout.decode('utf-8','replace')
        if not out.strip(): return None
        return json.loads(out)
    except Exception:
        return None

def _norm_upload(u: str) -> str:
    if not u: return ""
    if u.startswith("//"): u = "https:" + u
    return u

def _upgrade_thumb(url: str, target=800) -> str:
    if not url: return url
    m = re.search(r'/(\d+)px-', url)
    if m:
        try:
            w=int(m.group(1))
            if w < target:
                url = url[:m.start()] + ("/%dpx-" % target) + url[m.end():]
        except Exception: pass
    return url

def _thumb_width(url: str) -> int:
    m = re.search(r'/(\d+)px-', url)
    try: return int(m.group(1)) if m else 9999
    except Exception: return 9999

# 문서 '대표 이미지'로 자주 딸려오는, 설명력이 없는 파일들
_JUNK_IMG_RE = re.compile(
    r'(icon|logo|seal[_ ]of|great[_ ]seal|coat[_ ]of[_ ]arms|flag[_ ]of|'
    r'commons[-_]logo|question[_ ]book|ambox|emblem|crest)', re.I)

def _img_basename(url: str) -> str:
    """도메인·경로를 뺀 파일명만. (URL 전체로 검사하면 upload.wikimedia.org 때문에 전부 걸린다)"""
    return (url or "").split("?")[0].rstrip("/").split("/")[-1]

_STOP_TOKENS = {
    "the","of","a","an","in","on","and","or","for","to","with","by",
    "diagram","chart","graph","image","photo","picture","illustration",
    "theory","effect","concept","model","principle","example","crew","group",
}

def _toks(s: str) -> set:
    return {t for t in re.findall(r'[a-z0-9]+', (s or "").lower())
            if t not in _STOP_TOKENS and len(t) > 2}

def is_relevant_result(query: str, title: str) -> bool:
    """위키 검색이 '단어 하나 겹치는 엉뚱한 문서'를 물어오는 걸 막는다.
    예) "Self-determination theory diagram" → "Spacetime diagram" 은 걸러짐."""
    q, t = _toks(query), _toks(title)
    if not q or not t:
        return False
    overlap = len(q & t)
    if overlap == 0:
        return False
    # 결과 제목 토큰의 절반 이상이 질의에 들어 있어야 같은 주제로 본다
    return overlap >= max(1, (min(len(q), len(t)) + 1) // 2)

def is_usable_image(url: str, query: str, title: str) -> bool:
    if not url:
        return False
    if _JUNK_IMG_RE.search(_img_basename(url)):
        print(f"[img] 제외(아이콘·로고류): {title} ← {query}", flush=True)
        return False
    if not is_relevant_result(query, title):
        print(f"[img] 제외(주제 불일치): {title} ← {query}", flush=True)
        return False
    return validate_image_url(url)


def wiki_search_image(query: str, lang: str = "en", n: int = 1) -> list:
    if not query: return []
    import urllib.parse as _u
    base = f"https://{lang}.wikipedia.org"
    url = f"{base}/w/rest.php/v1/search/page?q={_u.quote(query)}&limit={max(1,n*2)}"
    data = _curl_json(url)
    out = []
    if data and isinstance(data.get("pages"), list):
        for p in data["pages"]:
            th = (p or {}).get("thumbnail") or {}
            src = _upgrade_thumb(_norm_upload(th.get("url","")), 800)
            if src and is_usable_image(src, query, p.get("title","")):
                out.append({"url": src, "alt": p.get("title",""), "credit": "Wikimedia Commons",
                            "title": p.get("title","")})
            if len(out) >= n: break
    return out

def wiki_summary_image(title: str, lang: str = "en") -> dict | None:
    if not title: return None
    import urllib.parse as _u
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{_u.quote(title)}"
    d = _curl_json(url)
    if not d: return None
    src = ((d.get("originalimage") or {}).get("source")
           or (d.get("thumbnail") or {}).get("source") or "")
    src = _upgrade_thumb(_norm_upload(src), 800)
    if src and is_usable_image(src, title, d.get("title", title)):
        return {"url": src, "alt": d.get("title", title), "credit": "Wikimedia Commons",
                "title": d.get("title", title)}
    return None

def fetch_images_for(queries: list, lang_order=("en","ko"), max_imgs=2) -> list:
    seen, out = set(), []
    for q in queries:
        if len(out) >= max_imgs: break
        for lang in lang_order:
            got = wiki_summary_image(q, lang) or (wiki_search_image(q, lang, 1) or [None])[0]
            if got and got["url"] not in seen:
                seen.add(got["url"]); out.append(got); break
    return out


# =========================================================================
# ★ 개선된 JSON 파서 (_try_parse v3)
# =========================================================================
def _slice_json_object(txt: str, key: str) -> str | None:
    """txt 안의 "key": { ... } 에서 중괄호 짝을 세어 객체 전체를 잘라낸다.
    non-greedy 정규식이 중첩 객체의 첫 }에서 끊기는 문제를 해결."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\{', txt)
    if not m:
        return None
    start = txt.index("{", m.end() - 1)
    depth, i, in_str, esc = 0, start, False, False
    while i < len(txt):
        ch = txt[i]
        if in_str:
            if esc:      esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if   ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return txt[start:i + 1]
        i += 1
    return None


def _slice_json_array(txt: str, key: str) -> str | None:
    """txt 안의 "key": [ ... ] 에서 대괄호 짝을 세어 배열 전체를 잘라낸다."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', txt)
    if not m:
        return None
    start = txt.index("[", m.end() - 1)
    depth, i, in_str, esc = 0, start, False, False
    while i < len(txt):
        ch = txt[i]
        if in_str:
            if esc:            esc = False
            elif ch == "\\":   esc = True
            elif ch == '"':    in_str = False
        else:
            if   ch == '"': in_str = True
            elif ch in "[{": depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    return txt[start:i + 1]
        i += 1
    return None


def _try_parse(raw: str) -> dict | None:
    """HTML 포함 JSON 파싱 — 5단계 폴백으로 강화"""
    if not raw:
        return None
    txt = raw.strip()

    # 1) 코드블록 제거
    txt = re.sub(r'^```(?:json)?\s*', '', txt)
    txt = re.sub(r'\s*```\s*$', '', txt)
    txt = txt.strip()

    # 2) 직접 파싱
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 3) 가장 바깥 { } 로 슬라이싱 후 파싱
    start = txt.find('{')
    end = txt.rfind('}')
    if start != -1 and end > start:
        candidate = txt[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4) 제어문자 제거 후 재시도
    try:
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', txt)
        return json.loads(cleaned)
    except Exception:
        pass

    # 5) ★ 핵심 수정: section_html 등 HTML 필드의 이스케이프되지 않은 따옴표/개행 때문에
    #    json.loads가 실패하는 경우 → 필드별로 안전하게 추출
    try:
        result = {}

        # 단순 문자열 필드 (HTML 아닌 것들)
        for key in ("anchor", "chip", "script_js", "_step_version"):
            m = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)
            if m:
                try:
                    result[key] = json.loads('"' + m.group(1) + '"')
                except Exception:
                    result[key] = m.group(1)

        # HTML 필드: section_html, overview_html — 다음 키까지 탐욕적으로 추출
        for key in ("section_html", "overview_html"):
            # "key": "..." 에서 값 추출 — 다음 최상위 키 패턴이 나오기 전까지
            pat = re.compile(
                r'"' + key + r'"\s*:\s*"([\s\S]*?)"(?=\s*,\s*"(?:anchor|chip|section_html|overview_html|script_js|image_queries|figures|toggle|_step_version)"|\s*\})',
                re.S
            )
            m = pat.search(txt)
            if m:
                val = m.group(1)
                # 이스케이프 시퀀스 복원 (단, 실제 \n \t 는 그대로)
                try:
                    val = json.loads('"' + val + '"')
                except Exception:
                    # 복원 실패해도 원문 그대로 사용
                    pass
                result[key] = val

        # image_queries 배열
        iq_m = re.search(r'"image_queries"\s*:\s*(\[[^\]]*\])', txt)
        if iq_m:
            try:
                result["image_queries"] = json.loads(iq_m.group(1))
            except Exception:
                pass

        # toggle 객체 — 중첩({"items":[{...}]}) 구조라 중괄호 매칭으로 정확히 잘라낸다.
        fg_raw = _slice_json_array(txt, "figures")
        if fg_raw:
            try:
                result["figures"] = json.loads(fg_raw)
            except Exception as _e:
                print(f"[_try_parse] figures 파싱 실패: {_e}", flush=True)

        tg_raw = _slice_json_object(txt, "toggle")
        if tg_raw:
            try:
                result["toggle"] = json.loads(tg_raw)
            except Exception as _e:
                print(f"[_try_parse] toggle 파싱 실패: {_e}", flush=True)

        if result.get("anchor") and (result.get("section_html") or result.get("chip")):
            print(f"[_try_parse] 필드별 추출 성공: anchor={result.get('anchor')}", flush=True)
            result["_parse_fallback"] = True
            return result

    except Exception as e:
        print(f"[_try_parse] 필드별 추출도 실패: {e}", flush=True)

    return None


# =========================================================================
# 생성 프롬프트 + 후처리
# =========================================================================
GEN_SYSTEM = """너는 한국 고등학교 영어 내신 강사를 위한 '수업배경자료' 제작자다.
영어 지문 1개를 받아, 그 지문의 핵심 '소재(개념·장소·현상·연구·인물)'의 배경을 파고드는 학습 카드를 만든다.
목표는 백과사전이 아니라 "이 배경지식이 지문 독해와 내신 시험에 어떻게 쓰이는지"를 학생이 알게 하는 것.

[사실 검증 — 매우 중요]
- 학자/연도/지명/수치/연구/작품명 같은 사실은 반드시 web_search로 교차 확인한다. 확인 안 된 건 쓰지 않는다(환각 금지).

[수업 쓸모 — 반드시 지킬 것]
- 카드마다 .inpassage 에 "이 배경이 지문 ❶❷ 문장 독해에 어떻게 연결되는지"를 구체적으로 적는다.
- 마지막 카드는 .card danger 로 '시험 포인트' 카드를 만든다. 아래 세 축을 반드시 모두 담는다.
  ① 수능형 객관식: 빈칸·순서·삽입·요약·어휘 중 이 지문에 실제로 붙는 것만. 정답 근거 문장과 오답 함정을 같이 적는다.
  ② 서술형(내신 배점이 큰 부분): 아래 [우리 학원 서술형 출제 유형]에서 이 지문에 맞는 유형 코드를 골라
     "어느 절이 뚫릴지 / 요약문 (A)(B)에 무엇이 들어갈지 / 어떤 단어가 품사 변형으로 나올지"를 구체적으로 지목한다.
     지문에 안 맞는 유형은 억지로 넣지 않는다.
  ③ 어법: 아래 [내신 빈출 어법 포인트] 목록 안에서만 고른다. 목록에 없으면 어법 얘기를 아예 쓰지 마라.
     [어법 출제 원칙]의 금지 항목에 걸리는 자리는 언급하지 않는다.
- 소재를 고를 때 "이 소재가 수능 연계·내신에서 왜 반복해서 나오는지"를 한 번은 짚는다.
- 'theme' 와 'overview' 는 지문의 논리 흐름(주제·전환)을 한 줄로 요약한다.

[그림 — 사진이 아니라 직접 그린 SVG 모식도]
- <img> 태그를 절대 만들지 마라. 외부 이미지 URL도 쓰지 마라.
- 대신 출력 JSON 의 "figures" 에 인라인 SVG 모식도를 1~2개 직접 그려 넣어라.
- 무엇을 그리나 — 아래 레퍼토리에서 지문에 맞는 걸 고른다(원장님 검수본에서 실제로 쓰인 형태):
  · 두 개념 대비 (예: 내적 동기 ↔ 외적 동기, 안에서 타오름 ↔ 밖에서 당김)
  · 겹치는 원 3개 (예: 자율성·유능감·관계성이 겹치는 곳 = 내적 동기)
  · 정상 vs 이상 비교 (예: 자유롭게 굽는 관절 ↔ 굳어 고정된 관절)
  · 가로 타임라인 (예: 1999 입문 → 2007 창단 → 2009 졸업 → 2010 패럴림픽)
  · 경로·여정도 (지점을 선으로 잇고 각 지점에 연도·사건)
  · 누적 계단 (예: 삼진 2,600 · 실험 1,000 · 오디션 탈락이 쌓여 성공 ★)
  · 전후 대비 (예: 같은 극장, 30년 전 탈락 ✗ → 30년 후 토니상 🏆)
  ★ 장식용 그림·아이콘·초상화는 만들지 마라. 설명이 되는 그림만.
  ★ 글로 한 줄이면 끝나는 단순 정의에는 그림을 만들지 마라(figures 를 빈 배열로 둬라).
- SVG 작성 규칙:
  · 반드시 viewBox 기반. width/height 속성 금지. 폭은 600 고정, 높이는 120~260 사이.
    (예: <svg viewBox='0 0 600 190' role='img' aria-label='내적 동기 대 외적 동기'>)
  · role='img' 와 aria-label 을 반드시 넣는다.
  · 속성값은 작은따옴표(') 사용 — JSON 이스케이프를 줄이기 위해서다.
  · 색은 디자인 토큰을 기본으로 쓴다: #0f2e2a(ink) #16b89b(teal) #0e8f79(teal-deep)
    #e6f7f2(mint) #e0922a(amber) #e8493f(danger) #2f7de0(blue) #7a5cd6(purple)
    #9aa7a3(gray) #dbe8e3(line) #ffffff. 명암·배경용 파생 음영은 써도 된다.
  · 라벨은 한국어로, 학술어는 영어를 작게 병기한다 (예: 자율성 / autonomy).
  · text 는 font-size 11~15, font-weight='800', text-anchor='middle'.
  · ✗ ★ 🏆 정도의 기호는 의미 전달에 도움되면 써도 된다.
  · script·foreignObject·image·외부 href 절대 금지. 순수 도형·선·텍스트만.
  · 한 그림당 2만 자 이내.
- caption 규칙: 그림 설명에서 끝내지 말고 <b>지문과의 연결</b>까지 한 문장에 담는다.
  (예: "내적 동기 = 안에서 타오르는 불꽃 · 외적 동기 = 밖에서 당기는 보상. 지문 속 인물들은 전자예요.")
  ★ 실제와 다른 도식(지도 등)이면 "(실제 지리와 다름)"처럼 반드시 밝힌다. 사실처럼 오해되면 안 된다.

[정형 인터랙티브 1개 — 반드시 포함]
- 출력 JSON 의 "toggle" 에 핵심 대조/분류를 2~3개 항목으로 넣어라(탭 토글로 렌더된다). 형식:
  "toggle": { "title":"질문/제목", "items":[ {"label":"버튼명","body":"설명(핵심어 <b>볼드</b>)"}, ... ] }

[출력 형식 — JSON only, 코드블록·설명·인용표시 금지]
★★★ section_html 안에 작은따옴표(') 를 절대 사용하지 말 것. 반드시 큰따옴표(") 또는 HTML 엔티티(&apos;)를 사용하라.
★★★ JSON 문자열 값 안에서 큰따옴표는 반드시 \\" 로 이스케이프하라.
★★★ JSON 문자열 값 안에 실제 줄바꿈(개행)을 넣지 말 것. \\n 으로 표현하라.

{
  "anchor": "tb_xxx",
  "chip": "지문번호 + 짧은소재 + 이모지 1개",
  "section_html": "<section id=\\"tb_xxx\\"> ... </section>",
  "overview_html": "<h3>{이모지} {지문번호} — {소재}</h3>...",
  "figures": [ {"title":"그림 제목", "caption":"이 그림이 뭘 보여주는지 한 줄", "svg":"<svg viewBox='0 0 600 260'>...</svg>"} ],
  "toggle": { "title":"...", "items":[ {"label":"...","body":"..."} ] },
  "script_js": ""
}

[section_html 규칙]
- 최상위 <section id="{anchor}">.
- 모든 HTML 속성값은 큰따옴표(") 사용. 작은따옴표(') 절대 금지.
- 순서: <div class="sec-num"> → <h2><span class="bar"></span>{한글소제목}</h2>
  → <div class="sec-en"> → <span class="theme"> → 카드 3~4개 → <div class="keyterm">
- 카드: <div class="card amber|danger|blue|purple"> 안에
  <div class="chead"><span class="ctitle">제목<span class="en">English</span></span><span class="ctag">라벨</span></div>
  <div class="csub">부제</div> <div class="cbody">설명(<b>핵심어</b>만 볼드)</div>
  끝에 <div class="inpassage"><span class="lab">지문 속</span> ❶❷ ...연결...</div>
- keyterm: <div class="keyterm"><div class="kt"><h4>용어</h4><p>정의</p></div> 2개</div>
- <cite>/<sup>/각주/[1] 같은 인용표시 절대 금지.
- 톤: 친근한 구어체("~예요","~죠").
"""

GEN_USER_TMPL = """다음은 분석할 영어 지문이다. 지문 번호: {label}

<passage>
{passage}
</passage>

이 지문의 핵심 소재를 파악하고, web_search로 사실(학자/연도/지명/수치/연구)을 교차 확인한 뒤,
시스템 지시의 JSON을 출력하라. 그림이 필요하면 figures 에 SVG 모식도를 직접 그려 넣어라(사진·외부 URL 금지).
anchor 는 "tb_{anchor_hint}" 형식으로 만들고, 지문 번호는 "{label}"를 사용하라.

★ JSON 출력 시 반드시 지킬 것:
1. section_html 내 모든 HTML 속성을 큰따옴표(")로 작성
2. JSON 문자열 내 큰따옴표는 \\" 로 이스케이프
3. JSON 문자열 내 줄바꿈은 \\n 으로 표현 (실제 개행 금지)
4. 코드블록(```) 없이 JSON 객체만 출력

JSON 외 아무것도 출력하지 마라."""

# ---- 새니타이즈 ----
_ALLOWED_IMG_RE = re.compile(r'<img\b[^>]*>', re.I)

def _strip_dangerous(html: str) -> str:
    if not html: return ""
    html = re.sub(r'<\s*(script|style|iframe)[^>]*>.*?<\s*/\s*\1\s*>', '', html, flags=re.I|re.S)
    html = re.sub(r'<\s*(script|style|iframe)[^>]*/?>', '', html, flags=re.I)
    html = re.sub(r'\son\w+\s*=\s*"[^"]*"', '', html, flags=re.I)
    html = re.sub(r"\son\w+\s*=\s*'[^']*'", '', html, flags=re.I)
    html = re.sub(r'\son\w+\s*=\s*[^\s>]+', '', html, flags=re.I)
    def _img(m):
        tag = m.group(0)
        src = re.search(r'src\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        if not src or not any(d in src.group(1) for d in _IMG_ALLOW):
            return ''
        return tag
    html = _ALLOWED_IMG_RE.sub(_img, html)
    return html

# ─────────────────────────────────────────────────────────────
# 직접 그린 SVG 모식도 검증·정리
# ─────────────────────────────────────────────────────────────
_SVG_ALLOWED_TAGS = {
    "svg","g","defs","title","desc","path","rect","circle","ellipse","line",
    "polyline","polygon","text","tspan","marker","linearGradient","radialGradient",
    "stop","pattern","clipPath","mask","use","symbol",
}
_SVG_ALLOWED_LOWER = {t.lower() for t in _SVG_ALLOWED_TAGS}   # ★ 태그 비교는 소문자로 통일
_SVG_MAX_LEN = 20000

def sanitize_svg(svg: str) -> str | None:
    """모델이 만든 인라인 SVG를 안전하고 반응형으로 만든다. 문제가 있으면 None."""
    if not svg or "<svg" not in svg:
        return None
    svg = svg.strip()
    if len(svg) > _SVG_MAX_LEN:
        print(f"[svg] 제외(너무 큼 {len(svg)}자)", flush=True)
        return None

    # 스크립트·외부 참조 제거
    svg = re.sub(r"<\s*(script|foreignObject|image|iframe)\b[\s\S]*?<\s*/\s*\1\s*>", "", svg, flags=re.I)
    svg = re.sub(r"<\s*(script|foreignObject|image|iframe)\b[^>]*/?>", "", svg, flags=re.I)
    svg = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", svg, flags=re.I)

    # 외부 URL 참조 금지 (href/xlink:href/url())
    has_ext_href = re.search(r"(?:xlink:)?href\s*=\s*[\"']\s*(?:https?:)?//", svg, re.I)
    has_ext_url  = any("http" in u for u in re.findall(r"url\(([^)]*)\)", svg))
    if has_ext_href or has_ext_url:
        print("[svg] 제외(외부 URL 참조)", flush=True)
        return None

    # 허용 태그만 (대소문자 무시)
    tags = {t.lower() for t in re.findall(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9]*)", svg)}
    bad = tags - _SVG_ALLOWED_LOWER
    if bad:
        print(f"[svg] 제외(허용 안 된 태그: {sorted(bad)})", flush=True)
        return None

    # viewBox 필수 — 없으면 반응형이 안 됨
    m = re.search(r"<svg\b[^>]*>", svg, re.I)
    if not m or not re.search(r"viewBox\s*=", m.group(0), re.I):
        print("[svg] 제외(viewBox 없음)", flush=True)
        return None

    # 고정 width/height 제거 (CSS가 100%로 처리)
    head = m.group(0)
    new_head = re.sub(r"\s(?:width|height)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", head, flags=re.I)
    svg = svg.replace(head, new_head, 1)
    return svg


def build_figures(data: dict) -> list:
    """figures 조각을 검증해 사용 가능한 것만 남긴다."""
    figs = data.get("figures") or []
    if isinstance(figs, dict):
        figs = [figs]
    out = []
    for f in figs:
        if not isinstance(f, dict):
            continue
        clean = sanitize_svg(f.get("svg", "") or "")
        if not clean:
            continue
        out.append({"svg": clean,
                    "caption": (f.get("caption") or f.get("title") or "").strip()})
        if len(out) >= 3:
            break
    return out


def _normalize_markup(html: str) -> str:
    if not html: return ""
    html = re.sub(r'</?(?:cite|abbr|mark|small|time|sup|sub|figure|figcaption|article|header|footer|main|aside)\b[^>]*>', '', html, flags=re.I)
    def _to_span(cls, h):
        pat = re.compile(r"<div(\s+class=['\"]%s['\"][^>]*)>((?:(?!<div\b).)*?)</div>" % cls, re.I|re.S)
        prev=None
        while prev!=h:
            prev=h; h=pat.sub(lambda m: "<span%s>%s</span>" % (m.group(1), m.group(2)), h)
        return h
    for cls in ("ctitle","ctag","en","lab"):
        html = _to_span(cls, html)
    def _kt(m):
        en = re.search(r"kt-en['\"]\s*>(.*?)<", m.group(0), re.S)
        ko = re.search(r"kt-ko['\"]\s*>(.*?)<", m.group(0), re.S)
        de = re.search(r"kt-desc['\"]\s*>(.*?)<", m.group(0), re.S)
        en=(en.group(1).strip() if en else ""); ko=(ko.group(1).strip() if ko else ""); de=(de.group(1).strip() if de else "")
        head = en + ((" ("+ko+")") if ko else "")
        return "<div class='kt'><h4>%s</h4><p>%s</p></div>" % (head, de)
    html = re.sub(r"<div class=['\"]kt['\"]>.*?</div>\s*(?=<div class=['\"]kt['\"]>|</div>)", _kt, html, flags=re.I|re.S)
    html = re.sub(r"<div class=['\"]analogy['\"]>\s*<b>\s*비유\s*:?\s*</b>",
                  "<div class='analogy'><div class='ah'><span class='em'>비유</span>비유</div><p>", html, flags=re.I)
    html = re.sub(r"<div class=['\"]kt['\"]>\s*<h4>\s*</h4>\s*<p>\s*</p>\s*</div>", "", html, flags=re.I)
    html = re.sub(r"<div class=['\"]keyterm['\"]>\s*</div>", "", html, flags=re.I)
    return html

def _drop_unverified_images(data: dict) -> dict:
    imgs = data.get("images") or []
    good = set()
    for im in imgs:
        u = (im or {}).get("url","")
        if validate_image_url(u):
            good.add(u)
    def _img(m):
        tag = m.group(0)
        src = re.search(r'src\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        if not src: return ''
        u = src.group(1)
        if u in good or validate_image_url(u):
            good.add(u); return tag
        return ''
    data["section_html"] = _ALLOWED_IMG_RE.sub(_img, data.get("section_html",""))
    data["images"] = [im for im in imgs if (im or {}).get("url") in good]
    return data


def generate_topic_background(passage: str, passage_dir, label: str = "",
                              save_step_fn=None, load_step_fn=None,
                              step_name="topic_background", max_uses=5) -> dict:
    # 지문 1개에서 소재 2~3개를 교차 확인하려면 5회는 빠듯하다. 하한을 두고 env 로 조절 가능.
    try:
        max_uses = max(int(max_uses or 5), int(os.environ.get("TB_SEARCH_MAX_USES", "8")))
    except Exception:
        max_uses = 8

    # 1) 캐시 우선
    if load_step_fn is not None:
        try:
            cached = load_step_fn(passage_dir, step_name)
            if cached and cached.get("section_html"):
                ver = cached.get("_step_version")
                broken = cached.get("_parse_fallback") and not cached.get("script_js")
                if ver != GEN_VERSION:
                    print(f"[topic_background] {label} 캐시 버전 불일치({ver}≠{GEN_VERSION}) → 재생성", flush=True)
                elif broken:
                    print(f"[topic_background] {label} 캐시가 토글 유실 상태 → 재생성", flush=True)
                else:
                    print(f"[topic_background] {label} 캐시 히트", flush=True)
                    return cached
        except Exception:
            pass

    # 2) 생성
    anchor_hint = re.sub(r'[^0-9a-zA-Z]+', '_', (label or "x")).strip('_').lower() or "x"
    user = GEN_USER_TMPL.format(label=label or "지문", passage=passage.strip(),
                                anchor_hint=anchor_hint)

    # 1차: web_search 켜고 생성
    print(f"[topic_background] {label} 생성 시작 (web_search, max_uses={max_uses})", flush=True)
    system_prompt = GEN_SYSTEM + load_exam_context()
    raw = call_claude_with_search(system_prompt, user, max_uses=max_uses, max_tokens=20000)
    data = _try_parse(raw)

    # 2차 폴백: 파싱 실패 시 검색 끄고 재시도
    if data is None:
        print(f"[topic_background] {label} 1차 파싱 실패 → 검색 없이 재시도. raw앞부분: {raw[:160]!r}", flush=True)
        try:
            fb_user = user + "\n\n[중요] 이번엔 검색하지 말고, 위 지시의 JSON 객체 하나만 즉시 출력하라. 코드블록 금지. 인용 태그 절대 금지. section_html 내 속성은 모두 큰따옴표 사용."
            body = {"model":MODEL,"max_tokens":20000,"temperature":0.2,
                    "system":system_prompt,"messages":[{"role":"user","content":fb_user}]}
            fb = _curl_messages(body)
            fbtexts = [b.get("text","") for b in fb.get("content",[]) if b.get("type")=="text"]
            data = _try_parse("\n".join(fbtexts))
            if data is not None:
                data["_no_search"] = True   # 웹 검증 없이 만든 자료 — 사실 확인 주의
        except Exception as e:
            print(f"[topic_background] {label} 폴백 호출 에러: {e}", flush=True)

    if data is None:
        raise ValueError(f"topic_background JSON 파싱 최종 실패 ({label})")

    # 3) 보정 + 새니타이즈
    anchor = re.sub(r'[^0-9a-zA-Z_]+','', data.get("anchor") or f"tb_{anchor_hint}") or f"tb_{anchor_hint}"
    data["anchor"] = anchor
    sec = data.get("section_html","") or ""
    if "<section" in sec:
        sec = re.sub(r'(<section[^>]*\bid=)["\'][^"\']*["\']', r'\1"%s"' % anchor, sec, count=1)
        if "id=" not in sec.split(">",1)[0]:
            sec = sec.replace("<section", f'<section id="{anchor}"', 1)
    else:
        sec = f'<section id="{anchor}">{sec}</section>'
    data["section_html"] = _normalize_markup(_strip_dangerous(sec))
    data["overview_html"] = _normalize_markup(_strip_dangerous(data.get("overview_html","") or ""))
    # 사진 정책: 외부 이미지는 쓰지 않는다 — <img>가 있으면 무조건 제거
    data["section_html"] = _ALLOWED_IMG_RE.sub("", data["section_html"])
    data["overview_html"] = _ALLOWED_IMG_RE.sub("", data["overview_html"])
    data.setdefault("chip", (label or "지문") + " 📘")
    data.setdefault("script_js", "")
    data["_step_version"] = GEN_VERSION

    # 4) 그림(SVG 모식도) 주입 — 각 카드의 .inpassage 바로 앞에 끼워 넣는다
    try:
        figs = build_figures(data)
        data["figures"] = figs
        data["images"] = []          # 외부 사진은 더 이상 쓰지 않는다
        already = ("class='fig'" in data["section_html"]) or ('class="fig"' in data["section_html"])
        if figs and not already:
            def _figtag(f):
                cap = ('<div class="figcap">%s</div>' % _html.escape(f["caption"])) if f["caption"] else ""
                return '<div class="fig">%s%s</div>' % (f["svg"], cap)

            sec = data["section_html"]
            positions, start = [], 0
            while True:
                idx = sec.find('class="inpassage"', start)
                if idx == -1:
                    idx = sec.find("class='inpassage'", start)
                if idx == -1:
                    break
                pre = sec.rfind("<div", 0, idx)
                if pre != -1:
                    positions.append(pre)
                start = idx + 10
            if positions:
                for i in range(min(len(figs), len(positions)) - 1, -1, -1):
                    sec = sec[:positions[i]] + _figtag(figs[i]) + sec[positions[i]:]
                data["section_html"] = sec
            else:
                data["section_html"] = sec.replace("</section>", _figtag(figs[0]) + "</section>", 1)
        print(f"[topic_background] {label} 모식도 {len(figs)}개 삽입", flush=True)
    except Exception as _e:
        print(f"[topic_background] {label} 모식도 주입 경고: {_e}", flush=True)

    # 5) 토글 인터랙티브 HTML + JS
    try:
        tg = data.get("toggle") or {}
        items = tg.get("items") or []
        items = [it for it in items if isinstance(it,dict) and it.get("label") and it.get("body")][:3]
        if len(items) >= 2:
            tid = "tg_" + re.sub(r'[^0-9a-zA-Z_]','',data["anchor"])
            btns = "".join(
                "<button data-k='%d'%s>%s</button>" % (i, (" class='on'" if i==0 else ""), _html.escape(it["label"]))
                for i,it in enumerate(items))
            box = ("<div class='interbox'><div class='ihead'><span class='tap'>TAP</span>%s</div>"
                   "<div class='ctrls' id='%s_c'>%s</div><div class='panel' id='%s_p'></div></div>") % (
                   _html.escape(tg.get("title","비교해 보세요")), tid, btns, tid)
            import json as _j
            jdata = _j.dumps([{"t":it["label"],"d":_strip_dangerous(it["body"])} for it in items], ensure_ascii=False)
            js = (
                "(function(){var D=" + jdata + ";"
                "var c=document.getElementById('" + tid + "_c'),p=document.getElementById('" + tid + "_p');"
                "if(!c||!p)return;"
                "function set(k){p.innerHTML='<span class=\"em\">'+D[k].t+'</span><br>'+D[k].d;"
                "[].forEach.call(c.children,function(b){b.classList.toggle('on',(+b.dataset.k)===k);});}"
                "c.addEventListener('click',function(e){if(e.target.dataset.k)set(+e.target.dataset.k);});"
                "set(0);})();"
            )
            ki = data["section_html"].rfind("<div class='keyterm'")
            if ki == -1: ki = data["section_html"].rfind('<div class="keyterm"')
            if ki != -1:
                data["section_html"] = data["section_html"][:ki] + box + data["section_html"][ki:]
            else:
                data["section_html"] = data["section_html"].replace("</section>", box+"</section>",1)
            data["script_js"] = (data.get("script_js","") + "\n" + js).strip()
        else:
            data["_no_toggle"] = True
            print(f"[topic_background] {label} ⚠ toggle 항목 부족({len(items)}개) — 인터랙티브 없이 생성됨", flush=True)
    except Exception as _e:
        data["_no_toggle"] = True
        print(f"[topic_background] {label} 토글 빌드 경고: {_e}", flush=True)

    print(f"[topic_background] {label} 생성 완료 (img {len(data.get('images',[]))}개)", flush=True)

    # 6) 캐시 저장
    if save_step_fn is not None:
        try: save_step_fn(passage_dir, step_name, data)
        except Exception as e:
            print(f"[topic_background] {label} 캐시 저장 실패: {e}", flush=True)
    return data
