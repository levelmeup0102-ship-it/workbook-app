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

# 이미지 허용 도메인 (위키 계열만)
_IMG_ALLOW = ("upload.wikimedia.org",)

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
        if stop == "tool_use":
            messages.append({"role":"user","content":"계속 진행해서 최종 결과만 JSON으로 주세요."})
            continue
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
        if ("host_not_allowed" in head) or ("x-deny-reason" in head) or (not head.strip()):
            return True
        first = head.split("\n")[0] if head else ""
        if "404" in first or "410" in first:
            return False
        return True
    except Exception:
        return True

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
            if src and validate_image_url(src):
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
    if src and validate_image_url(src):
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
# ★ 개선된 JSON 파서 (_try_parse v2)
# =========================================================================
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
                r'"' + key + r'"\s*:\s*"([\s\S]*?)"(?=\s*,\s*"(?:anchor|chip|section_html|overview_html|script_js|image_queries|toggle|_step_version)"|\\s*\})',
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

        # toggle 객체 — 중첩 구조라 간단히 처리
        tg_m = re.search(r'"toggle"\s*:\s*(\{[\s\S]*?\})\s*(?=,\s*"|}\s*$)', txt)
        if tg_m:
            try:
                result["toggle"] = json.loads(tg_m.group(1))
            except Exception:
                pass

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
- 마지막 카드는 .card danger 로 '시험 포인트' 카드를 만든다: 이 소재가 빈칸·순서·요약·어법으로 어떻게 변형 출제될지, 학생이 헷갈릴 함정은 무엇인지.
- 'theme' 와 'overview' 는 지문의 논리 흐름(주제·전환)을 한 줄로 요약한다.

[이미지 — 텍스트로 직접 넣지 말 것]
- <img> 태그를 직접 만들지 마라. 대신 출력 JSON 의 "image_queries" 에 위키백과 영문 표제어를 2~3개 넣어라.
  중요: 인물 사진만 넣지 말 것. 지문 '소재' 자체를 보여주는 대상을 우선한다.
  반드시 구체적이고 실재하는 영문 위키 표제어로(추상어·일반명사 금지).

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
  "image_queries": ["English Wikipedia title 1", "..."],
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

이 지문의 핵심 소재를 파악하고, web_search로 사실(학자/연도/지명/수치/연구)과
(인물·장소가 핵심이면) Wikimedia 이미지 URL을 확인한 뒤, 시스템 지시의 JSON을 출력하라.
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
    # 1) 캐시 우선
    if load_step_fn is not None:
        try:
            cached = load_step_fn(passage_dir, step_name)
            if cached and cached.get("section_html"):
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
    raw = call_claude_with_search(GEN_SYSTEM, user, max_uses=max_uses, max_tokens=20000)
    data = _try_parse(raw)

    # 2차 폴백: 파싱 실패 시 검색 끄고 재시도
    if data is None:
        print(f"[topic_background] {label} 1차 파싱 실패 → 검색 없이 재시도. raw앞부분: {raw[:160]!r}", flush=True)
        try:
            fb_user = user + "\n\n[중요] 이번엔 검색하지 말고, 위 지시의 JSON 객체 하나만 즉시 출력하라. 코드블록 금지. 인용 태그 절대 금지. section_html 내 속성은 모두 큰따옴표 사용."
            body = {"model":MODEL,"max_tokens":20000,"temperature":0.2,
                    "system":GEN_SYSTEM,"messages":[{"role":"user","content":fb_user}]}
            fb = _curl_messages(body)
            fbtexts = [b.get("text","") for b in fb.get("content",[]) if b.get("type")=="text"]
            data = _try_parse("\n".join(fbtexts))
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
    data = _drop_unverified_images(data)
    data.setdefault("chip", (label or "지문") + " 📘")
    data.setdefault("script_js", "")
    data["_step_version"] = "v3"

    # 4) 이미지 주입
    try:
        queries = data.get("image_queries") or []
        if isinstance(queries, str): queries = [queries]
        existing = [im for im in (data.get("images") or []) if validate_image_url((im or {}).get("url",""))]
        fetched = fetch_images_for([q for q in queries if q], max_imgs=2) if queries else []
        all_imgs, seen = [], set()
        for im in (existing + fetched):
            u=(im or {}).get("url","")
            if u and u not in seen:
                seen.add(u); all_imgs.append(im)
        data["images"] = all_imgs[:3]
        if data["images"] and "class='fig'" not in data["section_html"] and 'class="fig"' not in data["section_html"]:
            def _figtag(im):
                return ("<div class=\"fig\"><img src=\"%s\" alt=\"%s\" style=\"width:100%%;border-radius:10px\">"
                        "<div class=\"figcap\">%s · 출처: Wikimedia Commons</div></div>") % (
                        im["url"], _html.escape(im.get("alt","")), _html.escape(im.get("alt","")))
            sec = data["section_html"]
            positions = []
            start = 0
            while True:
                idx = sec.find('class="inpassage"', start)
                if idx == -1:
                    idx = sec.find("class='inpassage'", start)
                if idx == -1: break
                pre = sec.rfind("<div", 0, idx)
                if pre != -1: positions.append(pre)
                start = idx + 10
            if positions:
                imgs = data["images"]
                for i in range(min(len(imgs), len(positions))-1, -1, -1):
                    sec = sec[:positions[i]] + _figtag(imgs[i]) + sec[positions[i]:]
                data["section_html"] = sec
            else:
                data["section_html"] = sec.replace("</section>", _figtag(data["images"][0])+"</section>",1)
    except Exception as _e:
        print(f"[topic_background] {label} 이미지 주입 경고: {_e}", flush=True)

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
    except Exception as _e:
        print(f"[topic_background] {label} 토글 빌드 경고: {_e}", flush=True)

    print(f"[topic_background] {label} 생성 완료 (img {len(data.get('images',[]))}개)", flush=True)

    # 6) 캐시 저장
    if save_step_fn is not None:
        try: save_step_fn(passage_dir, step_name, data)
        except Exception as e:
            print(f"[topic_background] {label} 캐시 저장 실패: {e}", flush=True)
    return data
