# -*- coding: utf-8 -*-
"""
수업배경자료 자동 생성 (web_search 사용).
topic_background.py 에서 import 해서 씀. pipeline 의 call_claude 인프라(curl)를 그대로 활용.
"""
import os, re, json, time, subprocess, tempfile, html as _html

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"

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
    """web_search 도구를 켠 멀티턴 호출. 최종 어시스턴트 text 합쳐 반환.
    Anthropic이 서버에서 검색을 실행하므로, 우리는 tool_use가 끝날 때까지 턴만 이어준다."""
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
        # web_search는 Anthropic이 server-side로 처리 → 보통 같은 응답에 결과까지 포함됨.
        # stop_reason 이 tool_use 면 한 번 더 이어준다(드묾).
        if stop == "tool_use":
            messages.append({"role":"user","content":"계속 진행해서 최종 결과만 JSON으로 주세요."})
            continue
        # 텍스트 블록만 추출
        texts = [b.get("text","") for b in content if b.get("type")=="text"]
        return "\n".join(t for t in texts if t).strip()
    # 마지막 턴 텍스트
    texts = [b.get("text","") for b in messages[-1]["content"] if isinstance(b,dict) and b.get("type")=="text"]
    return "\n".join(texts).strip()

_WIKI_IMG_RE = re.compile(r'^https://upload\.wikimedia\.org/wikipedia/[^\s"\'<>]+\.(?:jpg|jpeg|png|svg|gif|webp)(?:\?[^\s"\'<>]*)?$', re.I)

def validate_image_url(url: str, timeout=8) -> bool:
    """위키 직접 이미지 URL 검증.
    1) 도메인+형식(upload.wikimedia.org/.../*.jpg|png|svg)이 맞아야 한다(필수).
    2) 형식이 맞으면 통과시킨다. (네트워크 HEAD 검증은 가능할 때만 보조로 시도하되,
       차단/타임아웃 등으로 실패해도 형식이 맞으면 제거하지 않는다 — 환경별 네트워크 차이 때문.)
    """
    if not url or not _WIKI_IMG_RE.match(url.strip()):
        return False
    try:
        r = subprocess.run(['curl','-s','-I','-L','--max-time',str(timeout),url],
                           capture_output=True, timeout=timeout+3)
        head = r.stdout.decode('utf-8','replace').lower()
        # 프록시/방화벽 차단(host_not_allowed 등)이면 검증 불가 → 형식만으로 통과 인정
        if ("host_not_allowed" in head) or ("x-deny-reason" in head) or (not head.strip()):
            return True
        first = head.split("\n")[0] if head else ""
        # 위키 서버가 실제로 404/410(없는 파일)을 주면 탈락. 그 외는 통과.
        if "404" in first or "410" in first:
            return False
        return True
    except Exception:
        # 네트워크 검증 불가 환경 → 형식 통과로 인정
        return True

if __name__ == "__main__":
    # 이미지 검증 단위 테스트 (네트워크)
    ok = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Albert_Einstein_1947.jpg/220px-Albert_Einstein_1947.jpg"
    bad = "https://example.com/fake.jpg"
    print("allow domain check:", any(d in ok for d in _IMG_ALLOW), any(d in bad for d in _IMG_ALLOW))


# =========================================================================
# 생성 프롬프트 + 후처리
# =========================================================================
GEN_SYSTEM = """너는 한국 고등학교 영어 내신 강사를 위한 '수업배경자료' 제작자다.
주어진 영어 지문 1개를 받아, 그 지문이 다루는 '소재(주제 대상)'의 배경을 깊이 파고드는
학습 카드 묶음을 만든다. 인물이 아니라 소재(개념·장소·현상·연구) 중심으로 설명한다.

[사실 검증 — 매우 중요]
- 학자 이름, 연도, 지명, 수치, 연구 출처 같은 사실은 반드시 web_search로 교차 확인한다.
- 확인 안 된 사실은 쓰지 않는다. 추정·환각 금지.
- 인물이나 장소가 핵심이면, web_search로 'Wikimedia Commons' 이미지를 찾는다.
  반드시 upload.wikimedia.org 로 시작하는 직접 이미지 URL(.jpg/.png)만 사용한다.
  없으면 이미지 없이 둔다(가짜 URL 절대 금지).

[출력 형식 — JSON only]
아래 키만 가진 JSON 하나만 출력한다. 코드블록/설명 금지.
{
  "anchor": "tb_xxx",                  // 영문/숫자/언더스코어 짧은 id
  "chip": "8-4 관찰 드로잉 🎨",         // hero 칩 라벨 (지문번호 + 짧은소재 + 이모지 1개)
  "section_html": "<section id='tb_xxx'> ... </section>",
  "overview_html": "<h3>... </h3><div class='flow'>...</div><p>...</p>",
  "images": [ {"url":"https://upload.wikimedia.org/...","alt":"...","credit":"Wikimedia Commons"} ]
}

[section_html 작성 규칙 — 반드시 이 클래스만 사용]
- 최상위는 <section id='{anchor}'> ... </section>.
- 순서: <div class='sec-num'>지문 · NN</div> → <h2><span class='bar'></span>{한글소제목}</h2>
  → <div class='sec-en'>{영어 원제 추정}</div> → <span class='theme'>핵심 주제 — ...</span>
  → 카드 2~4개(.card / .card amber|danger|blue|purple) → <div class='keyterm'> kt 2개 </div>
- 카드 내부: .chead>(.ctitle>.en, .ctag), .csub, .cbody(<b>로 핵심어만), 필요시 .quote/.analogy/.ctable,
  그리고 거의 항상 .inpassage(<span class='lab'>지문 속</span> ❶❷ 문장연결).
- 어려운 개념엔 .analogy(비유) 또는 .ctable(비교표)를 넣어 직관적으로.
- 인물/장소 이미지가 있으면 .fig 안에 <img src='{검증된 위키 URL}' alt='...' style='width:100%;border-radius:10px'>
  + <div class='figcap'>설명 + 출처: Wikimedia Commons</div>.
- 톤: 친근한 한국어 구어체("~예요","~죠"). <b>는 핵심어에만. 사실 위주.
- 절대 <style>,<script>,onclick,onerror,iframe,외부 CSS 금지. 위 클래스 외 새 class 만들지 말 것.
- 이미지 src는 upload.wikimedia.org 외 금지.

[overview_html]
- <h3>{이모지} {지문번호} — {소재}</h3> + <div class='flow'>...<span class='node'>..</span><span class='ar'>→</span>..</div>
  + <p style='font-size:13.5px;color:#48605b;font-weight:600;margin-top:10px'>{한 줄 정리}</p>
"""

GEN_USER_TMPL = """다음은 분석할 영어 지문이다. 지문 번호: {label}

<passage>
{passage}
</passage>

이 지문의 핵심 소재를 파악하고, web_search로 사실(학자/연도/지명/수치/연구)과
(인물·장소가 핵심이면) Wikimedia 이미지 URL을 확인한 뒤, 시스템 지시의 JSON을 출력하라.
anchor 는 "tb_{anchor_hint}" 형식으로 만들고, 지문 번호는 "{label}"를 사용하라.
JSON 외 아무것도 출력하지 마라."""

# ---- 새니타이즈: 허용 클래스/태그만, 이미지 도메인 화이트리스트 ----
_ALLOWED_IMG_RE = re.compile(r'<img\b[^>]*>', re.I)

def _strip_dangerous(html: str) -> str:
    if not html: return ""
    # script/style/iframe/on* 제거
    html = re.sub(r'<\s*(script|style|iframe)[^>]*>.*?<\s*/\s*\1\s*>', '', html, flags=re.I|re.S)
    html = re.sub(r'<\s*(script|style|iframe)[^>]*/?>', '', html, flags=re.I)
    html = re.sub(r'\son\w+\s*=\s*"[^"]*"', '', html, flags=re.I)
    html = re.sub(r"\son\w+\s*=\s*'[^']*'", '', html, flags=re.I)
    html = re.sub(r'\son\w+\s*=\s*[^\s>]+', '', html, flags=re.I)
    # 위키 외 이미지 제거
    def _img(m):
        tag = m.group(0)
        src = re.search(r'src\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        if not src or not any(d in src.group(1) for d in _IMG_ALLOW):
            return ''  # 비허용 이미지는 통째 제거
        return tag
    html = _ALLOWED_IMG_RE.sub(_img, html)
    return html

def _drop_unverified_images(data: dict) -> dict:
    """images 리스트와 section_html 의 img 를 실제 검증(HEAD)으로 거름."""
    imgs = data.get("images") or []
    good = set()
    for im in imgs:
        u = (im or {}).get("url","")
        if validate_image_url(u):
            good.add(u)
    # section_html 의 img 중 검증 통과 못 한 것 제거
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
    """지문 1개 → 수업배경자료 조각(dict) 생성. 캐시 우선.
    save_step_fn/load_step_fn 은 pipeline.save_step/load_step 을 주입받는다.
    """
    # 1) 캐시 우선
    if load_step_fn is not None:
        try:
            cached = load_step_fn(passage_dir, step_name)
            if cached and cached.get("section_html"):
                return cached
        except Exception:
            pass

    # 2) 생성
    anchor_hint = re.sub(r'[^0-9a-zA-Z]+', '_', (label or "x")).strip('_').lower() or "x"
    user = GEN_USER_TMPL.format(label=label or "지문", passage=passage.strip(),
                                anchor_hint=anchor_hint)
    raw = call_claude_with_search(GEN_SYSTEM, user, max_uses=max_uses)

    # 3) JSON 파싱(견고)
    txt = re.sub(r'^```json\s*','',raw.strip()); txt = re.sub(r'\s*```$','',txt).strip()
    try:
        data = json.loads(txt)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', txt)
        if not m: raise ValueError("topic_background JSON 파싱 실패")
        data = json.loads(m.group())

    # 4) 보정 + 새니타이즈
    anchor = re.sub(r'[^0-9a-zA-Z_]+','', data.get("anchor") or f"tb_{anchor_hint}") or f"tb_{anchor_hint}"
    data["anchor"] = anchor
    sec = data.get("section_html","") or ""
    # section id 를 anchor 로 통일
    if "<section" in sec:
        sec = re.sub(r'(<section[^>]*\bid=)["\'][^"\']*["\']', r'\1"%s"' % anchor, sec, count=1)
        if "id=" not in sec.split(">",1)[0]:
            sec = sec.replace("<section", f'<section id="{anchor}"', 1)
    else:
        sec = f'<section id="{anchor}">{sec}</section>'
    data["section_html"] = _strip_dangerous(sec)
    data["overview_html"] = _strip_dangerous(data.get("overview_html","") or "")
    data = _drop_unverified_images(data)
    data.setdefault("chip", (label or "지문") + " 📘")
    data.setdefault("script_js", "")   # 자동생성은 인터랙티브 JS 없음(정적 카드/표/비유/이미지)
    data["_step_version"] = "v1"

    # 5) 캐시 저장
    if save_step_fn is not None:
        try: save_step_fn(passage_dir, step_name, data)
        except Exception:
            pass
    return data
