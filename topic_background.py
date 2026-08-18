# -*- coding: utf-8 -*-
"""
수업배경자료(0회독 3번째 탭) 렌더러.
- LLM을 돌리지 않습니다. step_cache의 step_name="topic_background" 레코드(지문별 조각)를
  cache_key로 불러와 LevelMeUp 디자인 한 벌로 조립합니다.
- main.py 의 /api/secret-note 에서 mode=="topic" 일 때 호출하세요.

조각(step_cache.data) 스키마:
  { "anchor": "tb_8_4",
    "section_html": "<section id='tb_8_4'> ... </section>",
    "script_js": "(function(){ ... })();",   # 이 지문 전용 인터랙티브
    "chip": "8-4 관찰 드로잉 🎨",
    "overview_html": "<h3>...</h3><div class='flow'>...</div><p>...</p>",
    "_step_version": "v1" }
"""

import os
import base64
import logging

TOPIC_STEP_NAME = "topic_background"
TOPIC_STEP_VERSION = "v5"   # tb_generate.GEN_VERSION 과 맞춘다

# 폰트 (LevelMeUp 디자인 시스템)
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=Gothic+A1:wght@400;500;700;800;900&display=swap" rel="stylesheet">'
)

# ── 폰트 임베드(서브셋) ──────────────────────────────────────────
# 브라우저 "웹페이지로 저장" 시 구글폰트 <link>가 로컬 경로로 깨지는 문제를 원천 차단한다.
# fonts/ 폴더에 아래 TTF가 있고 fonttools(+brotli)가 설치돼 있으면,
# 페이지에 실제 쓰인 글자만 골라 woff2로 서브셋해 <style>에 base64로 심는다.
# 하나라도 없으면(폰트 파일·라이브러리 부재) 조용히 기존 외부 <link>로 폴백한다.
#   · 폰트 파일: google/fonts 저장소의 OFL 폰트 (BlackHanSans / GothicA1)
#   · 폴더 위치: 환경변수 LEVELMEUP_FONT_DIR, 없으면 이 파일 옆 fonts/
# fontTools 서브셋은 기본적으로 DEBUG 로그를 수천 줄 쏟아낸다.
# 서버(Railway 등)의 로그 초당 제한을 초과해 앱이 지연되므로 ERROR 이상만 남긴다.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# 같은 글자 집합을 매 렌더마다 다시 서브셋하지 않도록 결과를 캐싱한다.
_FONT_EMBED_CACHE = {}

_EMBED_FONTS = [
    ("Black Han Sans", 400, "BlackHanSans-Regular.ttf"),
    ("Gothic A1", 400, "GothicA1-Regular.ttf"),
    ("Gothic A1", 500, "GothicA1-Medium.ttf"),
    ("Gothic A1", 700, "GothicA1-Bold.ttf"),
    ("Gothic A1", 800, "GothicA1-ExtraBold.ttf"),
    ("Gothic A1", 900, "GothicA1-Black.ttf"),
]
# 항상 포함할 기본 문자(버튼·영문 라벨·숫자·기호 보호)
_FONT_BASE_CHARS = (
    " 0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    ".,!?:;·—…\"\'()[]{}%&/+-→↑↓✓✗★"
)


def _font_dir():
    return (os.environ.get("LEVELMEUP_FONT_DIR")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))


def _embed_fonts_css(*texts):
    """페이지에 쓰인 글자만 서브셋해 @font-face(base64 woff2)로 임베드한 <style>을 반환.
    fonttools/brotli 미설치 또는 폰트 파일 없음 → None(외부 <link> 폴백)."""
    try:
        from fontTools import subset as _subset
        from fontTools.ttLib import TTFont
        import io as _io
    except Exception:
        return None
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # 로그 폭주 방지(이중 안전장치)
    fdir = _font_dir()
    if not all(os.path.exists(os.path.join(fdir, f)) for _, _, f in _EMBED_FONTS):
        return None
    chars = set(_FONT_BASE_CHARS)
    for t in texts:
        if t:
            chars |= set(t)
    # 폰트에 없는 이모지 등 BMP 밖 문자는 제외(서브셋 대상 아님)
    unicodes = tuple(sorted({ord(c) for c in chars if ord(c) <= 0xFFFF}))
    cached = _FONT_EMBED_CACHE.get(unicodes)   # 같은 글자 집합이면 재서브셋 생략
    if cached is not None:
        return cached
    faces = []
    try:
        for family, weight, fname in _EMBED_FONTS:
            font = TTFont(os.path.join(fdir, fname))
            opts = _subset.Options(flavor="woff2", desubroutinize=True,
                                   layout_features=[], notdef_outline=False,
                                   recalc_bounds=False, recalc_timestamp=False)
            ss = _subset.Subsetter(options=opts)
            ss.populate(unicodes=unicodes)
            ss.subset(font)
            font.flavor = "woff2"          # ★ 서브셋 후 flavor 지정해야 실제 woff2(brotli)로 저장됨
            buf = _io.BytesIO(); font.save(buf)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            faces.append(
                "@font-face{font-family:\'%s\';font-style:normal;font-weight:%d;"
                "font-display:swap;src:url(data:font/woff2;base64,%s) format(\'woff2\')}"
                % (family, weight, b64))
    except Exception as e:
        print(f"[fonts] 서브셋 실패 → 외부 링크 폴백: {e}", flush=True)
        return None
    css = "<style>" + "".join(faces) + "</style>"
    _FONT_EMBED_CACHE[unicodes] = css
    if len(_FONT_EMBED_CACHE) > 64:            # 메모리 무한증가 방지
        _FONT_EMBED_CACHE.pop(next(iter(_FONT_EMBED_CACHE)))
    print(f"[fonts] {len(unicodes)}자 서브셋 임베드 ({len(_EMBED_FONTS)}종)", flush=True)
    return css

# 디자인 시스템 CSS (스타터 템플릿과 동일 — 절대 수정 금지 권장)
LEVELMEUP_CSS = r""":root{
    --ink:#0f2e2a;--teal:#16b89b;--teal-deep:#0e8f79;--mint:#e6f7f2;--mint-2:#f3fbf8;
    --danger:#e8493f;--danger-soft:#fde9e7;--amber:#e0922a;--amber-soft:#fbf0dd;
    --blue:#2f7de0;--blue-soft:#e4eefc;--purple:#7a5cd6;--purple-soft:#efeafb;
    --gray:#9aa7a3;--line:#dbe8e3;--paper:#f7faf8;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    font-family:'Gothic A1',sans-serif;
    background:
      radial-gradient(circle at 12% 8%, rgba(22,184,155,.10), transparent 40%),
      radial-gradient(circle at 88% 92%, rgba(22,184,155,.08), transparent 42%),
      var(--paper);
    color:var(--ink);line-height:1.65;padding:32px 16px 80px;-webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:820px;margin:0 auto}
  .topbar{display:flex;align-items:center;justify-content:space-between;font-size:13px;font-weight:700;color:var(--teal-deep);letter-spacing:.5px;margin-bottom:14px}
  .topbar .brand{display:flex;align-items:center;gap:8px}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--teal);box-shadow:0 0 0 4px var(--mint)}
  .topbar .meta{color:var(--gray);font-weight:600}

  .hero{background:linear-gradient(135deg,var(--ink) 0%,#16403a 100%);border-radius:24px;padding:38px 34px 32px;color:#fff;position:relative;overflow:hidden;box-shadow:0 24px 50px -24px rgba(15,46,42,.55)}
  .hero::after{content:"";position:absolute;right:-40px;top:-40px;width:200px;height:200px;background:radial-gradient(circle,rgba(22,184,155,.45),transparent 70%)}
  .kicker{display:inline-block;font-size:12.5px;font-weight:800;letter-spacing:1.5px;color:#7fe9d2;background:rgba(22,184,155,.16);padding:6px 12px;border-radius:999px;margin-bottom:16px}
  .hero h1{font-family:'Black Han Sans',sans-serif;font-weight:400;font-size:clamp(26px,5.6vw,38px);line-height:1.2;letter-spacing:-.5px}
  .hero h1 .hl{color:#5fe6c9}
  .hero p{margin-top:14px;font-size:15px;color:#cfe9e3;max-width:94%}
  .herochips{margin-top:18px;display:flex;gap:8px;flex-wrap:wrap}
  .herochips a{font-size:12.5px;font-weight:800;color:#0f2e2a;background:#7fe9d2;padding:7px 13px;border-radius:999px;text-decoration:none;transition:.2s}
  .herochips a:hover{transform:translateY(-2px);background:#a7f0df}

  section{margin-top:48px;opacity:0;transform:translateY(18px);transition:.7s cubic-bezier(.2,.7,.2,1);scroll-margin-top:20px}
  section.in{opacity:1;transform:none}
  .sec-num{font-family:'Black Han Sans';color:var(--teal);font-size:15px;letter-spacing:1px}
  .sec-en{font-size:12.5px;color:var(--gray);font-weight:700;letter-spacing:.3px;margin-top:2px}
  h2{font-size:clamp(20px,4.5vw,26px);font-weight:900;letter-spacing:-.5px;margin:4px 0 6px}
  h2 .bar{display:inline-block;width:28px;height:5px;background:var(--teal);border-radius:3px;vertical-align:middle;margin-right:10px}
  .theme{font-size:14px;font-weight:700;color:var(--teal-deep);background:var(--mint);border-radius:10px;padding:10px 14px;margin:12px 0 22px;display:inline-block;line-height:1.6}

  .card{background:#fff;border:1.5px solid var(--line);border-radius:18px;padding:24px 22px;margin-bottom:16px;box-shadow:0 14px 34px -28px rgba(15,46,42,.5);position:relative;overflow:hidden}
  .card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--teal)}
  .card.amber::before{background:var(--amber)}
  .card.danger::before{background:var(--danger)}
  .card.blue::before{background:var(--blue)}
  .card.purple::before{background:var(--purple)}
  .chead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:4px}
  .ctitle{font-size:18px;font-weight:900;letter-spacing:-.3px}
  .ctitle .en{font-size:12.5px;color:var(--gray);font-weight:700;margin-left:3px}
  .ctag{font-size:11.5px;font-weight:800;padding:3px 10px;border-radius:999px;background:var(--mint);color:var(--teal-deep)}
  .ctag.amber{background:var(--amber-soft);color:var(--amber)}
  .ctag.danger{background:var(--danger-soft);color:var(--danger)}
  .ctag.blue{background:var(--blue-soft);color:var(--blue)}
  .ctag.purple{background:var(--purple-soft);color:var(--purple)}
  .csub{font-size:13px;color:var(--gray);font-weight:700;margin-bottom:13px}
  .cbody{font-size:14.3px;color:#33473f;font-weight:500;line-height:1.74}
  .cbody b{color:var(--ink);font-weight:800}
  .cbody+.cbody{margin-top:12px}
  .quote{margin:14px 0 4px;padding:12px 16px;background:var(--mint-2);border-left:4px solid var(--teal);border-radius:0 10px 10px 0;font-size:14px;font-weight:600;color:var(--ink)}
  .quote.amber{background:var(--amber-soft);border-left-color:var(--amber)}
  .quote.danger{background:var(--danger-soft);border-left-color:var(--danger)}
  .quote.blue{background:var(--blue-soft);border-left-color:var(--blue)}
  .quote .ko{display:block;color:var(--gray);font-weight:600;font-size:12.5px;margin-top:5px}
  .inpassage{margin-top:14px;font-size:13px;font-weight:700;color:var(--teal-deep);background:#fff;border:1.5px dashed var(--teal);border-radius:10px;padding:10px 13px}
  .inpassage .lab{font-family:'Black Han Sans';font-size:12px;margin-right:6px}
  .card.amber .inpassage{border-color:var(--amber);color:var(--amber)}
  .card.danger .inpassage{border-color:var(--danger);color:var(--danger)}
  .card.blue .inpassage{border-color:var(--blue);color:var(--blue)}
  .card.purple .inpassage{border-color:var(--purple);color:var(--purple)}

  .keyterm{display:grid;gap:10px;margin-top:16px}
  @media(min-width:620px){.keyterm{grid-template-columns:1fr 1fr}}
  .kt{background:var(--ink);color:#e8f6f2;border-radius:12px;padding:14px 16px}
  .kt h4{font-size:14px;font-weight:900;color:#7fe9d2;margin-bottom:4px}
  .kt p{font-size:13px;font-weight:500;color:#cfe9e3;line-height:1.6}

  .callout{background:var(--mint);border-left:5px solid var(--teal);border-radius:0 14px 14px 0;padding:18px 20px;margin-top:18px;font-size:14.5px;font-weight:600;color:var(--ink)}
  .callout b{color:var(--teal-deep)}

  .fig{margin:16px 0 4px;background:linear-gradient(180deg,#fbfdfc,#f1f8f5);border:1.5px solid var(--line);border-radius:14px;padding:14px}
  .fig svg{display:block;width:100%;height:auto}
.fig img{display:block;width:100%;height:auto;border-radius:10px}
  .figcap{font-size:12.5px;color:var(--gray);font-weight:700;text-align:center;margin-top:8px;line-height:1.5}

  .ctrls{display:flex;gap:7px;flex-wrap:wrap;margin:14px 0}
  .ctrls button{font-family:'Gothic A1',sans-serif;font-size:13px;font-weight:800;color:var(--teal-deep);background:#fff;border:1.5px solid var(--line);border-radius:999px;padding:8px 14px;cursor:pointer;transition:.18s}
  .ctrls button:hover{border-color:var(--teal)}
  .ctrls button.on{background:var(--teal);color:#fff;border-color:var(--teal);box-shadow:0 6px 16px -8px var(--teal-deep)}
  .ctrls.amber button{color:var(--amber)} .ctrls.amber button.on{background:var(--amber);border-color:var(--amber);color:#fff;box-shadow:none}
  .ctrls.danger button{color:var(--danger)} .ctrls.danger button.on{background:var(--danger);border-color:var(--danger);color:#fff;box-shadow:none}
  .ctrls.blue button{color:var(--blue)} .ctrls.blue button.on{background:var(--blue);border-color:var(--blue);color:#fff;box-shadow:none}
  .ctrls.purple button{color:var(--purple)} .ctrls.purple button.on{background:var(--purple);border-color:var(--purple);color:#fff;box-shadow:none}

  .panel{background:var(--mint-2);border-radius:12px;padding:15px 16px;font-size:14px;font-weight:600;color:var(--ink);line-height:1.7;min-height:54px;border:1px solid var(--line)}
  .panel b{color:var(--teal-deep)}
  .panel .em{display:inline-block;font-size:12px;font-weight:800;padding:2px 9px;border-radius:999px;background:var(--mint);color:var(--teal-deep);margin-bottom:7px}

  .interbox{background:linear-gradient(135deg,#fff,var(--mint-2));border:1.5px solid var(--teal);border-radius:16px;padding:18px;margin-top:16px}
  .interbox .ihead{font-size:13px;font-weight:900;color:var(--teal-deep);letter-spacing:.3px;margin-bottom:2px;display:flex;align-items:center;gap:7px}
  .interbox .ihead .tap{font-family:'Black Han Sans';font-size:11px;background:var(--teal);color:#fff;padding:2px 9px;border-radius:999px;letter-spacing:.5px}
  .interbox.danger{border-color:var(--danger)} .interbox.danger .ihead{color:var(--danger)} .interbox.danger .ihead .tap{background:var(--danger)}
  .interbox.amber{border-color:var(--amber)} .interbox.amber .ihead{color:var(--amber)} .interbox.amber .ihead .tap{background:var(--amber)}
  .interbox.blue{border-color:var(--blue)} .interbox.blue .ihead{color:var(--blue)} .interbox.blue .ihead .tap{background:var(--blue)}
  .interbox.purple{border-color:var(--purple)} .interbox.purple .ihead{color:var(--purple)} .interbox.purple .ihead .tap{background:var(--purple)}
  .ihint{font-size:12px;color:var(--gray);font-weight:700;margin:2px 0 0}

  .connect{background:linear-gradient(135deg,#fff,var(--mint-2));border:1.5px solid var(--line);border-radius:18px;padding:22px}
  .connect h3{font-size:16px;font-weight:900;margin-bottom:10px}
  .flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:13.5px;font-weight:800;color:var(--ink)}
  .flow .node{background:var(--mint);border:1px solid var(--teal);border-radius:10px;padding:8px 12px}
  .flow .ar{color:var(--teal);font-family:'Black Han Sans'}

  footer{margin-top:50px;text-align:center;font-size:12px;color:var(--gray);font-weight:600;line-height:1.8}
  footer b{color:var(--teal-deep)}

  .ctable{width:100%;border-collapse:separate;border-spacing:0;margin:14px 0 4px;font-size:13px;font-weight:700;overflow:hidden;border-radius:12px;border:1.5px solid var(--line)}
  .ctable th,.ctable td{padding:11px 10px;text-align:center;border-bottom:1px solid var(--line)}
  .ctable thead th{background:var(--ink);color:#e8f6f2;font-weight:900;font-size:12.5px}
  .ctable tbody tr:last-child td{border-bottom:none}
  .ctable .lcell{background:var(--mint-2);font-weight:900;color:var(--ink);text-align:left;padding-left:14px}
  .ctable .win{background:var(--mint)}
  .ctable .bad{background:var(--danger-soft);color:var(--danger);font-weight:900}
  .ctable .good{background:var(--mint);color:var(--teal-deep);font-weight:900}
  .tnote{font-size:12.5px;color:var(--gray);font-weight:700;line-height:1.6;margin-top:6px}
  .tnote b{color:var(--danger)}

  .analogy{background:linear-gradient(135deg,#fff,var(--amber-soft));border:1.5px solid var(--amber);border-radius:14px;padding:16px 18px;margin:16px 0 4px}
  .analogy .ah{font-size:13px;font-weight:900;color:var(--amber);margin-bottom:6px;display:flex;align-items:center;gap:7px}
  .analogy .ah .em{font-family:'Black Han Sans';font-size:11px;background:var(--amber);color:#fff;padding:2px 9px;border-radius:999px}
  .analogy p{font-size:14px;font-weight:600;color:#4a3a18;line-height:1.72}
  .analogy b{color:#7a5a1f;font-weight:900}

  .chain{margin-top:12px}
  .chain .step{display:flex;gap:11px;align-items:flex-start;padding:11px 13px;border-radius:11px;background:var(--mint-2);border:1px solid var(--line);margin-bottom:8px;opacity:0;transform:translateX(-8px);transition:.45s cubic-bezier(.2,.7,.2,1)}
  .chain .step.show{opacity:1;transform:none}
  .chain .step .n{flex:none;width:24px;height:24px;border-radius:50%;background:var(--danger);color:#fff;font-family:'Black Han Sans';font-size:13px;display:flex;align-items:center;justify-content:center;margin-top:1px}
  .chain .step p{font-size:13.5px;font-weight:600;color:var(--ink);line-height:1.6}
  .chain .step p b{color:var(--danger);font-weight:900}
  .chain .step.fin{background:var(--ink)}
  .chain .step.fin .n{background:#7fe9d2;color:var(--ink)}
  .chain .step.fin p{color:#e8f6f2}
  .chain .step.fin p b{color:#7fe9d2}
  /* teal-tone chain (silo→whole) */
  .chain.teal .step .n{background:var(--teal)}
  .chain.teal .step p b{color:var(--teal-deep)}

  /* two-faces compare grid */
  .twoface{display:grid;gap:10px;margin-top:14px}
  @media(min-width:560px){.twoface{grid-template-columns:1fr 1fr}}
  .ff{border-radius:13px;padding:15px 16px;border:1.5px solid var(--line)}
  .ff h4{font-size:14px;font-weight:900;margin-bottom:7px;display:flex;align-items:center;gap:6px}
  .ff ul{list-style:none;font-size:13px;font-weight:600;line-height:1.7}
  .ff li{padding-left:14px;position:relative}
  .ff li::before{content:"•";position:absolute;left:0;font-weight:900}
  .ff.bad{background:var(--danger-soft);border-color:var(--danger)}
  .ff.bad h4,.ff.bad li::before{color:var(--danger)}
  .ff.good{background:var(--mint);border-color:var(--teal)}
  .ff.good h4,.ff.good li::before{color:var(--teal-deep)}"""

# 공통 스크립트 (스크롤 페이드업) — 문서당 1회
TOPIC_COMMON_JS = r"""const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}})},{threshold:.10});
  document.querySelectorAll('section').forEach(s=>io.observe(s));
  /* 외부 이미지가 죽었을 때 깨진 아이콘 대신 그림 박스를 통째로 없앤다 */
  document.querySelectorAll('.fig img').forEach(function(img){
    function drop(){ var f=img.closest('.fig'); if(f) f.remove(); }
    img.addEventListener('error', drop);
    if (img.complete && img.naturalWidth === 0) drop();
  });"""


def _hero(school_name, chips_html, n):
    sub = "지문 속 소재 깊이 파기" if n == 1 else f"{n}개 지문 · 지문 속 소재 깊이 파기"
    return (
        '<div class="topbar"><div class="brand"><span class="dot"></span>'
        + (school_name or "LevelMeUp") + ' · 수업배경자료</div>'
        '<div class="meta">0회독 · 수업배경자료</div></div>'
        '<div class="hero">'
        '<span class="kicker">' + sub + '</span>'
        '<h1>지문이 다룬 <span class="hl">그 소재들</span>,<br>실제로 어떤 이야기일까</h1>'
        '<p>각 지문의 핵심 소재를 배경부터 파고들었어요. 토글·단계 데모를 직접 만져보세요.</p>'
        + ('<div class="herochips">' + chips_html + '</div>' if chips_html else '')
        + '</div>'
    )

def _overview_section(overviews_html):
    if not overviews_html.strip():
        return ""
    return (
        '<section><div class="sec-num">한눈에</div>'
        '<h2><span class="bar"></span>한 줄로</h2>'
        '<div class="connect">' + overviews_html + '</div></section>'
    )

def _footer(school_name):
    return (
        '<footer>출처: 소재 배경은 공개 자료 교차 확인 · 삽화·데모는 학습용으로 직접 제작한 모식도입니다<br>'
        '<b>' + (school_name or "LevelMeUp") + '</b> 수업자료 · '
        '<span style="color:var(--teal-deep)">with Claude</span></footer>'
    )

def _doc(body_html, scripts_js, title, font_head=None):
    return (
        '<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<title>' + title + '</title>' + (font_head or _FONTS) +
        '<style>' + LEVELMEUP_CSS + '</style></head><body><div class="wrap">'
        + body_html +
        '</div><script>' + scripts_js + '</script></body></html>'
    )


def render_topic_background(passages_data, school_name="레벨미업학원"):
    """passages_data: [{label, topic:{...조각...}}, ...] (topic 없으면 플레이스홀더)
    한 번의 호출에 여러 지문을 받아 한 페이지로 조립합니다.
    """
    frags = [p["topic"] for p in passages_data if p.get("topic")]
    missing = [p.get("label", "?") for p in passages_data if not p.get("topic")]

    if not frags:
        body = (_hero(school_name, "", 0) +
                '<section class="in"><div class="card"><div class="cbody">'
                '아직 이 지문들의 <b>수업배경자료</b>가 없습니다. '
                + (("(미생성: " + ", ".join(missing) + ")") if missing else "") +
                '</div></div></section>' + _footer(school_name))
        return _doc(body, TOPIC_COMMON_JS, (school_name or "LevelMeUp") + " 수업배경자료",
                    _embed_fonts_css(body, TOPIC_COMMON_JS))

    chips = "".join('<a href="#%s">%s</a>' % (f.get("anchor", ""), f.get("chip", "")) for f in frags)
    sections = "\n".join(f.get("section_html", "") for f in frags)
    overviews = "\n".join(f.get("overview_html", "") for f in frags)
    scripts = "\n".join(f.get("script_js", "") for f in frags)

    note = ""
    if missing:
        note = ('<section class="in"><div class="card amber"><div class="cbody">'
                'ⓘ 다음 지문은 수업배경자료가 아직 없어 제외했어요: <b>'
                + ", ".join(missing) + '</b></div></div></section>')

    body = (_hero(school_name, chips, len(frags)) + sections + note
            + _overview_section(overviews) + _footer(school_name))
    scripts_all = TOPIC_COMMON_JS + "\n" + scripts
    title = (school_name or "LevelMeUp") + " 수업배경자료"
    # 폰트 임베드(가능하면). 실패 시 None → _doc 이 외부 <link> 로 폴백.
    font_head = _embed_fonts_css(body, scripts_all, chips, title)
    return _doc(body, scripts_all, title, font_head)
