"""
seosul/renderer.py
서술형 종합 세트(dict) → 2회독 동일 디자인 HTML (WeasyPrint로 PDF화).
- 변형문제 모듈의 var_style.css / logo2.png 를 재사용
- 지문은 passage_sentences의 {{A}}~{{E}} 자리표시를 답 길이에 비례한 빈칸으로 치환
- 학생용은 어법 오류를 일반체로(굵게 X), 교사용은 굵게 표시
"""
import os
import re
import base64

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def _style() -> str:
    with open(os.path.join(ASSET_DIR, "var_style.css"), encoding="utf-8") as f:
        return f.read()

def _logo_uri() -> str:
    p = os.path.join(ASSET_DIR, "logo2.png")
    if not os.path.exists(p):
        return ""
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

_EXTRA = """
.hdr-title{white-space:nowrap;}
/* 페이지 아래 여백 축소(로고 자리 확보용 22mm 불필요) → 로고가 진짜 모서리로 */
@page{ margin: 11mm 13mm 10mm 13mm; }
/* 로고: 인쇄 시 모든 페이지 우하단 모서리에 고정. Chrome 인쇄 기준 음수 오프셋으로 여백 안쪽까지 내림.
   화면('열기')에선 숨김. running element는 Chrome에서 본문 위로 튀므로 미사용. */
.footer-logo{ position:fixed !important; right:-7mm; bottom:-6mm; }
.footer-logo img{ height:15mm; }
@media screen{ .footer-logo{ display:none !important; } }
.qblock{margin-top:32px;}
.qhead{margin-bottom:6px;}
.ibl{display:inline-block;border-bottom:1.4px solid #111;text-align:center;vertical-align:bottom;
     color:var(--accent);font-weight:bold;font-size:8pt;line-height:1.25;margin:0 1px;}
.cond{font-size:8pt;color:#666;margin:2px 0 5px;}
.fix-row{display:flex;align-items:flex-end;margin-bottom:10px;gap:6px;}
.fix-row .fnum{width:22px;color:var(--accent);font-weight:bold;}
.fix-row .l1{width:150px;border-bottom:1px solid #999;height:17px;}
.fix-row .farrow{color:var(--accent);font-weight:bold;}
.fix-row .l2{flex:1;border-bottom:1px solid #999;height:17px;}
.wr-row{display:flex;align-items:flex-end;margin-bottom:11px;}
.wr-row .wtag{width:34px;color:var(--primary);font-weight:bold;font-size:9.5pt;}
.wr-row .wline{flex:1;border-bottom:1px solid #999;height:18px;}
.summary-box{border:1px solid var(--border);border-radius:5px;padding:8px 13px;line-height:1.95;margin:5px 0 7px;}
.se-line{display:flex;gap:14px;flex-wrap:wrap;font-size:9.3pt;margin-top:5px;}
.se-line .sb{display:inline-block;width:96px;border-bottom:1px solid #999;}
.wb{display:inline-block;min-width:64px;border-bottom:1.2px solid #111;text-align:center;margin:0 1px;}
.sc-lab{color:var(--primary);font-weight:bold;margin:0 2px;}
.sc-blank{display:inline-block;border-bottom:1.4px solid #111;height:14px;vertical-align:middle;margin:0 3px;}
.sd-ans{display:flex;align-items:baseline;gap:8px;margin-bottom:4px;flex-wrap:wrap;}
.sd-ans .sd-n{color:var(--accent);font-weight:bold;}
.sd-ans .sd-why{color:#333;font-size:9pt;flex:1;min-width:240px;}
/* 정답 컴팩트: (A)(B) 한 줄 인라인, 빈 띠 제거 */
.ans-line{font-size:9pt;line-height:1.75;margin:2px 0;}
.ans-line .lab{color:var(--primary);font-weight:bold;margin:0 3px 0 0;}
.ans-line .av{color:var(--accent);font-weight:bold;padding:0 6px;background:var(--bg-orange);border-radius:3px;margin-right:14px;}
.ans-line .nt{color:#666;font-size:7.6pt;margin-right:14px;}
.ans-subhdr{font-size:9pt;font-weight:bold;color:var(--primary);margin:9px 0 3px;}
.ans-subhdr:first-of-type{margin-top:2px;}
"""

ANS_HDR = '<div class="ans-hdr"><span class="hdr-title">정답 및 해설</span></div>'

# 문항번호 색 (영작/요약=주황 write, 어법=남색 기본, 어휘=보라 core)
_QNUM_CLS = {"SA": "write", "SC": "write", "SD": "", "SE": "core"}


def _blank_width(answer: str) -> int:
    """정답 길이에 비례한 빈칸 폭(px)."""
    n = len(answer)
    return max(58, min(190, int(28 + n * 4.2)))


def _build_passage(s: dict, answers_by_label: dict, teacher: bool) -> str:
    """passage_sentences의 {{A}}~ 자리표시를 빈칸으로 치환. 어법 오류는 학생용=일반체."""
    text = " ".join(s["passage_sentences"])
    # 빈칸 치환
    def repl(m):
        lab = m.group(1)
        w = _blank_width(answers_by_label.get(lab, "______"))
        return f'<span class="ibl" style="width:{w}px">({lab})</span>'
    text = re.sub(r"\{\{\s*\(?([A-Za-z가-힣]+)\)?\s*\}\}", repl, text)
    # 어법 오류 강조 (교사용만)
    if teacher:
        for w in s.get("_error_words", []):
            text = re.sub(rf"(?<![\w]){re.escape(w)}(?![\w])", f"<b>{w}</b>", text, count=1)
    return f'<div class="psg-box">{text}</div>'


def _split_bogi(bogi):
    """보기를 단어 단위로 분리. 관사(a/an/the)는 바로 뒤 단어와 붙여 한 덩어리로 유지."""
    words = []
    for chunk in bogi:
        toks = str(chunk).split()
        i = 0
        while i < len(toks):
            if toks[i].lower() in ("a", "an", "the") and i + 1 < len(toks):
                words.append(toks[i] + " " + toks[i + 1]); i += 2
            else:
                words.append(toks[i]); i += 1
    return words


def _render_problem(s: dict, teacher: bool, school_name: str) -> str:
    """문제부 HTML 조각 (헤더 + 지문 + 문항)."""
    ref = s.get("passage_ref", {})
    ref_label = f'{ref.get("book","")} · {ref.get("unit","")} {ref.get("pid","")}'.strip(" ·")

    ans_by_label = {}
    for it in s["items"]:
        if it["type"] in ("SA", "SE", "SC"):
            for k, v in it.get("answers", {}).items():
                ans_by_label[k] = v
            for bl in it.get("blanks", []) if isinstance(it.get("blanks"), list) else []:
                ans_by_label[bl["label"]] = bl["answer"]

    body = [f'''<div class="hdr">
  <span class="hdr-title">{school_name} &nbsp;&middot;&nbsp; 서술형 종합문제 <span class="var-badge">단일 지문</span></span>
  <span class="hdr-right">{ref_label}</span>
</div>''']
    body.append(_build_passage(s, ans_by_label, teacher))

    qno = 0
    for it in s["items"]:
        qno += 1
        cls = _QNUM_CLS.get(it["type"], "")
        ins = it["instruction"]
        head = (f'<div class="qblock"><div class="qhead">'
                f'<span class="q-num {cls}">{qno}</span> {ins}</div>')
        if it["type"] == "SA":
            cond = '<div class="cond">※ 어형 변형 및 단어 중복 가능</div>' if it.get("allow_inflect") else ""
            bogi = ' / '.join(it["bogi"])
            rows = "".join(f'<div class="wr-row"><span class="wtag">({k})</span><span class="wline"></span></div>'
                           for k in it["answers"])
            body.append(head + cond +
                        f'<div class="bogi"><span class="bogi-label">보기</span> {bogi}</div>' + rows + '</div>')
        elif it["type"] == "SC":
            bogi = ' / '.join(_split_bogi(it["bogi"]))
            summ = it["summary"]
            for k, v in it["answers"].items():
                w = min(340, max(130, len(v) * 7))
                blank = (f'<span class="sc-lab">({k})</span>'
                         f'<span class="sc-blank" style="width:{w}px"></span>')
                sentinel = f"\x00{k}\x00"
                # {{A}} / (A) / (A)____ 를 모두 표식으로 — 모델이 라벨과 placeholder를 같이 쓴 경우 대비
                summ = re.sub(r'\{\{\s*%s\s*\}\}' % re.escape(k), sentinel, summ)
                summ = re.sub(r'\(\s*%s\s*\)\s*_*' % re.escape(k), sentinel, summ)
                # 연달아 붙은 표식(예: (A){{A}})은 하나의 빈칸으로 합침
                summ = re.sub(r'(?:%s\s*)+' % re.escape(sentinel), sentinel, summ)
                summ = summ.replace(sentinel, blank)
            rows = "".join(f'<div class="wr-row"><span class="wtag">({k})</span><span class="wline"></span></div>'
                           for k in it["answers"])
            body.append(head + f'<div class="summary-box">{summ}</div>' +
                        f'<div class="bogi"><span class="bogi-label">보기</span> {bogi}</div>' + rows + '</div>')
        elif it["type"] == "SD":
            rows = "".join(f'<div class="fix-row"><span class="fnum">{chr(9311+i)}</span>'
                           f'<span class="l1"></span><span class="farrow">→</span><span class="l2"></span></div>'
                           for i in range(1, len(it["errors"]) + 1))
            body.append(head + rows + '</div>')
        elif it["type"] == "SE":
            bogi = ' / '.join(it["bogi"])
            slots = "".join(f'<span>({b["label"]}) <span class="sb"></span></span>' for b in it["blanks"])
            body.append(head + f'<div class="bogi"><span class="bogi-label">보기</span> {bogi}</div>' +
                        f'<div class="se-line">{slots}</div></div>')
    return "".join(body)


def _render_answer(s: dict) -> str:
    """정답부 HTML 조각. '정답 및 해설' 큰 헤더는 조립부에서 1회만 붙이고,
    여기선 지문별 작은 라벨 + 해설 블록만. 배점 표기 없음."""
    ref = s.get("passage_ref", {})
    ref_label = f'{ref.get("book","")} · {ref.get("unit","")} {ref.get("pid","")}'.strip(" ·")
    body = [f'<div class="ans-subhdr">{ref_label}</div>']
    qno = 0
    for it in s["items"]:
        qno += 1
        title_map = {"SA": "본문 빈칸 영작", "SC": "요약문 빈칸",
                     "SD": "어법 틀린 곳 고치기", "SE": "어휘 품사 변형"}
        blk = [f'<div class="ans-block"><div class="ans-block-title">'
               f'{qno}. {title_map.get(it["type"])} ({it["type"]})</div>']
        if it["type"] in ("SA", "SC"):
            cells = "".join(f'<span class="lab">({k})</span><span class="av">{v}</span>'
                            for k, v in it["answers"].items())
            blk.append(f'<div class="ans-line">{cells}</div>')
            if it.get("structure"):
                blk.append(f'<div class="grammar-note"><b>구조</b> {it["structure"]}</div>')
        elif it["type"] == "SD":
            for i, e in enumerate(it["errors"], 1):
                why = e.get("why", "") or f'{e.get("wrong","")} → {e.get("right","")}'
                blk.append(f'<div class="sd-ans"><span class="sd-n">{chr(9311+i)}</span>'
                           f'<span class="sd-why">{why}</span></div>')
        elif it["type"] == "SE":
            cells = "".join(f'<span class="lab">({b["label"]})</span>'
                            f'<span class="av">{b["answer"]}</span>'
                            f'<span class="nt">{b.get("note","")}</span>'
                            for b in it["blanks"])
            blk.append(f'<div class="ans-line">{cells}</div>')
        if it.get("why"):
            blk.append(f'<div class="expl-box"><span class="expl-label">근거</span>{it["why"]}</div>')
        blk.append('</div>')
        body.append("".join(blk))
    return "".join(body)


def render_fragments(s: dict, teacher: bool = False, school_name: str = "레벨미업학원"):
    """(문제부, 정답부) 조각 반환. api에서 모드별로 재조합."""
    return _render_problem(s, teacher, school_name), _render_answer(s)


def wrap_document(body_inner: str, school_name: str = "레벨미업학원") -> str:
    """조각들을 완성 HTML로 감싼다. 로고는 fixed라 모든 페이지 우하단에 1개만 둔다."""
    logo = f'<div class="footer-logo"><img src="{_logo_uri()}" alt="로고"></div>'
    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
            f'<title>서술형 종합문제 - {school_name}</title>'
            f'<style>{_style()}{_EXTRA}</style></head><body>{logo}{body_inner}</body></html>')


def render_seosul_set(s: dict, teacher: bool = False,
                      school_name: str = "레벨미업학원") -> str:
    """단일 세트: 문제 → 정답(페이지 분리) 한 문서."""
    prob, ans = render_fragments(s, teacher, school_name)
    body = prob + '<div class="ans-start"></div>' + ANS_HDR + ans + '<div class="ans-end"></div>'
    return wrap_document(body, school_name)


def render_pdf(s: dict, out_path: str, teacher: bool = False) -> str:
    from weasyprint import HTML
    html = render_seosul_set(s, teacher=teacher)
    HTML(string=html).write_pdf(out_path)
    return out_path
