# -*- coding: utf-8 -*-
"""_s142 관문 회귀 테스트 — 세 길이 전부 같은 목록을 보는지 확인한다.

이 테스트가 지키는 것: 목록에 단어를 넣었는데 어느 한 길만 막히는 상황.
그게 _s135('why?')·_s140('years' 'people') 의 원인이었다.
"""
import os, re, sys, ast
os.environ.setdefault("SUPABASE_URL", ""); os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variation import vocab_q3 as V

fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

# ── 1. 관문 자체 ─────────────────────────────────────────────
for w, tag in [("why", "기능어"), ("why?", "기능어(구두점 붙어도)"), ("The", "기능어"),
               ("however", "담화표지"), ("Similarly,", "담화표지"),
               ("years", "구체명사"), ("people", "구체명사"), ("audience", "구체명사"),
               ("of", "너무 짧음"), ("", "빈 값")]:
    check(V.blocked_reason(w) is not None, f"관문이 '{w}'({tag})를 안 막았다")

for w in ["significant", "modesty", "dissuade", "abandon", "rarely", "concern"]:
    check(V.blocked_reason(w) is None, f"관문이 정상 단어 '{w}'를 막았다")

# ── 2. 세 길이 전부 관문을 지나는가 ──────────────────────────
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "variation", "vocab_q3.py"), encoding="utf-8").read()
tree = ast.parse(src)
LISTS = {"_VOCAB_STOP", "_CONCRETE", "_DISCOURSE_MARKER"}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name not in ("blocked_reason", "is_discourse_marker"):
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & LISTS
        check(not used, f"{node.name}() 이 관문을 안 거치고 {sorted(used)} 를 직접 본다")

check("answer_pos_ok" not in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)},
      "죽은 검사대 answer_pos_ok 가 아직 있다")

# ── 3·4. 지문·5항목을 실제 모양으로 갖춘 fixture ─────────────
#   normalize / validate 는 '5개인가' '한 문장에 몰리지 않았나' 를 먼저 보므로
#   서로 다른 문장에서 5자리를 잡고, 그중 한 자리에만 금지어를 넣는다.
PA = "Sleep matters a lot. This finding seems significant to many."
PB = "The researchers found that people gain years of insight however clearly today."
PC = "It makes the outcome durable and strong. Everyone involved stays engaged and alert."
paras = [("A", PA), ("B", PB), ("C", PC)]
T = {0: PA.split(), 1: PB.split(), 2: PC.split()}

# (para, 단어) — 전부 서로 다른 문장
SLOTS = [(0, "matters"), (0, "significant"), (1, "insight"), (2, "durable"), (2, "engaged")]
# 선지·반의어는 서로 어근이 겹치면 안 된다 (_s139·_s141)
SHOWN = ["counts", "trivial", "blindness", "fragile", "bored"]
ANTON = ["ignores", "minor", "clarity", "sturdy", "eager"]

def five(bad_word=None):
    """5항목 fixture. bad_word 가 있으면 3번 자리(para B)를 그 단어로 바꾼다."""
    items = []
    for k, (pi, w) in enumerate(SLOTS):
        items.append({"n": k + 1, "para": pi, "idx": T[pi].index(w),
                      "original": w, "shown": SHOWN[k],
                      "antonym": ANTON[k], "is_answer": (k == 2)})
    if bad_word:
        items[2].update(para=1, idx=T[1].index(bad_word), original=bad_word)
    return items

BAD = ["people", "years", "however", "clearly"]

# ── 3. LLM 픽 경로(평소 쓰는 길) ─────────────────────────────
check(V.normalize_llm_vocab(five(), paras) is not None,
      "정상 fixture 가 LLM 픽 경로에서 거부됐다 — 테스트가 무의미해진다")
for bad in BAD:
    check(V.normalize_llm_vocab(five(bad), paras) is None,
          f"LLM 픽 경로가 '{bad}' 를 통과시켰다")

# ── 4. 마지막 확인(validate_vocab) ───────────────────────────
check(not V.validate_vocab(five(), paras, pid="테스트"),
      "정상 fixture 를 validate_vocab 이 거부했다 — 테스트가 무의미해진다")
for bad in BAD:
    errs = V.validate_vocab(five(bad), paras, pid="테스트")
    check(any(bad in e for e in errs), f"validate_vocab 이 '{bad}' 를 안 잡았다")

# ── 5. 코드 픽(폴백)도 막는가 ────────────────────────────────
cands = V.vocab_candidates(paras)
bare = {c["bare"] for c in cands}
for bad in ["people", "years", "however", "clearly"]:
    check(bad not in bare, f"코드 픽이 '{bad}' 를 후보로 냈다")


# ── 6. 부정관사 a/an 보정 (_s143) ────────────────────────────
#   실측: 'an enclosed space' 의 enclosed 를 sealed 로 바꾸자 'an sealed' 가 됐다.
apara = [("A", "An echo chamber refers to an enclosed space where sound stays.")]
aitem = {"n": 1, "para": 0, "idx": 6, "original": "enclosed", "shown": "sealed",
         "antonym": "open", "is_answer": True}
rendered = re.sub(r"<[^>]+>", "", V.apply_vocab_items(apara, [aitem])[0][1])
check("to a " in rendered and "an sealed" not in rendered,
      f"렌더가 관사를 안 고쳤다: {rendered}")

# 소리 규칙이라 코드가 다 못 맞힌다 — 확실한 것만 고치고 애매하면 그대로 둔다
for prev, shown, want in [("an", "sealed", "a"), ("a", "enclosed", "an"),
                          ("an", "hour", "an"), ("a", "university", "a"),
                          ("an", "honest", "an"), ("An", "sealed", "A"),
                          ("the", "sealed", "the"), ("a", "FBI", "a")]:
    got = V._fix_article_token(prev, shown)
    check(got == want, f"관사 보정 '{prev} {shown}' → '{got}' (기대 '{want}')")

# 렌더에만 있으면 옛 캐시를 못 잡는다 — validate_vocab 도 본다
bad5 = [dict(aitem, n=k + 1, is_answer=(k == 0)) for k in range(5)]
check(any("부정관사" in e for e in V.validate_vocab(bad5, apara, pid="T")),
      "validate_vocab 이 관사 불일치를 안 잡았다")


# ── 7. 구두점 (_s144) ────────────────────────────────────────
import variation.generator as G

# (1) 약어 마침표는 문장 경계가 아니다 — 'to 7 p.m.' 이 'to 7 p' 로 잘리면 안 된다
for text, want in [
    ("extend the library's operating hours to 7 p.m.", "extend the library's operating hours to 7 p.m."),
    ("extend the library's operating hours to 7 p.m. This change would", "extend the library's operating hours to 7 p.m."),
    ("earned his Ph.D. from Cornell University in 1961", "earned his Ph.D. from Cornell University in 1961"),
    ("studied at Cornell University in 1961. In 1967", "studied at Cornell University in 1961"),
    ("convince more readers for the whole story.", "convince more readers for the whole story"),
]:
    got = G._cut_before_punct(text, 4, sentence_only=True)
    check(got == want, f"빈칸 절단 '{text[:40]}…' → '{got}' (기대 '{want}')")

# (2) 밑줄 밖으로 구두점 — 원문 문장 끝 마침표가 사라지면 안 된다
_t = "he showed that whales use them to communicate."
_r = V.apply_vocab_items([("A", _t)], [{"n": 1, "para": 0, "idx": 7,
      "original": "communicate.", "shown": "interact", "antonym": "x", "is_answer": True}])[0][1]
check(_r.endswith("</u>."), f"문장 끝 마침표가 밑줄 밖에 안 붙었다: {_r[-40:]}")
check("<u class=\"vword\">interact</u>" in _r, f"밑줄 안에 구두점이 남았다: {_r[-40:]}")

# (3) 주격 전용 대명사로 끝나면 거부, you/it 은 허용
for ph, want_ok in [("got a little rush of excitement, knowing I", False),
                    ("said that after the meeting he", False),
                    ("the real product being sold is you", True),
                    ("makes the whole thing work for it", True)]:
    ok = G._clean_boundary_ok(ph, ph + " was there", strict=False)
    check(ok == want_ok, f"경계 판정 '{ph}' → {'허용' if ok else '거부'} (기대 {'허용' if want_ok else '거부'})")

print("\n".join("❌ " + f for f in fails) if fails else "✅ 전부 통과")
sys.exit(1 if fails else 0)
