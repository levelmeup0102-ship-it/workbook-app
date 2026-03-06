#!/usr/bin/env python3
"""
ì˜ì–´ ì›Œí¬ë¶ ìžë™ ìƒì„± íŒŒì´í”„ë¼ì¸
- ì§€ë¬¸ txt ìž…ë ¥ â†’ Claude APIë¡œ ë ˆë²¨ë³„ ì½˜í…ì¸  ìƒì„± â†’ í…œí”Œë¦¿ â†’ PDF
- ë‹¨ê³„ë³„ JSON ì €ìž¥ (ì‹¤íŒ¨ ì‹œ í•´ë‹¹ ë‹¨ê³„ë§Œ ìž¬ì‹œë„)
- 20ê°œ ì§€ë¬¸ ìˆœì°¨ ì²˜ë¦¬ (ë™ì‹œ í˜¸ì¶œ ì—†ìŒ)
"""
import json, os, sys, time, random, re
from pathlib import Path
from anthropic import Anthropic

# ============================================================
# ì„¤ì •
# ============================================================
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
TEMPLATE_DIR = Path(__file__).parent
DATA_DIR = TEMPLATE_DIR / "data"
OUTPUT_DIR = TEMPLATE_DIR / "output"

client = None  # lazy init

def get_client():
    global client
    if client is None:
        if not API_KEY:
            print("âŒ ANTHROPIC_API_KEY í™˜ê²½ë³€ìˆ˜ë¥¼ ì„¤ì •í•˜ì„¸ìš”")
            sys.exit(1)
        client = Anthropic(api_key=API_KEY)
    return client

# ============================================================
# Claude API í˜¸ì¶œ (ìž¬ì‹œë„ í¬í•¨)
# ============================================================
def call_claude(system_prompt: str, user_prompt: str, max_retries=2, max_tokens=4096) -> str:
    """Claude API í˜¸ì¶œ + ìž¬ì‹œë„"""
    c = get_client()
    for attempt in range(max_retries + 1):
        try:
            resp = c.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            text = resp.content[0].text.strip()
            return text
        except Exception as e:
            print(f"  âš ï¸ ì‹œë„ {attempt+1} ì‹¤íŒ¨: {e}")
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
            else:
                raise

def call_claude_json(system_prompt: str, user_prompt: str, max_retries=2, max_tokens=4096) -> dict:
    """Claude API í˜¸ì¶œ â†’ JSON íŒŒì‹± (ìž¬ì‹œë„ í¬í•¨)"""
    text = call_claude(system_prompt, user_prompt, max_retries, max_tokens)
    # JSON ë¸”ë¡ ì¶”ì¶œ
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # í•œë²ˆ ë” ì‹œë„ - JSON ë¶€ë¶„ë§Œ ì¶”ì¶œ
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        raise ValueError(f"JSON íŒŒì‹± ì‹¤íŒ¨:\n{text[:500]}")

# ============================================================
# ë‹¨ê³„ë³„ ì €ìž¥/ë¡œë“œ
# ============================================================
def save_step(passage_dir: Path, step_name: str, data: dict):
    passage_dir.mkdir(parents=True, exist_ok=True)
    path = passage_dir / f"{step_name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ðŸ’¾ ì €ìž¥: {step_name}.json")

def load_step(passage_dir: Path, step_name: str) -> dict | None:
    path = passage_dir / f"{step_name}.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# ============================================================
# SYSTEM PROMPT (ê³µí†µ)
# ============================================================
SYS_JSON = """You are an English exam content generator for Korean high school students.
Return ONLY valid JSON. No markdown fences. No explanations. No preamble.
All Korean text must use proper Korean. All English must be grammatically correct."""

SYS_JSON_KR = """ë‹¹ì‹ ì€ í•œêµ­ ê³ ë“±í•™ìƒì„ ìœ„í•œ ì˜ì–´ ì‹œí—˜ ì½˜í…ì¸  ìƒì„±ê¸°ìž…ë‹ˆë‹¤.
ë°˜ë“œì‹œ ìœ íš¨í•œ JSONë§Œ ë°˜í™˜í•˜ì„¸ìš”. ë§ˆí¬ë‹¤ìš´, ì„¤ëª…, ì„œë¬¸ ì—†ì´ JSONë§Œ ì¶œë ¥í•˜ì„¸ìš”.
í•œêµ­ì–´ëŠ” ìžì—°ìŠ¤ëŸ½ê²Œ, ì˜ì–´ëŠ” ë¬¸ë²•ì ìœ¼ë¡œ ì •í™•í•˜ê²Œ ìž‘ì„±í•˜ì„¸ìš”."""

# ============================================================
# STEP 1: ê¸°ë³¸ ë¶„ì„ (ì–´íœ˜ + ë²ˆì—­ + í•µì‹¬ë¬¸ìž¥)
# ============================================================
def step1_basic_analysis(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step1_basic")
    if cached:
        print("  âœ… step1 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step1: ê¸°ë³¸ ë¶„ì„ (ì–´íœ˜/ë²ˆì—­/í•µì‹¬ë¬¸ìž¥)...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ì„ ë¶„ì„í•˜ì—¬ JSONì„ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[ìƒì„± í•­ëª©]
1. vocab: í•µì‹¬ ì–´íœ˜ 14ê°œ (ê°ê° word, meaning(í•œêµ­ì–´), synonyms(ì˜ì–´ ë™ì˜ì–´ 4ê°œ ì‰¼í‘œêµ¬ë¶„))
2. translation: ì§€ë¬¸ ì „ì²´ì˜ ìžì—°ìŠ¤ëŸ¬ìš´ í•œêµ­ì–´ ë²ˆì—­
3. sentences: ì§€ë¬¸ì˜ ëª¨ë“  ë¬¸ìž¥ì„ ê°œë³„ ë°°ì—´ë¡œ ë¶„ë¦¬
4. key_sentences: ì‹œí—˜ ì¶œì œ ê°€ëŠ¥ì„±ì´ ë†’ì€ í•µì‹¬ ë¬¸ìž¥ 8ê°œ (ì›ë¬¸ ê·¸ëŒ€ë¡œ)
5. test_a: vocabì—ì„œ ëœ» ì“°ê¸° í…ŒìŠ¤íŠ¸ìš© 5ê°œ ë‹¨ì–´ (ì˜ì–´)
6. test_b: vocabì—ì„œ ë™ì˜ì–´ í…ŒìŠ¤íŠ¸ìš© 5ê°œ ë‹¨ì–´ (test_aì™€ ê²¹ì¹˜ì§€ ì•Šê²Œ, ì˜ì–´)
7. test_c: vocabì—ì„œ ì² ìž í…ŒìŠ¤íŠ¸ìš© 5ê°œ (í•œêµ­ì–´ ëœ»)

JSON í˜•ì‹:
{{
  "vocab": [{{"word":"...", "meaning":"...", "synonyms":"..."}}],
  "translation": "...",
  "sentences": ["...", "..."],
  "key_sentences": ["...", "..."],
  "test_a": ["...", "..."],
  "test_b": ["...", "..."],
  "test_c": ["...", "..."]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=4096)
    save_step(passage_dir, "step1_basic", data)
    return data

# ============================================================
# STEP 2: Lv.5 ìˆœì„œ/ì‚½ìž…
# ============================================================
def step2_order(passage: str, sentences: list, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step2_order")
    if cached:
        print("  âœ… step2 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step2: Lv.5 ìˆœì„œ/ì‚½ìž… ìƒì„±...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ìœ¼ë¡œ ìˆœì„œ ë°°ì—´ + ë¬¸ìž¥ ì‚½ìž… ë¬¸ì œë¥¼ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[ê°œë³„ ë¬¸ìž¥]
{json.dumps(sentences, ensure_ascii=False)}

[ìƒì„± í•­ëª©]
1. order_intro: ì œì‹œë¬¸ (ì²« 1~2ë¬¸ìž¥)
2. order_paragraphs: (A)(B)(C) 3ê°œ ë‹¨ë½ (ê°ê° labelê³¼ text). ì •ë‹µ ìˆœì„œëŠ” ì›ë¬¸ ìˆœì„œëŒ€ë¡œ.
   - ëª¨ë“  ë¬¸ìž¥ì´ ë¹ ì§ì—†ì´ í¬í•¨ë˜ì–´ì•¼ í•¨
3. order_choices: 5ì§€ì„ ë‹¤ (í˜•ì‹: "â‘  (A)-(C)-(B)" ë“±). ì •ë‹µ 1ê°œ í¬í•¨.
4. order_answer: ì •ë‹µ ë²ˆí˜¸ (ì˜ˆ: "â‘£ (C)-(A)-(B)")
5. insert_sentence: ì‚½ìž…í•  ë¬¸ìž¥ 1ê°œ (ì•žë’¤ ë¬¸ë§¥ ë‹¨ì„œê°€ ëª…í™•í•œ ê²ƒ)
6. insert_passage: ì‚½ìž… ë¬¸ìž¥ì„ ëº€ ë‚˜ë¨¸ì§€ ì§€ë¬¸ì— ( â‘  )~( â‘¤ ) ìœ„ì¹˜ í‘œì‹œ
7. insert_answer: ì‚½ìž… ì •ë‹µ ë²ˆí˜¸
8. full_order_blocks: ì „ì²´ ë¬¸ìž¥ì„ (A)~ëê¹Œì§€ ê°œë³„ ë¸”ë¡ìœ¼ë¡œ ë¶„í•  (ê°ê° label, text)
9. full_order_answer: ì •ë‹µ ìˆœì„œ (ì˜ˆ: "(C)â†’(G)â†’(D)â†’...")

JSON í˜•ì‹:
{{
  "order_intro": "...",
  "order_paragraphs": [{{"label":"A","text":"..."}}, ...],
  "order_choices": ["â‘  ...", "â‘¡ ...", ...],
  "order_answer": "...",
  "insert_sentence": "...",
  "insert_passage": "...",
  "insert_answer": "...",
  "full_order_blocks": [{{"label":"A","text":"..."}}, ...],
  "full_order_answer": "..."
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=4096)
    # ë³€í™˜: order_paragraphsë¥¼ [label, text] í˜•íƒœë¡œ
    if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
        data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]
    if data.get("full_order_blocks") and isinstance(data["full_order_blocks"][0], dict):
        data["full_order_blocks"] = [[b["label"], b["text"]] for b in data["full_order_blocks"]]

    # ðŸ”’ ê²€ì¦: ì „ì²´ë°°ì—´ ë¸”ë¡ ìˆ˜ vs ì›ë¬¸ ë¬¸ìž¥ ìˆ˜
    block_count = len(data.get("full_order_blocks", []))
    sentence_count = len(sentences)
    if block_count != sentence_count:
        print(f"  âš ï¸ ë¬¸ìž¥ ìˆ˜ ë¶ˆì¼ì¹˜! ì›ë¬¸ {sentence_count}ê°œ vs ìƒì„± {block_count}ê°œ â†’ ìž¬ìƒì„±...")
        # ìºì‹œ ì‚­ì œ í›„ ìž¬ì‹œë„ (1íšŒ)
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
            print(f"  âš ï¸ ìž¬ì‹œë„ í›„ì—ë„ ë¶ˆì¼ì¹˜ ({block_count2} vs {sentence_count}) â†’ ì›ë¬¸ ë¬¸ìž¥ìœ¼ë¡œ ëŒ€ì²´")
            data["full_order_blocks"] = [[chr(65+i), s] for i, s in enumerate(sentences)]

    save_step(passage_dir, "step2_order", data)
    return data

# ============================================================
# STEP 3: Lv.6 ë¹ˆì¹¸ ì¶”ë¡ 
# ============================================================
def step3_blank(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step3_blank")
    if cached:
        print("  âœ… step3 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step3: Lv.6 ë¹ˆì¹¸ ì¶”ë¡  ìƒì„±...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ìœ¼ë¡œ ë¹ˆì¹¸ ì¶”ë¡  ë¬¸ì œë¥¼ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[ê·œì¹™]
- ì£¼ì œë¬¸(ê²°ë¡ ë¬¸)ì˜ í•µì‹¬ ë¶€ë¶„ì„ ë¹ˆì¹¸ìœ¼ë¡œ ë§Œë“¤ê¸°
- ë¹ˆì¹¸ì€ 15ë‹¨ì–´ ì´ë‚´ë¡œ (ë„ˆë¬´ ê¸´ ë¹ˆì¹¸ ê¸ˆì§€)
- ë¹ˆì¹¸ ë¬¸ìž¥ ì™¸ì˜ ë‹¤ë¥¸ ë¬¸ìž¥ì€ ì›ë¬¸ ê·¸ëŒ€ë¡œ ìœ ì§€ (ìƒëžµ/ì¶•ì•½/ë³€í˜• ì ˆëŒ€ ê¸ˆì§€)
- ë¹ˆì¹¸ì„ ì œì™¸í•œ ë‚˜ë¨¸ì§€ ë¬¸ìž¥ ë¶€ë¶„ë„ ì ˆëŒ€ ë³€í˜•í•˜ì§€ ë§ ê²ƒ
- ì„ ì§€ 12ê°œ: ì •ë‹µ 6~7ê°œ + ì˜¤ë‹µ 5~6ê°œ
- ì •ë‹µ: ì›ë¬¸ í•µì‹¬ í‘œí˜„ì„ ë™ì˜ì–´/ë¹„ìœ ì  í‘œí˜„ìœ¼ë¡œ ë³€í˜•
- ì˜¤ë‹µ: ì§€ë¬¸ ë‚´ìš© ì™œê³¡, ë°˜ëŒ€ ì˜ë¯¸, ë¯¸ì–¸ê¸‰ ë‚´ìš©
- ê° ì„ ì§€ëŠ" 15ë‹¨ì–´ ì´ë‚´ë¡œ ê°„ê²°í•˜ê²Œ
- ★중복 금지★ 선지들 간에 의미가 겹치면 안 됨 (동의어/유사표현으로 같은 뜻의 선지 2개 이상 금지)
- 정답 선지들도 각각 서로 다른 측면/관점에서 빈칸을 설명해야 함

[JSON í˜•ì‹]
{{
  "blank_passage": "ë¹ˆì¹¸ì´ í¬í•¨ëœ ì „ì²´ ì§€ë¬¸ (ë¹ˆì¹¸ì€ ____ë¡œ í‘œì‹œ)",
  "blank_answer_korean": "ë¹ˆì¹¸ ì •ë‹µ ë‚´ìš© í•œêµ­ì–´",
  "blank_options": ["â‘  ...", "â‘¡ ...", ... "â‘« ..."],
  "blank_correct": ["â‘¡", "â‘¢", "â‘¤", ...],
  "blank_wrong": ["â‘ ", "â‘£", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)

    # === 후처리: 선지 순서 랜덤 셔플 (정답 위치 고정 방지) ===
    options = data.get("blank_options", [])
    correct_set = set(data.get("blank_correct", []))
    wrong_set = set(data.get("blank_wrong", []))
    if options and len(options) >= 2:
        CIRCLE_NUMS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"]
        # 선지 텍스트만 추출 (번호 제거)
        texts = []
        old_correct_texts = []
        old_wrong_texts = []
        for opt in options:
            text = re.sub(r'^[①-⑫]\s*', '', opt).strip()
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
        random.shuffle(texts)
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
        print(f"  🔀 빈칸 선지 셔플 완료: 정답 위치 {new_correct}")

    save_step(passage_dir, "step3_blank", data)
    return data

# ============================================================
# STEP 4: Lv.7 ì£¼ì œ ì°¾ê¸°
# ============================================================
def step4_topic(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step4_topic")
    if cached:
        print("  âœ… step4 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step4: Lv.7 ì£¼ì œ ì°¾ê¸° ìƒì„±...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ìœ¼ë¡œ ì£¼ì œ ì°¾ê¸° ë¬¸ì œë¥¼ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[ê·œì¹™]
- ì§€ë¬¸ì€ ì›ë¬¸ ê·¸ëŒ€ë¡œ (ìƒëžµ/ë³€í˜• ê¸ˆì§€)
- ì„ ì§€ 12ê°œ: ì •ë‹µ 5ê°œ + ì˜¤ë‹µ 7ê°œ
- ì •ë‹µ: ì£¼ì œë¬¸ í‚¤ì›Œë“œë¥¼ ë™ì˜ì–´ë¡œ ì¹˜í™˜í•œ í‘œí˜„
- ì˜¤ë‹µ: ì§€ë¬¸ ë¯¸ì–¸ê¸‰, ë¶€ë¶„ì  ë‚´ìš©, ì™œê³¡
- ì¶”ë¡ ì  ì‚¬ê³  ê¸ˆì§€: ê¸€ì—ì„œ ì§ì ‘ ì–¸ê¸‰ëœ ë‚´ìš©ë§Œ ì •ë‹µ

[JSON í˜•ì‹]
{{
  "topic_passage": "ì›ë¬¸ ì „ë¬¸ (ê·¸ëŒ€ë¡œ)",
  "topic_options": ["â‘  ...", "â‘¡ ...", ... "â‘« ..."],
  "topic_correct": ["â‘¡", "â‘£", ...],
  "topic_wrong": ["â‘ ", "â‘¢", ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=3000)
    save_step(passage_dir, "step4_topic", data)
    return data

# ============================================================
# STEP 5: Lv.8 ì–´ë²•
# ============================================================
def step5_grammar(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step5_grammar")
    if cached:
        print("  âœ… step5 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step5: Lv.8 ì–´ë²• ìƒì„±...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ìœ¼ë¡œ ì–´ë²• ë¬¸ì œ 2ì¢…ë¥˜ë¥¼ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[어법 괄호형 Lv.8-1 규칙]
- 원문의 모든 문장을 빠짐없이 포함 (출제 대상 아닌 문장도 원문 그대로)
- ★절대금지★ 원문에 없는 문장을 추가하거나, 원문 문장을 축약/변형하지 말 것
- 원문 문장 수와 생성된 지문의 문장 수가 반드시 동일해야 함
- 13개 괄호: (N)[정답 / 오답] 형태
- 출제 포인트: 시제, 대명사, 동명사, 완료시제, to부정사, 형용사/부사, 관계대명사, 분사, 사역동사 등

[★출제 금지 어법 - 아래 패턴은 절대 출제하지 말 것 (둘 다 정답이므로)★]
- start/begin/continue/love/hate/like/prefer 뒤 to V vs V-ing 선택 금지
- try/remember/forget/stop/regret/need/want 뒤 to V vs V-ing 선택 금지
- help 뒤 to V vs V원형 선택 금지
- 목적격 관계대명사 생략 여부 출제 금지
- 위 동사 포함 문장은 다른 어법 포인트(시제, 수일치, 분사 등)로 출제하거나 출제 제외


[ì–´ë²• ì„œìˆ í˜• Lv.8-2 ê·œì¹™]
- ì›ë¬¸ì˜ ëª¨ë“  ë¬¸ìž¥ì„ ë¹ ì§ì—†ì´ í¬í•¨
- ì›ë¬¸ ë¬¸ìž¥ì„ ì¶•ì•½í•˜ê±°ë‚˜ ì¶”ê°€í•˜ì§€ ë§ ê²ƒ (ì›ë¬¸ ê·¸ëŒ€ë¡œ + ì˜¤ë¥˜ë§Œ ì‚½ìž…)
- 8ê°œ ë¬¸ë²• ì˜¤ë¥˜ ì‚½ìž… (ë°‘ì¤„ ì—†ì´)
- í•œ ë¬¸ìž¥ì— ìµœëŒ€ 1ê°œ ì˜¤ë¥˜
- ì˜¤ë¥˜ ê°„ê²©ì„ ê³ ë¥´ê²Œ ë¶„í¬

[JSON í˜•ì‹]
{{
  "grammar_bracket_passage": "ê´„í˜¸ í¬í•¨ ì „ì²´ ì§€ë¬¸",
  "grammar_bracket_count": 13,
  "grammar_bracket_answers": [{{"num":1, "answer":"go", "wrong":"will go", "reason":"if ì¡°ê±´ì ˆ í˜„ìž¬ì‹œì œ"}}, ...],
  "grammar_error_passage": "ì˜¤ë¥˜ í¬í•¨ ì „ì²´ ì§€ë¬¸",
  "grammar_error_count": 8,
  "grammar_error_answers": [{{"num":1, "original":"watch", "error":"watching", "reason":"tend to + ë™ì‚¬ì›í˜•"}}, ...]
}}"""

    data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)

    # === 후처리 1: 원문에 없는 문장 추가 감지 ===
    gen_passage = data.get("grammar_bracket_passage", "")
    # 괄호 제거 후 문장 수 비교
    clean_gen = re.sub(r'\(\d+\)\[.*?\]', '', gen_passage)
    clean_gen = re.sub(r'\(\d+\)\[([^/\]]+)\s*/\s*[^\]]+\]', r'\1', gen_passage)
    orig_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage) if s.strip()]
    gen_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_gen) if s.strip()]
    if len(gen_sentences) > len(orig_sentences) + 1:
        print(f"  ⚠️ 8-1 문장 추가 감지! 원문 {len(orig_sentences)}개 vs 생성 {len(gen_sentences)}개 → 재생성...")
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)

    # === 후처리 2: 둘 다 정답인 동사 감지 ===
    DUAL_VERBS = ['start', 'begin', 'continue', 'love', 'hate', 'like', 'prefer',
                  'try', 'remember', 'forget', 'stop', 'regret', 'need', 'want', 'help']
    bracket_answers = data.get("grammar_bracket_answers", [])
    flagged = []
    for a in bracket_answers:
        if isinstance(a, dict):
            ans = a.get("answer", "").lower()
            wrong = a.get("wrong", "").lower()
            # to V vs V-ing 패턴 감지
            if (ans.endswith("ing") and wrong.startswith("to ")) or \
               (wrong.endswith("ing") and ans.startswith("to ")) or \
               (ans.startswith("to ") and "ing" in wrong) or \
               (wrong.startswith("to ") and "ing" in ans):
                flagged.append(f"({a.get('num','')}) {ans}/{wrong}")
    if flagged:
        print(f"  ⚠️ 둘 다 정답 가능 감지: {flagged} → 재생성 권장")
        # 재생성 시도
        data = call_claude_json(SYS_JSON, prompt, max_tokens=4000)

    save_step(passage_dir, "step5_grammar", data)
    return data

# ============================================================
# STEP 6: Lv.9 ì–´íœ˜ì‹¬í™” + ë‚´ìš©ì¼ì¹˜
# ============================================================
def step6_vocab_content(passage: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step6_vocab_content")
    if cached:
        print("  âœ… step6 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step6: Lv.9 ì–´íœ˜ì‹¬í™” + ë‚´ìš©ì¼ì¹˜ ìƒì„±...")
    prompt = f"""ë‹¤ìŒ ì˜ì–´ ì§€ë¬¸ìœ¼ë¡œ ì–´íœ˜ ì‹¬í™” + ë‚´ìš© ì¼ì¹˜ ë¬¸ì œë¥¼ ìƒì„±í•˜ì„¸ìš”.

[ì§€ë¬¸]
{passage}

[Lv.9-1 Part A ê·œì¹™]
- ì›ë¬¸ì˜ ëª¨ë“  ë¬¸ìž¥ì„ ë¹ ì§ì—†ì´ í¬í•¨
- 9ê°œ ê´„í˜¸: (N)[ì •ë‹µ / í˜¼ë™ì–´] (ì² ìž/ë°œìŒ ìœ ì‚¬í•œ í˜¼ë™ì–´)

[Lv.9-1 Part B ê·œì¹™]
- 8ê°œ ë‹¨ì–´, ê° 5ê°œ ì„ íƒì§€ (ì •ë‹µ 2~3ê°œ)
- ì˜¤ë‹µ: ì² ìž/ë°œìŒ ìœ ì‚¬í•˜ì§€ë§Œ ì˜ë¯¸ ë‹¤ë¥¸ ë‹¨ì–´

[Lv.9-2 ê·œì¹™]
- í•œêµ­ì–´ ì„ ì§€ 10ê°œ (ì •ë‹µ 6ê°œ, ì˜¤ë‹µ 4ê°œ - í‚¤ì›Œë“œ 1ê°œ ë³€í˜•)
- ì˜ì–´ ì„ ì§€ 10ê°œ (ì •ë‹µ 5ê°œ, ì˜¤ë‹µ 5ê°œ - í‚¤ì›Œë“œ 1ê°œ ë³€í˜•)

[JSON í˜•ì‹]
{{
  "vocab_advanced_passage": "ê´„í˜¸ í¬í•¨ ì§€ë¬¸",
  "vocab_parta_answers": [{{"num":1, "answer":"tend", "wrong":"intend"}}, ...],
  "vocab_partb": [{{"word":"tend to", "choices":"be inclined to / intend to / be prone to / pretend to / be apt to"}}, ...],
  "vocab_partb_answers": [{{"num":1, "correct":["be inclined to", "be prone to", "be apt to"]}}, ...],
  "content_match_kr": ["â‘  í‰ì†Œ ê·¹ìž¥ì—ì„œ í˜¼ìž ì˜í™”ë¥¼ ë³¸ë‹¤.", ...],
  "content_match_kr_answer": ["â‘¡", "â‘¢", ...],
  "content_match_en": ["â‘  The writer normally watches movies alone.", ...],
  "content_match_en_answer": ["â‘¡", "â‘£", ...]
}}"""

    data = call_claude_json(SYS_JSON_KR, prompt, max_tokens=4000)
    save_step(passage_dir, "step6_vocab_content", data)
    return data

# ============================================================
# STEP 7: Lv.10 ì˜ìž‘ (API ë¶ˆí•„ìš” - í”„ë¡œê·¸ëž˜ë°ìœ¼ë¡œ ì²˜ë¦¬)
# ============================================================
def step7_writing(sentences: list, translation: str, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step7_writing")
    if cached:
        print("  âœ… step7 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step7: Lv.10 ì˜ìž‘ ìƒì„± (ë¡œì»¬ ì²˜ë¦¬)...")
    # í•œêµ­ì–´ ë¬¸ìž¥ ë¶„ë¦¬
    kr_sentences = [s.strip() for s in re.split(r'(?<=[.!?ë‹¤])\s+', translation) if s.strip()]

    writing_items = []
    for i, eng in enumerate(sentences):
        words = eng.split()
        # ëŒ€ë¬¸ìžâ†’ì†Œë¬¸ìž ë³€í™˜ (ì²« ë‹¨ì–´, I/ê³ ìœ ëª…ì‚¬ ì œì™¸)
        processed = []
        for j, w in enumerate(words):
            if j == 0 and w[0].isupper() and w not in ['I', 'I,']:
                # ê³ ìœ ëª…ì‚¬ ì²´í¬ (ê°„ë‹¨ížˆ: 2ê¸€ìž ì´ìƒ ëŒ€ë¬¸ìž ì‹œìž‘)
                if not (len(w) > 1 and w[1:].islower() and w[0].isupper() and any(c.isupper() for c in w)):
                    w = w[0].lower() + w[1:]
            processed.append(w)
        # ë§ˆì¹¨í‘œ/ëŠë‚Œí‘œ/ë¬¼ìŒí‘œ ì œê±°
        last = processed[-1]
        if last.endswith(('.', '!', '?')):
            processed[-1] = last[:-1]
        # ì…”í”Œ
        shuffled = processed.copy()
        random.shuffle(shuffled)
        scrambled = ' / '.join(shuffled)

        kr = kr_sentences[i] if i < len(kr_sentences) else f"ë¬¸ìž¥ {i+1}"
        writing_items.append({
            "korean": kr,
            "scrambled": scrambled,
            "answer": eng
        })

    data = {"writing_items": writing_items}
    save_step(passage_dir, "step7_writing", data)
    return data

# ============================================================
# STEP 8: ì •ë‹µ ìƒì„±
# ============================================================
def step8_answers(all_data: dict, passage_dir: Path) -> dict:
    cached = load_step(passage_dir, "step8_answers")
    if cached:
        print("  âœ… step8 ìºì‹œ ì‚¬ìš©")
        return cached

    print("  ðŸ”„ step8: ì •ë‹µ íŽ˜ì´ì§€ ìƒì„±...")
    # ì •ë‹µ HTML ìƒì„±
    lines = []

    # Lv.1
    lines.append('<p class="ast">Lv.1 ì–´íœ˜ í…ŒìŠ¤íŠ¸</p>')
    lines.append('<p>A. (ì–´íœ˜ í…ŒìŠ¤íŠ¸ ì •ë‹µì€ í•™ìƒì´ ì§ì ‘ í™•ì¸)</p>')

    # Lv.5
    s2 = all_data.get("step2", {})
    lines.append('<p class="ast">Lv.5 ìˆœì„œ ë°°ì—´</p>')
    lines.append(f'<p>ì •ë‹µ: {s2.get("order_answer","")}</p>')
    lines.append(f'<p>ì‚½ìž… ì •ë‹µ: {s2.get("insert_answer","")}</p>')
    lines.append(f'<p>ì „ì²´ ë°°ì—´: {s2.get("full_order_answer","")}</p>')

    # Lv.6
    s3 = all_data.get("step3", {})
    correct = ', '.join(s3.get("blank_correct", []))
    lines.append('<p class="ast">Lv.6 ë¹ˆì¹¸ ì¶”ë¡ </p>')
    lines.append(f'<p>ì •ë‹µ: {correct}</p>')

    # Lv.7
    s4 = all_data.get("step4", {})
    correct = ', '.join(s4.get("topic_correct", []))
    lines.append('<p class="ast">Lv.7 ì£¼ì œ ì°¾ê¸°</p>')
    lines.append(f'<p>ì •ë‹µ: {correct}</p>')

    # Lv.8 ê´„í˜¸
    s5 = all_data.get("step5", {})
    lines.append('<p class="ast">Lv.8 ì–´ë²• (ê´„í˜¸)</p>')
    for a in s5.get("grammar_bracket_answers", []):
        if isinstance(a, dict):
            lines.append(f'<p>({a.get("num","")}) {a.get("answer","")}</p>')

    # Lv.8 ì„œìˆ í˜•
    lines.append('<p class="ast">Lv.8 ì„œìˆ í˜•</p>')
    for a in s5.get("grammar_error_answers", []):
        if isinstance(a, dict):
            lines.append(f'<p>{a.get("error","")}->{a.get("original","")}({a.get("reason","")})</p>')

    # Lv.9-1
    s6 = all_data.get("step6", {})
    lines.append('<p class="ast">Lv.9-1 ì–´íœ˜ Part A</p>')
    for a in s6.get("vocab_parta_answers", []):
        if isinstance(a, dict):
            lines.append(f'<p>({a.get("num","")}) {a.get("answer","")}</p>')

    lines.append('<p class="ast">Lv.9-1 ì–´íœ˜ Part B</p>')
    for a in s6.get("vocab_partb_answers", []):
        if isinstance(a, dict):
            correct_list = ', '.join(a.get("correct", []))
            lines.append(f'<p>{a.get("num","")}: {correct_list}</p>')

    # Lv.9-2
    kr_ans = ', '.join(s6.get("content_match_kr_answer", []))
    en_ans = ', '.join(s6.get("content_match_en_answer", []))
    lines.append('<p class="ast">Lv.9-2 ë‚´ìš©ì¼ì¹˜</p>')
    lines.append(f'<p>í•œêµ­ì–´: {kr_ans}</p>')
    lines.append(f'<p>ì˜ì–´: {en_ans}</p>')

    # Lv.10
    s7 = all_data.get("step7", {})
    lines.append('<p class="ast">Lv.10 ì˜ìž‘</p>')
    for item in s7.get("writing_items", []):
        lines.append(f'<p>{item.get("answer","")}</p>')

    answers_html = '\n'.join(lines)
    data = {"answers_html": answers_html}
    save_step(passage_dir, "step8_answers", data)
    return data

# ============================================================
# ì „ì²´ ë°ì´í„° â†’ í…œí”Œë¦¿ ë³€ìˆ˜ë¡œ ë³€í™˜
# ============================================================
def merge_to_template_data(passage: str, meta: dict, all_steps: dict) -> dict:
    """ëª¨ë“  ë‹¨ê³„ ê²°ê³¼ë¥¼ í…œí”Œë¦¿ ë³€ìˆ˜ë¡œ ë³‘í•©"""
    s1 = all_steps["step1"]
    s2 = all_steps["step2"]
    s3 = all_steps["step3"]
    s4 = all_steps["step4"]
    s5 = all_steps["step5"]
    s6 = all_steps["step6"]
    s7 = all_steps["step7"]
    s8 = all_steps["step8"]

    return {
        # ë©”íƒ€ ì •ë³´
        "subject": meta.get("subject", ""),
        "publisher": meta.get("publisher", ""),
        "lesson_num": meta.get("lesson_num", ""),
        "lesson_n": meta.get("lesson_n", ""),
        "challenge_title": meta.get("challenge_title", ""),
        # ì§€ë¬¸/ë²ˆì—­
        "passage": passage,
        "translation": s1.get("translation", ""),
        # Lv.1 ì–´íœ˜
        "vocab": s1.get("vocab", []),
        "test_a": s1.get("test_a", []),
        "test_b": s1.get("test_b", []),
        "test_c": s1.get("test_c", []),
        # Lv.3 í•µì‹¬ë¬¸ìž¥
        "key_sentences": s1.get("key_sentences", []),
        # Lv.5 ìˆœì„œ/ì‚½ìž…
        "order_intro": s2.get("order_intro", ""),
        "order_paragraphs": s2.get("order_paragraphs", []),
        "order_choices": s2.get("order_choices", []),
        "insert_sentence": s2.get("insert_sentence", ""),
        "insert_passage": s2.get("insert_passage", ""),
        "full_order_blocks": s2.get("full_order_blocks", []),
        # Lv.6 ë¹ˆì¹¸
        "blank_passage": s3.get("blank_passage", ""),
        "blank_options": s3.get("blank_options", []),
        # Lv.7 ì£¼ì œ
        "topic_passage": s4.get("topic_passage", ""),
        "topic_options": s4.get("topic_options", []),
        # Lv.8 ì–´ë²•
        "grammar_bracket_passage": s5.get("grammar_bracket_passage", ""),
        "grammar_bracket_count": s5.get("grammar_bracket_count", 13),
        "grammar_error_passage": s5.get("grammar_error_passage", ""),
        "grammar_error_count": s5.get("grammar_error_count", 8),
        # Lv.9
        "vocab_advanced_passage": s6.get("vocab_advanced_passage", ""),
        "vocab_partb": s6.get("vocab_partb", []),
        "content_match_kr": s6.get("content_match_kr", []),
        "content_match_en": s6.get("content_match_en", []),
        # Lv.10 ì˜ìž‘
        "writing_items": s7.get("writing_items", []),
        # ì •ë‹µ
        "answers_html": s8.get("answers_html", ""),
    }

# ============================================================
# PDF ë Œë”ë§
# ============================================================
def render_pdf(template_data: dict, output_path: Path):
    """Jinja2 + WeasyPrintìœ¼ë¡œ PDF ìƒì„±"""
    from jinja2 import Environment, FileSystemLoader
    from weasyprint import HTML

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    tmpl = env.get_template("template.html")
    html = tmpl.render(**template_data)
    HTML(string=html).write_pdf(str(output_path))
    print(f"  ðŸ“„ PDF ìƒì„±: {output_path.name}")

# ============================================================
# ë©”ì¸: ë‹¨ì¼ ì§€ë¬¸ ì²˜ë¦¬
# ============================================================
def process_passage(passage: str, meta: dict, passage_id: str, force=False):
    """ì§€ë¬¸ 1ê°œ â†’ ì „ì²´ ì›Œí¬ë¶ ìƒì„±"""
    passage_dir = DATA_DIR / passage_id
    if force:
        import shutil
        if passage_dir.exists():
            shutil.rmtree(passage_dir)

    print(f"\n{'='*50}")
    print(f"ðŸ“ ì§€ë¬¸ ì²˜ë¦¬: {passage_id} ({meta.get('challenge_title','')})")
    print(f"{'='*50}")

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage) if s.strip()]
    all_steps = {}

    # Step 1: ê¸°ë³¸ ë¶„ì„
    all_steps["step1"] = step1_basic_analysis(passage, passage_dir)
    sentences_from_api = all_steps["step1"].get("sentences", sentences)

    # Step 2: Lv.5 ìˆœì„œ/ì‚½ìž…
    all_steps["step2"] = step2_order(passage, sentences_from_api, passage_dir)

    # Step 3: Lv.6 ë¹ˆì¹¸
    all_steps["step3"] = step3_blank(passage, passage_dir)

    # Step 4: Lv.7 ì£¼ì œ
    all_steps["step4"] = step4_topic(passage, passage_dir)

    # Step 5: Lv.8 ì–´ë²•
    all_steps["step5"] = step5_grammar(passage, passage_dir)

    # Step 6: Lv.9 ì–´íœ˜+ë‚´ìš©ì¼ì¹˜
    all_steps["step6"] = step6_vocab_content(passage, passage_dir)

    # Step 7: Lv.10 ì˜ìž‘ (ë¡œì»¬)
    translation = all_steps["step1"].get("translation", "")
    all_steps["step7"] = step7_writing(sentences_from_api, translation, passage_dir)

    # Step 8: ì •ë‹µ
    all_steps["step8"] = step8_answers(all_steps, passage_dir)

    # ë³‘í•© + PDF
    template_data = merge_to_template_data(passage, meta, all_steps)

    # ðŸ”’ ì½˜í…ì¸  ê¸¸ì´ ê²€ì¦ (íŽ˜ì´ì§€ ë°€ë¦¼ ë°©ì§€)
    warnings = []
    bp = template_data.get("blank_passage", "")
    if len(bp) > 1200:
        warnings.append(f"blank_passage ê¸¸ì´ {len(bp)} (ê¶Œìž¥ 1200 ì´ë‚´)")
    gp = template_data.get("grammar_bracket_passage", "")
    if len(gp) > 1600:
        warnings.append(f"grammar_bracket_passage ê¸¸ì´ {len(gp)} (ê¶Œìž¥ 1600 ì´ë‚´)")
    gep = template_data.get("grammar_error_passage", "")
    if len(gep) > 1200:
        warnings.append(f"grammar_error_passage ê¸¸ì´ {len(gep)} (ê¶Œìž¥ 1200 ì´ë‚´)")
    if warnings:
        print(f"  âš ï¸ ì½˜í…ì¸  ê¸¸ì´ ê²½ê³  (íŽ˜ì´ì§€ ë°€ë¦¼ ê°€ëŠ¥):")
        for w in warnings:
            print(f"     - {w}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_name = f"{meta.get('lesson_num','')}ê³¼_{meta.get('challenge_title','ì›Œí¬ë¶')}_ì›Œí¬ë¶.pdf"
    pdf_path = OUTPUT_DIR / pdf_name
    render_pdf(template_data, pdf_path)

    print(f"âœ… ì™„ë£Œ: {pdf_name}")
    return pdf_path

# ============================================================
# ë°°ì¹˜ ì²˜ë¦¬: ì—¬ëŸ¬ ì§€ë¬¸
# ============================================================
def process_batch(passages: list[dict]):
    """ì—¬ëŸ¬ ì§€ë¬¸ì„ ìˆœì°¨ ì²˜ë¦¬
    passages: [{"passage": "...", "meta": {...}, "id": "01"}, ...]
    """
    results = []
    total = len(passages)
    for i, item in enumerate(passages):
        print(f"\nðŸ”µ [{i+1}/{total}] ì²˜ë¦¬ ì‹œìž‘...")
        try:
            pdf = process_passage(item["passage"], item["meta"], item["id"])
            results.append({"id": item["id"], "status": "done", "pdf": str(pdf)})
        except Exception as e:
            print(f"âŒ ì‹¤íŒ¨: {item['id']} - {e}")
            results.append({"id": item["id"], "status": "error", "error": str(e)})

    # ê²°ê³¼ ìš”ì•½
    print(f"\n{'='*50}")
    print("ðŸ“Š ê²°ê³¼ ìš”ì•½")
    done = sum(1 for r in results if r["status"] == "done")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"  âœ… ì„±ê³µ: {done}/{total}")
    if err:
        print(f"  âŒ ì‹¤íŒ¨: {err}/{total}")
        for r in results:
            if r["status"] == "error":
                print(f"     - {r['id']}: {r['error']}")
    return results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ì‚¬ìš©ë²•: python pipeline.py <ì§€ë¬¸íŒŒì¼.txt> [ê³¼ë²ˆí˜¸] [Challengeì œëª©]")
        print("  ë˜ëŠ”: python pipeline.py --batch <ëª©ë¡íŒŒì¼.json>")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            batch = json.load(f)
        process_batch(batch)
    else:
        passage_file = sys.argv[1]
        with open(passage_file, 'r', encoding='utf-8') as f:
            passage = f.read().strip()
        lesson_num = sys.argv[2] if len(sys.argv) > 2 else "1"
        title = sys.argv[3] if len(sys.argv) > 3 else Path(passage_file).stem
        meta = {
            "subject": "ìˆ˜íŠ¹ ì˜ì–´", "publisher": "EBS",
            "lesson_num": lesson_num, "lesson_n": lesson_num,
            "challenge_title": title
        }
        process_passage(passage, meta, f"passage_{lesson_num}_{title}")
