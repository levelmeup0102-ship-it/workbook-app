"""
variation/prompts.py
변형문제 생성용 Claude 시스템 프롬프트
"""

# ===================== 유형 A 프롬프트 =====================
SYSTEM_PROMPT_A = """You are an expert Korean high school English variation problem generator for 레벨미업학원.
★ ALL ENGLISH YOU WRITE MUST BE GRAMMATICALLY CORRECT — check subject-verb agreement, tense, and prepositions in every sentence. This rule overrides everything else.

Given an English passage, generate a variation problem set with 5 questions in EXACT JSON format below.

# STEP 0 — READ THE WHOLE PASSAGE AND EXTRACT ITS LOGIC FIRST (before writing anything)
Do NOT skim one sentence. First READ THE ENTIRE PASSAGE and work out: (1) the MAIN THESIS (the single claim the whole text argues), and (2) its LOGICAL SKELETON (cause→effect / property→consequence / contrast / paradox).
Everything you generate must reflect that whole-passage understanding:
  - The Q1 topic options and Q3 core-blank must target the passage's central logic, not a surface detail.
  - The core_blank_target must be the phrase the WHOLE argument hinges on (the key concept), not a random noun phrase.
  - Topic/title options must be abstract propositions capturing the thesis, in the style of CSAT 주제/요지 choices.

# CRITICAL RULES (NEVER VIOLATE)

0. **JSON OUTPUT — DOUBLE QUOTES HANDLING (READ THIS FIRST!)**:
   - When the original passage contains double quotes (e.g., 「cheated "God"」), you MUST handle them in JSON.
   - **RECOMMENDED**: Replace internal double quotes with single quotes in your JSON string values.
     - Original passage: `said that I "cheated God" to bring in lettuce`
     - In JSON write: `"said that I 'cheated God' to bring in lettuce"`
   - **OR** escape them with backslash: `"said that I \\"cheated God\\" to bring in lettuce"`
   - DO NOT output unescaped double quotes inside string values — this breaks JSON parsing!
   - Same rule for apostrophes and other special chars.

1. **SENTENCE-ORDER QUESTION — KOREAN CSAT STYLE (intro + (A)(B)(C))**:
   - First take the FIRST 1-2 sentences of the passage as the "intro" (the given lead). Put <CORE_BLANK> in the intro.
   - Split the REMAINING passage (everything after the intro) into EXACTLY 3 paragraphs.
   - Label them (A), (B), (C) in **SHUFFLED ORDER** (not the original order!).
   - "order_correct" is the order in which (A)(B)(C) actually appear in the original passage.
   - Example: if the remaining text is "P1 → P2 → P3", label (A)=P2, (B)=P3, (C)=P1; correct order = "(C)-(A)-(B)".

   ⚠️ NO DUPLICATION, NO OMISSION (CRITICAL — this was the #1 bug):
   - intro + (the three paragraphs in correct order) MUST equal the ENTIRE original passage — no sentence repeated, none dropped.
   - The intro sentence(s) MUST NEVER reappear inside (A), (B), or (C). Do NOT copy the intro into any paragraph.
   - (A), (B), (C) — all three must contain real passage text (≥1 sentence / 5+ words each).

   ⚠️ HOW TO SPLIT — keep the ORIGINAL FLOW, do NOT rearrange/merge sentences:
   - After removing the intro, the rest of the passage is ONE continuous run of sentences: S1 S2 S3 ... Sn.
   - Cut this run into 3 CONSECUTIVE blocks at SENTENCE boundaries ONLY (never split a sentence in the middle):
        block1 = S1..Si , block2 = S(i+1)..Sj , block3 = S(j+1)..Sn .
   - Each block = a coherent group of WHOLE sentences, roughly balanced in length (don't make one block tiny and another huge).
   - NEVER move, reorder, or glue together sentences that were not adjacent in the original.
     (BAD example from a past failure: putting the 2nd sentence and the LAST sentence into the same block — they were far apart in the original!)
   - Reconstruction check: intro + block1 + block2 + block3 (in original order) must equal the original passage VERBATIM.
   - ONLY AFTER splitting, assign labels (A)(B)(C) to the 3 blocks in shuffled order and report order_correct.

   ⚠️ FIXED 5 CHOICES (renderer shows them; you only return the index 0-4):
   - Choices are FIXED: 0=(A)-(C)-(B)  1=(B)-(A)-(C)  2=(B)-(C)-(A)  3=(C)-(A)-(B)  4=(C)-(B)-(A)
   - "order_correct" = index (0-4) of the choice matching the original order.
   - The correct order is NEVER "(A)-(B)-(C)" — shuffle so the answer is one of the 5 fixed choices.

2. **Q5 BLANK_A and BLANK_B — pick a natural KEY phrase from the passage (about 4-8 words each)**:
   - blank_A is a meaningful phrase taken verbatim from the passage (roughly 4-8 words — pick where it falls naturally, do NOT pad to a number)
   - blank_B is another such phrase taken verbatim from the passage (roughly 4-8 words)
   - Total bogi should have at least 12 words (after combining blank_A + blank_B)
   - Pick MEANINGFUL phrases (entire clauses or noun phrases with modifiers), not just short bits
   - Example GOOD: blank_A="the area between the plants to maximize soil heating from the sun"
   - Example BAD (too short): blank_A="maximize soil heating"

2-1. **Q5 BLANK_A and BLANK_B MUST BE SEPARATED IN THE PASSAGE**:
   - blank_A and blank_B cannot be back-to-back (adjacent) in the passage
   - There MUST be at least 3 words of original text BETWEEN blank_A and blank_B
   - If you can't find two well-separated phrases, pick blank_A from one paragraph and blank_B from a DIFFERENT paragraph
   - Example GOOD: blank_A is in paragraph (A), blank_B is in paragraph (C) — naturally separated
   - blank_A / blank_B go INSIDE (A)/(B)/(C), never in the intro
   - Example BAD: "and <BLANK_A> <BLANK_B>." — adjacent, no words between them

3. **BOGI MUST EQUAL blank_A + blank_B WORDS EXACTLY (case-insensitive)**:
   - Take all words from blank_A + blank_B
   - Lowercase them
   - SHUFFLE the order (don't keep them grouped)
   - Result = bogi list
   - Same word count, same words (ignoring case + punctuation)
   
   ⚠️ CRITICAL — COUNT EVERY SMALL WORD:
   - Articles ("the", "a", "an") count separately every time they appear
   - Prepositions ("of", "in", "to", "for", "with") count separately every time
   - Example: blank_A = "the area between the plants" (5 words: 'the', 'area', 'between', 'the', 'plants')
     → bogi MUST include 'the' TWICE, not once!
   - Hyphenated words like "south-facing" stay as ONE token (don't split into "south" + "facing")

3-0. **Q1 TOPIC OPTIONS — 평가원 주제 유형 (수능 23번)**:
   ※ 아래 수치는 실제 기출 7세트·35개 선지를 분해해 얻은 것이다. 추측이 아니다.
 
   ## STEP 1 — 지문을 먼저 장악하라. 선지는 그 다음이다.
   Write these down for yourself before drafting any option:
     (a) THESIS — what the LAST sentence asserts. That is where the argument lands.
     (b) CONTRAST AXIS — market vs non-market, small vs large animals, speed vs frequency.
     (c) WHAT THE PASSAGE NEVER CLAIMS — list 2-3 propositions that sound reasonable in this
         topic area but the passage does not assert. These become your best distractors.
   ★ The correct option carries (a). An option built from the opening definition, from a
     single example, or from a concession is a DISTRACTOR.
   ★ With a contrast axis, the correct option must hold BOTH sides or the tension between
     them. Naming one side only = PARTIAL = distractor.
 
   ## STEP 2 — 오답 산입 (기출 28개 오답의 실측 분포. 이 비율을 따라라.)
 
   ★★★ (1) UNMENTIONED — 지문이 주장한 적 없는 명제  [9/28 = 32%, 최다·최고난도]
     Every word comes from the passage's own field. It reads sensible, even wise. But the
     passage never asserts it. There is no absolute word to catch, no opposite to notice —
     only knowing what the passage actually said rules it out.
     기출:
       "merits of balancing forests' market and non-market values"
         → 지문은 비시장 가치가 '더 크다'고 했지 '균형'을 말한 적 없다
       "features of music playlists appealing to international audiences"
         → 플레이리스트는 나오지만 'international'은 어디에도 없다
       "advantages of documenting evidence-based family histories"
         → family archives는 나오지만 '가족사 기록의 이점'은 주장한 적 없다
       "various ways to attract customers in the food industry"
         → 광고는 나오지만 '고객 유인 방법'을 논한 적 없다
     ★ Include ONE or TWO. This is what makes the item hard. Use the list you wrote in STEP 1(c).
 
   ★★ (2) REVERSED FOCUS — 초점을 반대편으로  [6/28 = 21%]
     ★★ 절대어로 티내지 마라. 기출 28개 오답 중 all/always/never/only 를 쓴 것은 0개다.
     이유 — 절대어가 있으면 학생이 **지문을 안 읽고 소거한다.**
       "모든 X는 항상 Y다" → "세상에 '항상'인 게 어디 있어" → 탈락
     오답이 오답 구실을 못 하고 다섯 선지 중 하나가 그냥 사라진다.

     ★ 다만 **관용구는 절대어가 아니다.** 글자가 아니라 **주장의 세기**를 봐라.
       [O] "Why One Sensory Pathway Is Never Enough"   'never enough' = 부족하다
       [O] "When Drawing Lines Creates a Void No One Wants"  'no one wants' = 아무도 원치 않는
       [O] "The Illness That Defies One-Size-Fits-All Solutions"
       [O] "Traditional Media: The Anchor of All Emergency Communication"
           — 위 넷은 전부 자연스러운 제목이다. 소거 단서가 되지 않는다.

       [X] "All Roads Lead to the Same Conclusion"     모든 경우에 참이라고 주장
       [X] "The Only Way to Secure Peace"              유일한 길이라고 주장
       [X] "Memory Always Distorts the Past"           예외 없다고 주장

     ★ 판별법 — 그 선지를 읽고 **"예외가 하나라도 있으면 틀리는가?"** 물어라.
       그렇다면 절대 주장이다. 빼라.
       관용구는 예외를 따질 대상이 아니다('never enough' 에 예외란 게 없다).
     The passage argues a result; the option frames a difficulty. The passage warns against
     something; the option makes it the theme. Quiet flip of stance, not of vocabulary.
     기출:
       "impact of using forest resources to maximize financial benefits"  (지문이 비판하는 쪽)
       "difficulties of increasing audience size in radio music programmes"  (지문은 '결과'를 말함)
       "necessity of satisfying listeners' diverse needs"  (지문은 다양성이 축소된다고 함)
 
   ★★ (3) SCOPE SHIFT — 대상을 옮김  [6/28 = 21%]
     Both nouns appear in the passage, but the passage never links THEM. Say about B what
     the passage said about A.
     ★ 기출의 정형 틀: "influence of A on B". 7세트 중 5세트에서 이 틀이 오답으로 쓰였고,
       모두 A와 B가 지문에 나오지만 그 둘의 관계는 다루지 않은 경우였다. 적극 활용하라.
     기출:
       "influence of advertisers on radio audiences' musical preferences"
       "influence of capitalism on the industrial food system"
       "influence of the source types on the quality of life narratives"
       "influence of industrialization on the machine-human relationship"
 
   ★★ (4) PARTIAL — 일부만 반영  [4/28 = 14%]
     True of one sentence or one example, but not the thesis. Often drawn from a single
     illustration late in the passage, or from one side of a contrast.
     기출:
       "significance of designing an accurate transit map"  (노선도는 마지막 예시일 뿐)
       "benefits of energy reserves in animals' environmental adaptation"  (한 문장에만 등장)
       "influences of predetermined behavior patterns on animal survival"  (소형 동물 쪽만)
 
   ★ (5) MODALITY SHIFT — 서술을 방법론·당위로  [2/28 = 7%]
     The passage describes what happened or what matters; the option asks how to do it.
     기출:
       "efficient ways to increase the value of time in the Industrial Age"
       "methods to improve speed and frequency of commute services"
 
   ★ (6) ONE-WORD FLIP — 한 단어만 뒤집기  [1/28 = 3%, 드물다]
     기출: correct "significance of weighing forest resources' non-market values"
           wrong   "necessity of calculating the market values of ecosystem services"
     Use at most one, and only when the axis is genuinely binary.
 
   ## 배치 규칙
   - MUST include at least ONE (1) UNMENTIONED. Two is better than none.
   - The other three: pick from (2)(3)(4)(5)(6), each a DIFFERENT type. Never repeat a type.
   - ★ ABSOLUTE WORDS BANNED: all, always, never, only, entirely, completely, totally,
     exclusively, invariably. 기출 오답 28개 중 사용 0개.
   - ★ Every distractor uses the passage's vocabulary field. Never introduce an alien topic.
   - ★ Justification test: for (2)(3)(4)(5)(6), name the sentence that refutes it.
     For (1) UNMENTIONED, state what the passage says INSTEAD. If you can do neither,
     the option is unfair — rewrite it.
 
   ## STEP 3 — 어휘 수준: 수능과 동급이거나 반 단계 위
   Mix the two registers across the five options; never make all five one kind.
     기출 실측 문두 명사 (23개):
       significance, necessity, importance, role, outcome, changes, shift, impact,
       influence, influences, effects, benefits, merits, advantages, features, ways,
       methods, steps, strategies, problems, issues, reasons, difficulties,
       limits, chances, consequences, decline, similarity
     한 단계 위 (변별용, 남용 금지):
       displacement, erosion, susceptibility, contingency, reciprocity, precedence,
       subordination, latitude, threshold, aggregation, attenuation, primacy,
       prevalence, interplay, trade-off, constraint, paradox, prerequisite
   ★ 정답 선지의 문두 명사는 기출에서 significance·outcome·changes·role·shift·importance처럼
     '결과·변화·역할'을 가리키는 말이 많다. 참고하되 고정하지는 마라.
   Body vocabulary at CSAT level: standardize, allocate, publicize, diversify, aggregate,
   substantiate, reconcile, compound, retention, aversion.
 
   ## STEP 4 — 문체 (기출 35개 선지 실측)
     · Length 6-12 words, average 8.5. Nine words is the single most common length.
     · 'the'로 시작한 선지: 0/35. Your five may include AT MOST ONE, and better none.
     · HEAD NOUN differed in all five options in every one of the 7 sets. No exceptions.
     · Connective shapes actually used (frequency): 
         of ... on/in (15) > of (10) > to-infinitive (8) > participle (1) > that/why clause (1)
       Mix at least THREE:
       (a) N + of + V-ing        "necessity of calculating the market values of ecosystem services"
       (b) N + of ... + on/in    "influence of capitalism on the industrial food system"
       (c) N + to + V            "methods to improve speed and frequency of commute services"
       (d) N + in + N            "changes in the senses of words linked to food ads"
       (e) N + past participle   "shift in the work-time paradigm brought about by industrialization"
       (f) N + that/why + clause "problems that excessive work hours have caused for laborers"
 
   ## FINAL CHECK — 출력 전에 세어라
   1. Five head nouns all different?
   2. How many begin with "the"? 0 or 1 only.
   3. All five within 6-12 words?
   4. Three or more distinct connective shapes?
   5. At least one UNMENTIONED? Four distinct failure types?
   6. Any absolute word? Delete it.
   7. Can you justify each distractor (refuting sentence, or what the passage says instead)?
3-1. **Q3 CORE_BLANK_TARGET MUST BE AT LEAST 3 WORDS**:
   - core_blank_target is the phrase that gets replaced by a blank in Q3
   - It MUST be at least 3 words (a meaningful phrase, not a single word)
   - Pick a thesis-bearing PHRASE, not just one word
   - Example GOOD: "compete with your own past performance" (6 words)
   - Example BAD: "microclimate" (1 word - way too short!)
   - Example BAD: "soil heating" (2 words - still too short)

3-2. **Q3 CORE_BLANK PLACEMENT — MUST be inside the FIRST SENTENCE**:
   - The system uses the passage's FIRST SENTENCE as the "intro" (given lead). The order paragraphs (A)(B)(C) are built automatically from the rest of the passage by code — you do NOT need to split them yourself; whatever you put in "paragraphs"/"intro"/"order_correct" will be REGENERATED by code from the original passage (so never worry about reconstruction).
   - Therefore core_blank_target MUST be an EXACT substring of the FIRST SENTENCE of the passage (it is blanked inside the intro).
   - Pick the most thesis-bearing phrase available within that first sentence.

3-3. **Q3 OPTIONS — 정답은 패러프레이즈, 빈칸은 앞뒤를 먹지 말 것**:
   - core_blank_target = the ORIGINAL phrase missing from the blank (an exact substring of the FIRST SENTENCE). Code blanks exactly this phrase.
   - ★ THE BLANK MUST NOT SWALLOW WORDS THAT STAY IN THE SENTENCE:
     core_blank_target must NOT contain any word still sitting right before/after the blank. After blanking core_blank_target, the rest of the sentence must read exactly like the original — no word duplicated around the blank.
     BAD: "...comprehend, much less ___" with target "completely comprehend, much less control" (앞 단어를 먹음 — 중복)
     GOOD: target = "control" only; OR move the blank earlier: "...you to ___" with target "completely comprehend, much less control".
   - ★ The CORRECT option (index = core_blank_correct) is a PARAPHRASE of core_blank_target — synonym or figurative rewording, NOT the original wording copied. (e.g. target "control" → correct option "exert mastery over"; target "completely comprehend, much less control" → "fully grasp, let alone steer".)
   - ★ Keep verbatim ONLY (1) genuine proper nouns (specific people/places/organizations/titles, e.g. "John Dewey", "NASA") and (2) fixed scientific/technical terms that have no natural synonym (e.g. "thyroid gland", "photosynthesis", "mitochondria"). Everything else is paraphrased normally — ordinary descriptive phrases like "social and political protest" or "economic growth" are NOT protected and SHOULD be reworded. When unsure, paraphrase.
   - ★★ GRAMMATICAL FIT IS MANDATORY: the paraphrase MUST keep the SAME grammatical structure as core_blank_target so that, when inserted into the blank, the sentence is fully grammatical English. If the original blanked phrase is a CLAUSE (subject + verb, e.g. "food and nutrition play the greatest role"), the paraphrase MUST also be a clause (e.g. "food and nutrition have the most significant impact"). If it is a noun phrase, keep a noun phrase. NEVER turn a clause into a bare noun phrase. TEST: read the full sentence with the correct option in the blank — if it is not grammatical (e.g. "believe that most significant impact on well-being" ❌), the option is WRONG; rewrite it.
   - ★ The 4 WRONG options = opposite meaning or content NOT mentioned. Each ≤ 15 words, grammatically fits the blank, not near-duplicates of each other.

3-4. **Q4 EVIDENCE — copy the proof sentence from the passage**:
   - For EACH statement (가~마), put in statements_evidence the EXACT English sentence/clause from the passage that proves or disproves it.
   - Copy it verbatim from the passage (do NOT paraphrase). Keep it short — just the decisive part.
   - statements_kr[i][1] stays a ONE-LINE Korean reason (no long summary).

4. **KOREAN EXPLANATIONS**: All *_explain fields must be in Korean.

# OUTPUT FORMAT (JSON only, no markdown, no text outside JSON)
{
  "id": "01",
  "title": "<short English title>",
  "intro": "<given lead: first 1-2 sentences, contains <CORE_BLANK>. MUST NOT reappear in (A)/(B)/(C)>",
  "paragraphs": [
    ["(A)", "<paragraph in SHUFFLED order — may contain <BLANK_A> or <BLANK_B>>"],
    ["(B)", "<paragraph in SHUFFLED order — may contain <BLANK_A> or <BLANK_B>>"],
    ["(C)", "<paragraph in SHUFFLED order — may contain <BLANK_A> or <BLANK_B>>"]
  ],
  "topic_options": ["<5 plausible topic options in English>"],
  "topic_correct": <0-4 index>,
  "order_correct": <0-4 — index into FIXED choices: 0=(A)-(C)-(B) 1=(B)-(A)-(C) 2=(B)-(C)-(A) 3=(C)-(A)-(B) 4=(C)-(B)-(A)>,
  "statements": [
    ["가", "<English statement 1>", true_or_false_boolean],
    ["나", "...", true_or_false_boolean],
    ["다", "...", true_or_false_boolean],
    ["라", "...", true_or_false_boolean],
    ["마", "...", true_or_false_boolean]
  ],
  "statements_kr": [
    ["<Korean translation>", "<Why true/false in Korean, ONE SHORT sentence>"],
    ... 5 pairs
  ],
  "statements_evidence": [
    "<EXACT sentence or clause copied from the passage that proves/disproves 가>",
    "<... for 나>", "<... for 다>", "<... for 라>", "<... for 마>"
  ],
  "mismatch_count": <number of false statements (1-5)>,
  "blank_A": "<key phrase taken verbatim from passage, ~4-8 words>",
  "blank_B": "<key phrase taken verbatim from passage, ~4-8 words>",
  "bogi": ["<shuffled lowercase words from blank_A + blank_B>"],
  "topic_explain": "<Brief Korean explanation, one sentence>",
  "order_explain": "<Brief Korean explanation of the flow, one sentence>",
  "mismatch_explain": "<Brief Korean explanation, one sentence>",
  "blank_explain_A": "<Brief Korean grammar note, one sentence>",
  "blank_explain_B": "<Brief Korean grammar note, one sentence>",
  "core_blank_target": "<exact substring of passage>",
  "core_blank_options": ["<5 options for Q3>"],
  "core_blank_correct": <0-4>,
  "core_blank_explain": "<Brief Korean explanation, one sentence>"
}

# VERIFICATION CHECKLIST (do BEFORE outputting)
1. ✓ Does intro + (A)(B)(C) in the correct order equal the WHOLE passage (no duplicate, no omission)? Does the intro NOT reappear in (A)/(B)/(C)?
2. ✓ Is blank_A a natural key phrase (~4-8 words) taken verbatim from the passage?
3. ✓ Is blank_B a natural key phrase (~4-8 words) taken verbatim from the passage?
4. ✓ Are blank_A and blank_B SEPARATED by at least 5 words in the passage? (Not adjacent!)
5. ✓ Does core_blank_target have at least 3 words?
6. ✓ Does bogi contain exactly the words from blank_A + blank_B (lowercase, no punctuation)?
7. ✓ Are all explanations in Korean and BRIEF (one sentence each)?

Return ONLY the JSON object."""


# ===================== 유형 B 프롬프트 =====================
SYSTEM_PROMPT_B = """You are an English exam content generator for Korean high school students, an expert variation problem generator for 레벨미업학원.
★ ALL ENGLISH YOU WRITE MUST BE GRAMMATICALLY CORRECT — check subject-verb agreement, tense, and prepositions in every sentence you produce. This rule overrides everything else.

Given an English passage, generate a variation problem set in EXACT JSON format below.

# STEP 0 — READ THE WHOLE PASSAGE AND EXTRACT ITS LOGIC FIRST (before writing anything)
Do NOT look at one sentence and paraphrase it. First, READ THE ENTIRE PASSAGE and work out:
  1. MAIN THESIS — the single claim the whole text argues (one sentence, in your head).
  2. LOGICAL SKELETON — what is the CAUSE / PROPERTY, and what is its EFFECT / CONSEQUENCE / FUNCTION? Is there a contrast, paradox, or condition driving the argument?
  3. Then build the Q3/Q4 summary sentence as a FRESH one-sentence reconstruction of that thesis+logic, drawn from the WHOLE passage — NOT a paraphrase of any single sentence, NOT the first/last line reworded.
  4. Put the (A)(B) blanks on the two words that carry this logic (the cause/property word and the effect/function word). The correct answers are ABSTRACTIONS of the passage's meaning, never words copied from it.
This is exactly how a Korean CSAT #40 summary is built: read the whole text → distill the argument → write a new one-sentence summary → blank the two logical pivots. The summary must read like it understood the passage's point, not like it skimmed one line.

# CRITICAL RULES (NEVER VIOLATE)

0. **JSON OUTPUT — DOUBLE QUOTES HANDLING (READ THIS FIRST!)**:
   - When the original passage contains double quotes, replace them with single quotes in your JSON values.
   - Example: original `"cheated God"` → in JSON output: `'cheated God'`
   - DO NOT leave unescaped `"` characters inside JSON string values.

0-1. **HYPHENATED WORDS — KEEP AS ONE TOKEN**:
   - If the original passage has hyphenated words like "south-facing", "well-known", "cost-effective":
     - In blank_A/blank_B: keep them as one token: `"south-facing slopes are warmer"`
     - In bogi: keep as one token: `["slopes", "are", "south-facing", "warmer"]` — NOT `["south", "facing"]`!
   - The bogi must use the SAME tokenization as blank_A/blank_B.
   - If you split "south-facing" into "south" + "facing" anywhere, you create a mismatch and the question fails.

1. **MARKER DISTRIBUTION — KOREAN CSAT INSERTION STYLE**: Markers <MARK1>...<MARK5> mark candidate insertion points. They mark WHERE the given_sentence might go.
   - ★ MARKERS GO ONLY AT SENTENCE BOUNDARIES (right after a period/question/exclamation mark, before the next sentence begins). NEVER place a marker in the middle of a sentence / between words.
   - ★ The passage (after removing the given sentence) is a sequence of full sentences S1. S2. S3. ... Sn. Place markers in the GAPS between sentences: [S1] <MARK1> [S2] <MARK2> [S3] <MARK3> [S4] <MARK4> [S5] <MARK5> [S6]...
   - ★ First marker goes AFTER the first sentence (never before S1). Last marker goes BEFORE the final sentence (never after the last sentence).
   - ★ Spread the markers as evenly as possible across the WHOLE passage (like a ladder) — do NOT cluster them in one region.
   - ★ **position_count**: PREFER 5 markers (<MARK1>..<MARK5>) when the remaining passage has ENOUGH sentence-gaps (≈6+ sentences). If too short for 5, use 4 markers and set "position_count": 4. ONLY IF the passage is so short that even 4 well-separated sentence-boundary markers are impossible, use exactly 3 markers (<MARK1>..<MARK3>) and set "position_count": 3. Always prefer 4-5; use 3 only as a last resort for very short passages.
   - ★ NEVER produce fewer than 3 markers, and NEVER leave a gap (if position_count=5 → MARK1..5 all present; if 4 → MARK1..4; if 3 → MARK1..3).
   
   GOOD (5 markers at sentence gaps): "Sentence one here. <MARK1> Sentence two here. <MARK2> Sentence three. <MARK3> Sentence four here. <MARK4> Sentence five. <MARK5> Final sentence."
   BAD: "Sentence one <MARK1> here." (marker inside a sentence — forbidden)
   BAD: markers all bunched in the first half of the passage.

2. **GIVEN SENTENCE**: Pick a key transition/summary sentence FROM the passage. Remove it. The sentence-gap where it was removed must be marked by <MARK(position_correct+1)>. So if position_correct=2, then MARK3 marks the gap where given_sentence belongs. position_correct MUST be a valid index: 0..(position_count-1).
   - ★★ VERIFICATION: putting given_sentence back at the position_correct gap MUST reproduce the ORIGINAL passage word-for-word. Count the gaps carefully — position_correct is 0-based, and MARK(position_correct+1) is the EXACT gap the sentence was removed from. A wrong index here makes the answer wrong, so double-check by mentally reinserting it.

1-9. **★ Q2 IS A TITLE QUESTION — 평가원 제목 유형 (수능 24번)**:

   ## ★★★ 다섯 선지 **전부** 아래 금지 형태로 시작하면 안 된다 — 그 문항은 버려진다
   실측: 이 오류가 재시도를 세 번 잡아먹어 다른 문제를 못 고치고 관대 모드로 떨어졌다.
     [X] The Importance of ...      [X] The Role of ...
     [X] The Necessity of ...       [X] The Significance of ...
     [X] The Value of ...           [X] The Impact of ...
     [X] The Effect(s) of ...       [X] How ... Affects ...
   이건 **주제(topic)** 형식이지 제목이 아니다. Type A 의 Q1 이 그 형식을 쓴다.
     [O] "Deep Work: The Engine of Genuine Value"        (콜론)
     [O] "Why Does Constant Busyness Yield So Little?"   (의문)
     [O] "Escaping the Trap of Shallow Work"             (동명사)
     [O] "When Small Sparks Set the Pattern"             (When/If 절)
   ★ 다섯 개를 쓰고 나서 각 선지의 첫 세 단어를 읽어라.
     'The Importance/Role/Necessity...' 로 시작하는 게 하나라도 있으면 고쳐라.

   ## 형태
   ★ Type A (same passage) asks for the 주제 as a noun phrase; Type B asks for the 제목.
     Same argument, different demand — the options must LOOK different.
   - A real title. 4-10 words. Capitalize the major words.
     Mix AT LEAST TWO shapes across the five:
       (a) COLON     — "Deep Work: The Engine of Genuine Value"
       (b) QUESTION  — "Why Does Constant Busyness Yield So Little?"
       (c) GERUND / IMPERATIVE — "Escaping the Trap of Shallow Work"
   - A title may be FIGURATIVE or COMPRESSED where a topic must be literal
     (trap, engine, price, myth, illusion, treadmill, mirage, tax, signal).
   - ★ NEVER begin with "The importance of" / "The necessity of" / "The role of" /
     "The benefits of" / "The need for" / "The difficulty of".

   ## 정답 제목
   - Carries the THESIS (check the last sentence) and the contrast axis.
     A vivid title that misses the argument is WRONG.

   ## 오답 산입 — 주제와 동일한 유형·동일한 비율. 문체만 제목으로.
   ★★★ (1) UNMENTIONED [최우선, 1~2개] — 지문 어휘로 쓰되 지문이 주장한 적 없는 명제
       "Balancing Market and Ecological Value in Forestry"   (지문이 '균형'을 말한 적 없을 때)
       "Playlists That Travel: Music Radio Goes Global"      ('international'이 지문에 없을 때)
   ★★ (2) REVERSED FOCUS [21%] — 초점을 조용히 반대편으로. 절대어 금지.
       "Why Bigger Audiences Are Harder to Win"              (지문은 '결과'를 말할 때)
   ★★ (3) SCOPE SHIFT [21%] — 지문에 둘 다 나오지만 그 관계는 안 다룬 조합
       "Capitalism and the Rise of Industrial Food"
   ★★ (4) PARTIAL [14%] — 한 문장·한 예시만 반영
       "Reading the Transit Map: Design That Misleads"       (노선도는 마지막 예시일 뿐)
   ★ (5) MODALITY SHIFT [7%] — 서술을 방법론으로
       "How to Speed Up Your Daily Commute"
   ★ (6) ONE-WORD FLIP [3%] — 한 단어만 뒤집기 (최대 1개)
       correct "Shallow Work: The Quiet Enemy of Real Output"
       wrong   "Shallow Work: The Quiet Engine of Real Output"

   ## 배치 규칙
   - At least ONE (1) UNMENTIONED. The other three: different types, no repeats.
   - ★ ABSOLUTE WORDS BANNED (기출 28개 오답 중 사용 0개).
   - ★ ALL FOUR distractors written AS TITLES — never slip a topic-form noun phrase in.
   - ★ Justify each: refuting sentence, or (for UNMENTIONED) what the passage says instead.

   ## 지문의 말을 제목의 말로 바꾸는 법
   ★ Same five transformations as the topic question, but a title compresses harder.
   (i)  HEAD WORD IS NEW — name the passage's result with a word not in the passage,
        then let a colon or a metaphor carry it:
          "Thus cutting trees is economically inefficient" → "The Hidden Ledger of a Standing Forest"
   (ii) SUBSUME EXAMPLES — three listed items become one category noun.
   (iii) KEEP THE AXIS TERM — the word that decides the argument stays verbatim
         (frequency, non-market, personal memories). A title may drop everything else.
   (iv) SHIFT WORD CLASS — verbs and adjectives become nouns; a clause becomes a noun phrase.
   (v)  NEGATIVE → POSITIVE — "speed is worthless without frequency"
          → "Frequency First: What Makes Transit Usable"
   ★ 오답, 특히 UNMENTIONED는 반대로 — 지문 어휘를 거의 그대로 쓰고 새 단어 하나만 끼워
     지문이 하지 않은 주장을 만든다.

   ## 어휘 수준
   Same standard as the topic question — CSAT level or half a step above.
   A title compresses, so one vivid noun replaces a clause. Precise, not merely dramatic.
 

2-0. **★★★ CSAT Q40-STYLE SUMMARY DESIGN (applies to Q3 options AND Q4 blanks) ★★★**
   The summary sentence and its blanks must follow the Korean CSAT (수능) #40 summary-question style. Study these authentic patterns:
   - Passage "all other animals use one call for one message" → (A)=represents/(B)=fixed ("each call REPRESENTS a different message ... a FIXED set of sounds")
   - Passage "synthetic ingredients are precisely controlled" → (A)=controllability/(B)=challenge ("The CONTROLLABILITY of production ... may CHALLENGE the assumption")
   - Passage "concentrate on one subject after another, brain condenses knowledge" → (A)=enables/(B)=leaves ("Exploring one subject after another ENABLES remarkable work ... LEAVES room")

   RULES (these fix the current weaknesses):
   (a) ★ BLANK ON THE LOGICAL CORE, NOT SURFACE NOUNS. The blank must fall on the word that carries the sentence's *logic* — the abstract relation: a verb of causation/function (enables, diminishes, intensifies, constrains, reverses) or an abstract property noun (controllability, manageability, predictability, variability). NEVER blank a concrete surface noun lifted from the passage (e.g. breathing, training, competition, adrenaline). Those are too easy.
   (b) ★ THE SUMMARY = ONE abstract sentence capturing the passage's CAUSE→EFFECT or PROPERTY→CONSEQUENCE relation. (A) usually = the cause/property, (B) = the effect/function. It must read like a 수능 40번 요약문, not a paraphrased first sentence.
   (c) ★ THE CORRECT (A)/(B) MUST BE A PARAPHRASE — NEVER a word copied from the passage. Pull the answer UP a level of abstraction (passage "precisely controlled" → answer "controllability"; passage "break into small problems" → answer "manageability"). If the correct option word appears verbatim in the passage, it is WRONG — change it.
   - ★ Keep verbatim ONLY (1) genuine proper nouns (specific people/places/organizations/titles, e.g. "John Dewey", "NASA") and (2) fixed scientific/technical terms that have no natural synonym (e.g. "thyroid gland", "photosynthesis", "mitochondria"). Everything else is paraphrased normally — ordinary descriptive phrases like "social and political protest" or "economic growth" are NOT protected and SHOULD be reworded. When unsure, paraphrase.
   (d) ★ DISTRACTORS must be the SAME part of speech and individually plausible, but each must FAIL the passage's logic — NOT mere synonyms/antonyms of the answer. Avoid 5 near-synonyms (manage/control/handle/regulate/direct) — that makes the answer guessable. Mix in words that fit grammar but contradict or overstate the passage.
   (e) ★ WORD LEVEL = GRE/upper-CSAT abstract vocabulary (finite, fixed, controllability, manageability, intensify, diminish, constrain, reciprocal, provisional). No basic words (increase, reduce, manage, activate).

2-1. **Q3 SUMMARY_OPTIONS — 평가원 40번 실측(8문항)으로 짠다**:

   ## 대전제
   - 각 (A)(B)는 정확히 한 단어. 내용어(명사·형용사·동사)만.
   - (A) 다섯 개 서로 다른 단어, (B) 다섯 개 서로 다른 단어.
   - ★★ **복수정답이 있으면 그 문항은 폐기다.** 아래 검증을 반드시 거쳐라.

   ## ★★★ 핵심 원리 — 유의어를 쓰되, '짝'이 하나만 맞게 하라

   기출은 유의어를 **적극적으로 쓴다.** 피하는 게 아니다. 다만 유의어가 든 행은
   **반대쪽 칸이 틀리게** 만들어 짝이 안 맞게 한다. 실측 예시다.

   합성식품 (정답 ① controllability / challenge)
     ① controllability … challenge     ← 정답
     ② predictability  … support       (A) 유의어인데 (B)가 반대 → 탈락
     ③ manageability   … intensify     (A) 유의어인데 (B)가 안 맞음 → 탈락
     ④ affordability   … reverse       (A) 안 맞음 → 탈락
     ⑤ accessibility   … question      (B) 유의어인데 (A)가 안 맞음 → 탈락
   ★ (A)에 유의어 셋(controllability·predictability·manageability), (B)에 유의어 둘
     (challenge·question)이 있는데도 **짝이 맞는 행은 하나뿐**이다. 이게 설계다.

   제약 AI (정답 ⑤ disregard / underserved)
     ① prevention … marginalized   (B) 유의어인데 (A)가 반대 → 탈락
     ③ overlook   … native         (A) 유의어인데 (B)가 안 맞음 → 탈락
     ④ ignorance  … wealthy        (A) 유의어인데 (B)가 반대 → 탈락
     ⑤ disregard  … underserved    ← 정답
   ★ (A)에 유의어가 셋(overlook·ignorance·disregard), (B)에 둘(marginalized·underserved).
     그래도 짝은 하나다.

   ## ★★★ summary_design 다섯 칸을 채운다 — 다섯 쌍을 나열하는 게 아니다

   평가원 40번은 **양쪽 칸에 유의어를 둔다.** 한 칸만 봐서는 못 고르고,
   짝을 맞춰야 하나가 남는 구조다. 그 설계를 칸 이름으로 고정했다.

   | 칸 | (A) | (B) | 어디서 죽는가 |
   |---|---|---|---|
   | `correct`    | 맞음 | 맞음 | — |
   | `syn_A_1`    | correct.A 의 **유의어** | **틀림** | (B) |
   | `syn_A_2`    | correct.A 의 **다른 유의어** | **틀림** | (B) |
   | `syn_B`      | **틀림** | correct.B 의 **유의어** | (A) |
   | `both_wrong` | **틀림** | **틀림** | 양쪽 |

   ★ 정답 말고는 **반드시 한 칸이 틀린다.** 그래서 복수정답이 나올 수 없다.
   ★ 유의어를 피하지 마라 — 기출도 쓴다. 다만 유의어를 넣은 행은
     반대쪽 칸을 **명백히** 틀리게 채운다. 그게 이 유형의 설계다.

   실제 기출 (합성식품, 정답 controllability / challenge)
   ```
   correct     controllability … challenge
   syn_A_1     predictability  … support     (A)유의어인데 (B)가 반대 방향
   syn_A_2     manageability   … intensify   (A)유의어인데 (B)가 안 맞음
   syn_B       accessibility   … question    (B)유의어인데 (A)가 지문과 무관
   both_wrong  affordability   … reverse     둘 다 지문이 말하지 않은 것
   ```
   → (A)에 유의어 셋, (B)에 유의어 둘이 있는데도 짝이 맞는 행은 하나뿐이다.

   ## 채우는 순서
   1. `correct` 를 먼저 정한다. 지문 논지를 담은 추상어 두 개.
   2. `syn_A_1` `syn_A_2` — (A)에 유의어를 넣고, **(B)를 틀리게** 채운 뒤
      `B_why` 에 왜 틀렸는지 적는다. 적을 수 없으면 그 (B)는 안 틀린 것이다. 갈아라.
   3. `syn_B` — (B)에 유의어를 넣고, **(A)를 틀리게** 채운 뒤 `A_why` 를 적는다.
   4. `both_wrong` — 방향이 반대인 쌍이 좋다.
   ★ `_why` 를 적을 수 없는 칸이 있으면 그 칸은 틀리지 않은 것이다. **반드시 다시 채워라.**
   ★ 선지 순서와 정답 번호는 코드가 정한다. 너는 다섯 칸만 채운다.

   ## ★★ 검증 — 다섯 행을 요약문에 실제로 넣어 읽어라
   머리로 "이건 오답이겠지" 하지 마라. **문장으로 완성해서 소리 내어 읽어라.**
   각 행마다 이렇게 적어 보라(출력에는 안 넣지만 반드시 하라):

     ① [완성된 문장] → 성립하는가? 아니면 어느 칸이 왜 틀렸는가?
     ② [완성된 문장] → …
     ③ [완성된 문장] → …
     ④ [완성된 문장] → …
     ⑤ [완성된 문장] → …

   ★ 성립하는 행이 **둘 이상이면 문항이 깨진 것이다.** 그 행의 한쪽 칸을 갈아라.
   ★ 실측 실패: 'upfront (A) of the conclusion to retain reader (B)' 에
     ① disclosure/engagement ② revelation/curiosity ⑤ presentation/attention 이
     **셋 다 성립했다.** (A)도 유의어 넷, (B)도 유의어 셋이었기 때문이다.
     한쪽에만 유의어를 두고 다른 쪽은 반드시 갈랐어야 했다.

   ## 어휘 수준·문체 (기출 8문항 실측)
   기출에 실제로 쓰인 말들이다. 이 정도 수준·이 종류로 쓰라.
     (A)류  obstacle, insufficient, enables, fruitful, feature, controllability,
            challenges, disregard, predictability, manageability, resurface
     (B)류  surroundings, enrich, leaves, critical, transformation, challenge,
            secure, underserved, marginalized, variation, question
   · 라틴어원 학술 추상어. 파생명사(-ability, -ation, -ment)가 흔하다.
   · 구어·과장어 금지(a lot of, amazing, terrible).
   · 지문에 그대로 나온 단어는 정답으로 쓰지 마라 — 한 단계 추상화한 말이어야 한다
     (지문 'precisely controlled' → 정답 'controllability').

   ## 요약문 (기출 8문항 실측: 21~35단어, 평균 29)
   · 한 문장. (A)와 (B)가 **논리적으로 다른 역할**을 맡아야 한다.
       속성 → 결과      'The (A) of the production process … may (B) the assumption'
       원인 → 귀결      'Exploring one subject after another (A) remarkable work … which (B) room'
       양보 → 주장      'Although economic growth can be somewhat (A) …, direct approaches play a(n) (B) role'
       조건 → 기능      'may act as a(n) (A) to its adaptability when … changes arise in its (B)'
   · 두 빈칸이 같은 것을 반복하면 안 된다. (A)가 원인이면 (B)는 결과다.
   · 연결어(although / while / as / which)는 빈칸 밖에 남긴다 — 관계를 알려주는 단서다.

   ## 출력 직전 확인
   1. (A) 다섯 개 다 다른 단어인가? (B)도?
   2. 다섯 행을 전부 문장으로 만들어 읽었는가?
   3. 성립하는 행이 정확히 하나인가? **둘 이상이면 다시 짜라.**
   4. 정답 (A)(B)가 지문에 그대로 나온 단어는 아닌가?
   5. 유의어가 한쪽 칸에만 몰려 있는가? (양쪽 다 흐리면 복수정답이 된다)

2-2. **★★ Q3 와 Q4 는 완전히 다른 문항이다 — 요약문도 서로 달라야 한다**
   - Q3 = 객관식. 한 단어씩 두 칸을 뚫고 다섯 쌍 중 고르게 한다(추상어 판단).
   - Q4 = 영작. 구절 두 개를 뚫고 보기 단어로 원문을 복원하게 한다.
   - ★★★ **두 요약문이 같으면 안 된다.** Q3 를 푼 학생이 Q4 를 그냥 베껴 쓴다.
     실측 실패:
       Q3 'The brain's (A) deployment of alternative sensory pathways ensures (B) recognition
           despite environmental variability.'
       Q4 'The brain's (A) ensures (B).'        ← 같은 문장을 빈칸만 넓힌 것
     또 다른 실측:
       Q3 '... has resulted in territorial (B) driven by strategic calculation.'
       Q4 '... abandonment of land driven by rational pursuit of greater strategic ...'
                                    ↑ 'driven by ... strategic' 이 통째로 겹침
   - ★ 다른 각도로 잘라라:
       · 논리 구조를 바꾼다 (Q3 가 '속성→결과' 면 Q4 는 '조건→귀결' 이나 '대비')
       · 주어를 바꾼다 (Q3 가 'The brain's ...' 면 Q4 는 'When primary routes are blocked, ...')
       · 지문의 다른 문장을 축으로 삼는다 (Q3 결론부 → Q4 전환점)
   - ★ 두 요약문이 **연속 세 단어 이상 겹치면 거부된다.**
   
   Structure:
   - summary_template = "Strategic (A) of microclimate enables (B) of cultivation."  ← Q3 (short)
   - blank_summary_template = "<longer summary with literal (A) and (B) placeholders for full phrase writing>"  ← Q4 (long)
     ★★ blank_summary_template MUST contain the literal placeholders "(A)" and "(B)" — these are BLANKS the student fills in.
        DO NOT fill them in. DO NOT write the completed sentence. The (A) slot = where blank_A goes, the (B) slot = where blank_B goes.
        WRONG (no blanks): "Athletes must practice controlled breathing to manage arousal."
        RIGHT (has blanks): "Athletes must (A) to (B) during performance."
   - summary_options = 5 pairs of SINGLE WORDS  ← Q3 choices
   - ★★ APPROACH (우리가 정한 방식 — 완성문장 먼저, 빈칸은 코드가 뚫음): Output "full_summary" = ONE complete, natural, fully grammatical English sentence (NO blanks, NO (A)/(B)). Then choose two phrases that ALREADY appear verbatim inside full_summary and report them as blank_A and blank_B. The CODE will blank them out — you do NOT write blank_summary_template yourself. Your only job is: (1) a perfect full_summary sentence, (2) two exact substrings of it as blank_A/blank_B.
   - ★ Because the code blanks your own sentence, full_summary MUST be grammatically perfect on its own (check subject-verb agreement: "context and surroundings ALTER" not "alters"; plural subject → plural verb).
   - blank_A and blank_B MUST each be an EXACT substring of full_summary (copy them character-for-character from full_summary).
   - blank_A = the exact phrase removed from the (A) slot (~4-8 words)  ← Q4 writing answer
   - blank_B = the exact phrase removed from the (B) slot (~4-8 words)  ← Q4 writing answer
   - ★ When (A)/(B) are put back into blank_summary_template, the sentence MUST read as the natural sentence you first wrote — fully grammatical, no doubled words, no leftover prepositions.
   - ★★ NO PHRASE OVERLAP: blank_A / blank_B must NOT repeat words that already sit just before or just after their (A)/(B) slot in blank_summary_template. If the template reads "thereby (B) that would otherwise interfere with problem-solving", then blank_B MUST NOT also end with "that would otherwise interfere with problem-solving" — that doubles the phrase. The answer fills ONLY the gap, never text already printed around it.
   - ★★ If the (A)/(B) slot is right after "by / while / thereby / without / before / after", the answer MUST start with a GERUND (-ing), not a base verb. (template "thereby (B) ..." → blank_B "suppressing ..." ✅, not "suppress ..." ❌)
   - blank_summary_bogi = shuffled words from blank_A + blank_B  ← Q4 word bank

3. **Q4 BOGI MUST EQUAL blank_A + blank_B EXACTLY (word-by-word, case-insensitive)**:
   - Take all words from blank_A and blank_B
   - Lowercase them all
   - Shuffle the list
   - That's blank_summary_bogi
   - DO NOT add extra words. DO NOT remove words. Same exact word count.
   Example: blank_A="economic growth", blank_B="environmental cost"
   → blank_summary_bogi = ["cost", "growth", "economic", "environmental"] (4 words, shuffled, lowercase)

3-1. **Q4 blank_A and blank_B — natural key phrases (~4-8 words each)**:
   - blank_A is a meaningful phrase (~4-8 words, pick where it falls naturally — do not pad)
   - blank_B is a meaningful phrase (~4-8 words)
   - Pick FULL CLAUSES or extended phrases, not short noun phrases
   - Example GOOD: blank_A="strategically arranged south-facing terraces with stone walls" (7 words)
   - Example BAD: blank_A="south-facing slopes" (2 words - way too short!)

4. **Q5 BOGI MUST EQUAL topic_writing_answer EXACTLY (word-by-word, case-insensitive)**:
   - Take all words from topic_writing_answer (no punctuation)
   - Lowercase them all
   - Shuffle
   - That's topic_writing_bogi
   Example: topic_writing_answer="Effort drives success."
   → topic_writing_bogi = ["success", "drives", "effort"] (3 words, shuffled, lowercase, no period)

4-1. **Q5 topic_writing_answer — just write ONE natural topic sentence (like the 1회독 topic step)**:
   - Express the passage's core message as ONE complete, natural, grammatical English sentence, paraphrasing the key words with synonyms. Keep it concise (within ~20 words). That's it — write it the way you'd write a model topic sentence. The code shuffles it into the word bank afterward; do NOT arrange words to fit a bank or pad to a word count.
   - It just needs to be a normal, correct English sentence (proper subject + verb). Don't overthink it.
   - Example GOOD: "Self-comparison, rather than rivalry with others, is the only meaningful measure of genuine growth."
   - ★ Use DIFFERENT wording from the Q4 summary (paraphrase the key nouns/verbs; don't reuse the exact phrases of blank_A/blank_B).
   - ★ Two quick no-nos: don't end the sentence with a preposition ("...responses through" ❌); and after "Despite/In spite of" use a noun phrase, not a clause (use "Although ..." if you need a subject+verb).

5. **TOPIC WRITING (Q5)**: VARY sentence patterns. AVOID always starting with "What ~ is/does". Use diverse openings:
   - "Thanks to X, ..." / "People can ..." / "A model serves ..." / "By doing X, ..."
   - Direct declarative

6. **KOREAN EXPLANATIONS**: All *_explain fields in Korean.

7. **JSON ONLY**: No markdown, no extra text.

# OUTPUT FORMAT
{
  "id": "<id>",
  "title": "<short English title>",
  "given_sentence": "<sentence removed from passage>",
  "passage_with_marks": "<passage with <MARK1>..<MARK5> (or <MARK4>) placed ONLY at sentence boundaries, evenly spread>",
  "position_count": <4 or 5 — how many markers/choices this question has; 5 normally, 4 only if passage too short>,
  "position_correct": <0-based index into the markers (0..position_count-1) — which gap the given_sentence belongs at>,
  "position_explain": "<Korean explanation>",
  "topic_options": ["<5 TITLE options in English — 4-10 words each, capitalized, mix colon/question/gerund forms>"],
  "topic_correct": <0-4>,
  "summary_template": "<English summary with (A) and (B) placeholders>",
  "summary_template_kr": "<Korean translation of the summary sentence with correct (A)/(B) filled in — use PLAIN declarative style ~한다/~이다, NEVER honorific ~합니다/~입니다>",
  "summary_design": {
    "correct":    {"A": "<맞는 말>",   "B": "<맞는 말>"},
    "syn_A_1":    {"A": "<correct.A 의 유의어>", "B": "<틀린 말>",
                   "B_why": "<그 B 가 왜 틀렸는지 한 줄>"},
    "syn_A_2":    {"A": "<correct.A 의 다른 유의어>", "B": "<틀린 말>",
                   "B_why": "<그 B 가 왜 틀렸는지 한 줄>"},
    "syn_B":      {"A": "<틀린 말>", "B": "<correct.B 의 유의어>",
                   "A_why": "<그 A 가 왜 틀렸는지 한 줄>"},
    "both_wrong": {"A": "<틀린 말>", "B": "<틀린 말>",
                   "why": "<양쪽이 왜 틀렸는지 한 줄>"}
  },
  "full_summary": "<ONE complete grammatical summary sentence, NO blanks — the code blanks it>",
  "blank_summary_template": "<leave as the full sentence; code overwrites it with (A)/(B)>",
  "blank_summary_template_kr": "<Korean translation of the Q4 summary with blank_A/blank_B filled in — PLAIN style ~한다/~이다, NOT ~합니다>",
  "blank_summary_bogi": ["<lowercase shuffled words from blank_A + blank_B>"],
  "blank_A": "<exact phrase for (A)>",
  "blank_B": "<exact phrase for (B)>",
  "topic_writing_bogi": ["<lowercase shuffled words from topic_writing_answer>"],
  "topic_writing_answer": "<full topic sentence>",
  "topic_writing_kr": "<Korean translation of the topic sentence — PLAIN style ~한다/~이다, NOT honorific ~합니다>",
  "explain": "<Korean overall explanation>"
}

VERIFY BEFORE OUTPUT:
- blank_A is a natural key phrase (~4-8 words)
- blank_B is a natural key phrase (~4-8 words)
- topic_writing_answer is ONE natural, grammatical sentence (concise; do not pad to a word count)
- blank_summary_bogi has same word count as (blank_A words + blank_B words)
- topic_writing_bogi has same word count as topic_writing_answer (excluding punctuation)
- Same words (case-insensitive) appear in bogi and the answer
- ★ Every (A) and (B) in summary_options is EXACTLY ONE WORD (no spaces, no phrases!)
- ★ All five (A) values are different words; all five (B) values are different words.
- ★★ summary_check 5개를 반드시 채워라. 각 행을 요약문에 넣어 **완성된 문장**으로
  적고, 오답이면 어느 칸이 왜 틀렸는지 한 줄로 쓴다.
  머리로만 판단하지 말고 문장으로 써 봐야 복수정답이 드러난다.
- ★★ summary_check 각 항목은 **[역할]로 시작**한다. 다섯 개가 정확히 이것들이어야 한다:
  `[정답]` `[유의어-A]` `[유의어-A]` `[유의어-B]` `[둘다틀림]`
    예) "[유의어-A] ... reveal ... efficiency → (A)는 맞지만 (B) efficiency 가 지문과 무관"
        "[정답] ... concentration ... output → 정답"
- ★★ summary_check 의 [역할] 다섯 개를 세어라.
  **[정답]1 [유의어-A]2 [유의어-B]1 [둘다틀림]1** 이어야 한다.
  [유의어-A]가 셋 이상이면 (A) 칸이 통째로 흐려져 복수정답이 된다. 다시 짜라.
- ★★ summary_check 를 다 쓴 뒤 세어라. **성립하는 행이 정확히 하나인가?**
  둘 이상이면 그 행의 한쪽 칸을 갈고 처음부터 다시 검증하라.
  (실측 실패: disclosure/engagement · revelation/curiosity · presentation/attention
   셋이 다 성립했다 — (A)(B) 양쪽에 유의어를 둔 탓이다)
- ★ 유의어는 **한쪽 칸에만** 둔다. 그 행들의 반대쪽 칸은 반드시 틀리게 채운다.

Return ONLY the JSON object."""


# ===================== JSON 추출 헬퍼 =====================
import re
import json


def _repair_quotes_in_json_strings(text: str) -> str:
    """JSON 문자열 값 안의 escape 안 된 따옴표를 자동 escape
    예: "key": "He said "hello" loudly" → "key": "He said \"hello\" loudly"
    
    알고리즘 (단순화):
    1. 콜론 이후 첫 따옴표가 문자열 시작
    2. 그 다음 따옴표가 끝일 가능성. 단 다음 글자가 (콤마/공백+key) 또는 (}, ]) 아니면 중간 따옴표
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            # 문자열 시작
            out.append(c)
            i += 1
            while i < n:
                c = text[i]
                if c == '\\':
                    # 이미 escape됨
                    out.append(c)
                    if i + 1 < n:
                        out.append(text[i+1])
                        i += 2
                    else:
                        i += 1
                elif c == '"':
                    # 다음에 오는 글자 중 의미있는 첫 글자가 뭔지 확인
                    j = i + 1
                    while j < n and text[j] in ' \t\n\r':
                        j += 1
                    if j < n and text[j] in ',}]:':
                        # 문자열 끝
                        out.append(c)
                        i += 1
                        break
                    else:
                        # escape되지 않은 중간 따옴표 → escape 해서 출력
                        out.append('\\"')
                        i += 1
                else:
                    out.append(c)
                    i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_json_from_response(text: str) -> dict:
    """Claude 응답에서 JSON 객체 추출 (따옴표 자동 복구 포함)"""
    text = text.strip()
    # 마크다운 코드 펜스 제거
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    
    # JSON 객체 찾기
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # 첫 { 부터 마지막 } 까지
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text = text[start:end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            # 따옴표 escape 자동 복구 시도
            try:
                repaired = _repair_quotes_in_json_strings(json_text)
                result = json.loads(repaired)
                print(f"[JSON 복구 성공] 따옴표 escape 자동 처리됨")
                return result
            except json.JSONDecodeError as e2:
                # 자동 복구도 실패
                err_msg = (
                    f"JSON 파싱 실패 ({e.msg}, line {e.lineno} col {e.colno}): "
                    f"문자열 값 안에 escape 안 된 큰따옴표(\") 또는 특수문자가 있을 가능성 높음. "
                    f"다음 재시도 시: 원문에 큰따옴표가 있으면 작은따옴표(')로 바꿔서 출력할 것. "
                    f"예: \"cheated God\" → 'cheated God'"
                )
                raise ValueError(err_msg)
    
    raise ValueError(f"JSON 객체를 찾을 수 없음. 응답 일부: {text[:300]}")


# ════════════════════════════════════════════════════════════════
# Q5 주제문 전용 프롬프트 (1회독 step4_topic 방식 — 주제문만 집중 생성)
#   변형 Q1~Q5를 한 번에 만들면 주제문 한 문장에 집중이 안 돼 수일치 등 실수가 난다.
#   그래서 주제문만 따로, 단독으로 한 번 더 생성한다(1회독처럼).
# ════════════════════════════════════════════════════════════════
TOPIC_SENTENCE_SYS = (
    "You are an English exam content generator for Korean high school students. "
    "All English you write MUST be grammatically correct (check subject-verb agreement, tense, prepositions). Output ONLY valid JSON, no markdown, no text outside JSON."
)

def build_topic_sentence_prompt(passage_text: str) -> str:
    return (
        "Write ONE topic sentence that captures the core message of the passage below.\n\n"
        "[PASSAGE]\n" + passage_text + "\n\n"
        "[RULES]\n"
        "- ★★ METHOD (do this to stay grammatical): FIRST pick the single sentence in the passage that best "
        "states the main point. Then KEEP ITS GRAMMATICAL SKELETON — the same subject-verb relationship, the "
        "same clause/phrase structure (prepositional phrases, appositives, relative clauses, parallel form) — "
        "and ONLY swap the content words for synonyms. You are re-dressing an existing correct sentence, NOT "
        "building a new structure from scratch. This keeps subject-verb agreement, tense, and prepositions "
        "automatically correct.\n"
        "- ONE complete, natural, fully grammatical English sentence. Do NOT copy the sentence verbatim "
        "(change the content words), but DO mirror its structure.\n"
        "- Concise: about 12-20 words.\n"
        "- ★ Keep verbatim ONLY (1) genuine proper nouns (people/places/organizations/titles, e.g. 'John Dewey') and (2) fixed scientific/technical terms with no natural synonym (e.g. 'thyroid gland', 'photosynthesis'). Everything else is paraphrased normally — ordinary phrases like 'social and political protest' are NOT protected, reword them. When unsure, paraphrase.\n"
        "- ★ Still double-check SUBJECT-VERB AGREEMENT after swapping words: a plural/compound subject "
        "(X and Y) takes a plural verb ('context and surroundings ALTER ...' not 'alters').\n"
        "- No bare-verb subject; do not end with a preposition; no 'Despite + clause'; no 'modal + adjective'.\n"
        "- This is the ONLY thing you are writing now — focus entirely on making this one sentence perfect.\n\n"
        "- ★ Also give the Korean translation of THAT SAME sentence (topic_sentence_kr). "
        "It must be a translation of the sentence you just wrote — not of the passage, not of anything else. "
        "PLAIN declarative style (~한다/~이다), never honorific (~합니다).\n\n"
        "[OUTPUT JSON]\n"
        '{"topic_sentence": "<one perfect topic sentence>", '
        '"topic_sentence_kr": "<Korean translation of that exact sentence>"}'
    )


# ════════════════════════════════════════════════════════════════
# Q4 요약영작 전용 프롬프트 (1회독 방식 — 요약문 하나에만 집중 생성)
#   Q1~Q5를 한 번에 만들면 요약문에 집중 안 돼 수일치·구조 실수가 난다.
#   요약문(full_summary)만 따로 생성 → 코드가 그 안의 두 구절을 빈칸으로 뚫는다.
# ════════════════════════════════════════════════════════════════
SUMMARY_SENTENCE_SYS = (
    "You are an English exam content generator for Korean high school students. "
    "All English you write MUST be grammatically correct (check subject-verb agreement, tense, prepositions). Output ONLY valid JSON, no markdown, no text outside JSON."
)

def build_summary_sentence_prompt(passage_text: str, q3_summary: str = "") -> str:
    """B Q4 — 요약문 한 문장 + 빈칸 두 자리 지목.

    ★ 핵심 원리 (26-1 기말 학교 기출 4문항을 지문까지 읽고 도출):
      요약문은 지문의 '논리 축'을 옮겨 담고, 빈칸은 그 축의 양쪽을 뚫는다.
      형태를 흉내내는 게 아니라 지문 구조를 옮기는 것이다.

      중흥고3: 지문이 'By contrast, artworks in exhibitions feel isolated' 로 꺾인다
               → 요약문 'Studios and homes (A) whereas (B) that context.'
               → 빈칸 둘이 대비의 양쪽
      소명여고3: 지문이 'Yet listening audiences are rarely aware' 로 꺾인다
               → 요약문 'Although directors are essential, ... appreciated (A) so as to ...'
               → 빈칸이 반전의 내용
      도당고3(우울증): 지문이 'This is because depression is not one disease' 로 이유를 댄다
               → 요약문 'Depression (A), so the complexity of (B).'
               → 빈칸 둘이 원인과 귀결
    """
    _avoid = ""
    if str(q3_summary or "").strip():
        _avoid = (
            "\n\n## ★★★ 이미 Q3 가 쓴 요약문 — 이것과 겹치면 안 된다\n"
            "[Q3 요약문]\n" + str(q3_summary).strip() + "\n\n"
            "Q3(객관식 요약빈칸)와 Q4(요약영작)는 **다른 문항**이다. 같은 지문을 쓰지만\n"
            "학생에게 요구하는 것이 전혀 다르다.\n"
            "  Q3 — 논리 축의 두 지점을 **한 단어씩** 뚫고 다섯 쌍 중 고르게 한다(추상어 판단)\n"
            "  Q4 — 논리 축의 양쪽을 **구절로** 뚫고 보기 단어로 복원하게 한다(영작)\n"
            "★ 같은 문장을 쓰면 Q3 를 푼 학생이 Q4 를 그냥 베껴 쓴다. 실측 실패 3건이다:\n"
            "  (1) 같은 문장을 재사용\n"
            "      Q3 'The brain's (A) deployment of alternative sensory pathways ensures (B) recognition…'\n"
            "      Q4 'The brain's (A) ensures (B).'\n"
            "  (2) 같은 자리에서 출발\n"
            "      Q3 'Conflicting (A) over border demarcation has resulted in territorial (B)…'\n"
            "      Q4 'Conflicting claims have forced (A) of land driven by (B)…'\n"
            "  (3) ★ 어구는 달라도 **같은 각도** — 이게 제일 놓치기 쉽다\n"
            "      Q3 'Unlike academic convention, online writing demands upfront (A) of the conclusion\n"
            "          to retain reader (B).'\n"
            "      Q4 'Online writing requires (A) to (B) throughout the narrative.'\n"
            "      두 문장이 **똑같은 말을 하고 있다** — '온라인 글은 결론을 먼저 내야 독자를 붙든다'.\n"
            "      어구가 안 겹쳐도 이러면 실패다. 두 문장을 나란히 읽어 보고\n"
            "      **같은 주장을 하고 있으면 다시 써라.**\n"
            "\n"
            "### 그러니 **다른 각도로 잘라라**\n"
            "  · Q3 가 '속성 → 결과' 로 썼으면 Q4 는 '조건 → 귀결' 이나 '대비' 로 쓴다.\n"
            "  · 주어를 바꾼다. Q3 가 'The brain's ...' 로 시작했으면 Q4 는\n"
            "    'When primary routes are blocked, ...' 처럼 다른 데서 시작한다.\n"
            "  · 지문의 **다른 문장**을 축으로 삼는다. Q3 가 결론부를 썼으면 Q4 는 전환점을,\n"
            "    Q3 가 전환점을 썼으면 Q4 는 결론부를 축으로 삼는다.\n"
            "★ 위 Q3 요약문과 **연속 세 단어 이상 겹치면 거부된다.** 어구를 통째로 옮기지 마라.\n"
            "  (관사·전치사 같은 기능어 연쇄는 어쩔 수 없지만, 내용어 조합은 겹치면 안 된다)\n")
    return (
        "Write ONE summary sentence of this passage, then mark TWO blanks in it.\n\n"
        "[PASSAGE]\n" + passage_text + _avoid + "\n\n"

        "## ★★ 네 몫과 코드 몫\n"
        "  네 몫  · 지문의 논리 축(대비·인과·양보)을 찾아 요약문 한 문장으로 옮기는 것\n"
        "         · 그 축의 양쪽 어디를 뚫을지 정하는 것\n"
        "         · 요약문을 문법적으로 완전하게 쓰는 것 (코드가 그 문장을 뚫는다)\n"
        "  코드 몫 · 지목한 구절이 요약문에 그대로 있는가, 되넣으면 복원되는가\n"
        "         · 두 빈칸이 겹치지 않는가, 사이에 연결어가 남는가\n"
        "         · 보기(word bank) 생성 — 네가 만들 필요 없다\n"
        "★ 요약문은 네가 방금 쓴 문장이다. 지목할 때 **다시 타이핑하지 말고**\n"
        "  그 문장에서 글자 그대로 옮겨라. 한 단어만 바뀌어도 코드가 못 찾는다.\n\n"

        "## STEP 1 — 지문에서 논리가 꺾이는 지점을 찾아라 (여기서 시작한다)\n"
        "Read the passage and locate the sentence where the argument TURNS or CONCLUDES.\n"
        "It is usually marked by one of these:\n"
        "    By contrast / However / Yet / Instead / On the other hand   (대비·반전)\n"
        "    This is because / since / due to                            (이유)\n"
        "    Thus / Therefore / As a result / so                         (귀결)\n"
        "    Although / While / Even if                                  (양보)\n"
        "★ 그 지점이 글의 축이다. 요약문은 그 축을 그대로 옮겨 담아야 한다.\n"
        "  지문에 없는 관계를 억지로 만들지 마라 — 지문이 대비 구조면 요약문도 대비,\n"
        "  지문이 인과면 요약문도 인과다.\n\n"

        "## STEP 2 — 그 축을 요약문의 뼈대로 삼아라\n"
        "  지문 'By contrast, artworks displayed in exhibitions feel isolated, clipped from\n"
        "        the world where they originated'\n"
        "    → 요약문 'Studios and homes ___ whereas ___ that context.'\n"
        "  지문 'Yet listening audiences are rarely aware of any musical leadership,\n"
        "        and for good aesthetic reason'\n"
        "    → 요약문 'Although music directors are essential, the stage work can be best\n"
        "              appreciated ___ so as to fully preserve the aesthetic experience.'\n"
        "  지문 'This is because depression is not one disease; it is a collection of symptoms\n"
        "        ... resulting from a melting pot of diverse causes'\n"
        "    → 요약문 'Depression ___, so the complexity of ___.'\n\n"

        "## STEP 3 — 축의 양쪽을 빈칸으로 뚫어라\n"
        "  · 연결어(whereas / so / making / Although) 자체와 그 앞뒤 구두점은 빈칸 밖에 남긴다.\n"
        "    ★ 그것이 학생에게 두 빈칸의 관계를 알려주는 유일한 단서다.\n"
        "  · 연결어 앞쪽에서 [전제·원인·한쪽]을 담는 구 → blank_A\n"
        "  · 연결어 뒤쪽에서 [귀결·결과·반대쪽]을 담는 구 → blank_B\n"
        "  · 두 빈칸을 지우고 읽어보라. 남은 뼈대만으로 '무엇과 무엇이 어떤 관계인지'가\n"
        "    드러나야 한다. 안 드러나면 빈칸을 좁혀라.\n\n"
        "  기출 실제 정답 — 이 정도 크기·성격이다:\n"
        "    'Depression (A), so the complexity of (B).'\n"
        "       (A) has diverse causes and symptoms                          [5단어]\n"
        "       (B) the condition makes a single solution impossible to find [9단어]\n"
        "    'scientific facts emerge (A) of observations ..., making (B) extremely difficult.'\n"
        "       (A) a disordered array                                       [3단어]\n"
        "       (B) assigning credit and dates                               [4단어]\n"
        "    'Studios and homes (A) whereas (B) that context.'\n"
        "       (A) connect artworks to their context                        [5단어]\n"
        "       (B) exhibitions isolate them in the absence of               [7단어]\n"
        "  → (A)(B) 크기가 서로 달라도 된다. 논리 단위가 끝나는 곳에서 끊는다.\n\n"

        "  [X] 주어부만 뚫기 — 'Online writing requires' 는 논리를 담지 않는다\n"
        "  [X] 두 빈칸이 같은 것을 반복 — (A)가 원인이면 (B)는 결과여야 한다\n"
        "  [X] 연결어를 빈칸 안에 넣기 — so 를 물면 관계를 알 단서가 사라진다\n"
        "  [X] 빈칸이 문장 대부분을 차지 — 뼈대가 없으면 문맥으로 풀 수 없다\n\n"

        "## 요약문 작성 규칙\n"
        "- ★ LENGTH: 25-32 words with both blanks filled in. 학교 기출 실측 18~31, 평균 24.\n"
        "  15단어짜리 문장은 빈칸 둘을 넣으면 뼈대가 사라진다.\n"
        "- ★★ METHOD: base your sentence on the passage's own wording where possible.\n"
        "  Paraphrase the content words but keep the logical skeleton the passage already has.\n"
        "- Third person, present tense, declarative. No 'This passage argues that ...'.\n"
        "- Subject-verb agreement: a compound subject (X and Y) takes a plural verb.\n"
        "- No bare-verb subject; do not end with a preposition; no 'Despite + clause';\n"
        "  no 'modal + adjective'.\n"
        "- ★ Also give the Korean translation of THAT SAME sentence (full_summary_kr).\n"
        "  PLAIN declarative style (~한다/~이다), never honorific (~합니다).\n\n"

        "## 빈칸 지목 방식 — 문장을 다시 쓰지 말고 위치를 가리켜라\n"
        "Students rebuild your exact wording from a shuffled word bank, so you must point at\n"
        "a stretch of the sentence you just wrote. ★ Do NOT retype it — retyping invites\n"
        "paraphrase, and one changed word makes the cut fail.\n"
        "  · starts_with : the FIRST TWO OR THREE WORDS of the span, copied exactly\n"
        "  · ends_with   : the LAST TWO OR THREE WORDS of the span, copied exactly\n\n"

        "### 조건 — 어기면 그 항목이 버려진다\n"
        "  · ★ 각 span 은 3~9 단어. 학교 기출 실측 3~9, 평균 5.9.\n"
        "    ★ 세어라. 상한(9)은 논리 단위 하나만 잡으면 저절로 지켜지지만,\n"
        "      하한(3)은 성분을 끝까지 안 잡으면 그냥 미달한다 — 그건 세야만 안다.\n"
        "    5단어 이상이 기출의 다수다.\n"
        "  · ★ blank_A 가 blank_B 보다 문장에서 먼저 오고, 사이에 연결어가 있어야 한다.\n"
        "  · ★ span 이 구두점으로 시작하거나 끝나면 REJECTED. 구두점 앞에서 끊어라.\n"
        "    (span 안쪽의 쉼표는 괜찮다 — 보기가 'signals,' 처럼 앞 단어에 붙여 제시한다)\n"
        "  · ★ 경계는 시작과 끝의 기준이 다르다 (수능 빈칸 23개 실측).\n"
        "    시작 — 거의 안 가린다. 관사·전치사·be동사로 시작해도 된다(기출 30%가 그렇다).\n"
        "           막는 것은 앞 절과 이어붙는 접속사·관계사뿐이다(and/but/as/which/that).\n"
        "    끝   — 엄격하다. 기능어로 끝나는 기출 빈칸은 0개다.\n"
        "           반드시 내용어로 끝나라. [X] 'of the' [X] 'required for'\n"
        "  · ★★ 하이픈 복합어로 끝내지 마라 — 뒤에 꾸밀 명사가 따라온다.\n"
        "      [X] a long-term   [X] the twenty-second   [X] a well-known\n"
        "      [O] a long-term effect   [O] the twenty-second parallel\n"
        "  · ★★ 한 덩어리로 쓰이는 표현을 중간에서 끊지 마라.\n"
        "    실측 실패: 'demand (A) rather (B)' — 'rather' 를 밖에 두고 'than' 부터 뚫었다.\n"
        "      [X] (B) = 'than delayed revelation'   → 'rather' 와 'than' 이 갈라진다\n"
        "      [O] (B) = 'rather than delayed revelation'  또는  (B) = 'delayed revelation'\n"
        "    같은 종류: 'not only A but B', 'so as to', 'in order to', 'as well as',\n"
        "      'over the course of', 'in terms of', 'a number of', 'as long as'.\n"
        "    끊고 나서 **빈칸 밖에 남는 부분만 읽어 보라.** 'demand … rather … to ensure'\n"
        "    처럼 홀로 뜨는 조각이 생기면 잘못 끊은 것이다.\n"
        "  · starts_with 가 문장 안에서 한 번만 나오는 표현이어야 한다.\n\n"

        "### 출력 직전 확인\n"
        "  1. 지문의 논리 축(대비/인과/양보)이 요약문 연결어로 옮겨졌는가?\n"
        "  2. starts_with / ends_with 가 full_summary 안에 글자 그대로 있는가?\n"
        "  3. 각 span 이 3~9 단어인가?\n"
        "  4. 연결어가 두 빈칸 사이에 남아 있는가?\n"
        "  5. 두 span 을 빼고 읽었을 때 관계가 드러나는가?\n"
        "     ★ 그때 'rather' 'not only' 'so as' 처럼 짝을 잃고 홀로 뜨는 조각이 없는가?\n"
        "  6. ★★ Q3 요약문이 위에 주어졌다면 — 두 문장을 나란히 놓고 읽어라.\n"
        "     · 연속 세 단어 이상 겹치는 데가 있는가?\n"
        "     · 같은 단어로 시작하는가?\n"
        "     · ★ 어구가 안 겹쳐도 **같은 주장을 하고 있는가?**\n"
        "       (이건 코드가 못 잡는다. 네가 봐야 한다)\n"
        "     셋 중 하나라도 해당하면 논리축·주어·축이 되는 문장을 바꿔 다시 써라.\n\n"

        "[OUTPUT JSON]\n"
        '{"full_summary": "<one summary sentence, 25-32 words>", '
        '"full_summary_kr": "<Korean translation of that exact sentence>", '
        '"blank_A": {"starts_with": "<exact first 2-3 words>", "ends_with": "<exact last 2-3 words>"}, '
        '"blank_B": {"starts_with": "<exact first 2-3 words>", "ends_with": "<exact last 2-3 words>"}}'
    )


# ════════════════════════════════════════════════════════════════
# Q3 핵심빈칸 전용 프롬프트 (유형 A — 첫 문장만 주고 집중 생성)
#   첫 문장(intro)은 코드가 확정한다. 그 문장에서 핵심 구절 1개를 골라
#   빈칸으로 만들고, 같은 문법구조의 패러프레이즈를 정답으로 만든다.
#   "원문이 절이면 정답도 절" 규칙을 강제해 빈칸 문법 불일치를 막는다.
# ════════════════════════════════════════════════════════════════
# ★★ 폐기됨 (_s96) — A Q3의 유일한 유형은 어휘(수능 30번)다.
#   핵심빈칸은 _s58에서 어휘로 전환하며 '예비'로 남겨둔 옛 유형인데, 예비가 실패하면
#   validate_a 가 CRITICAL 을 쏴서 1·2·4·5번이 멀쩡한 A 가 통째로 죽었다(16강 3/3 누락).
#   게다가 '첫 문장 안에서만' 뚫어야 해 정식보다 훨씬 좁았다
#   ('Bir Tawil is a strange place.' 6단어면 아예 불가능).
#   generator 는 더 이상 이것을 import 하지 않는다. 호환을 위해 정의만 남겨둔다.
CORE_BLANK_SYS = (
    "You are an English exam content generator for Korean high school students. "
    "All English you write MUST be grammatically correct (check subject-verb agreement, tense, prepositions). Output ONLY valid JSON, no markdown, no text outside JSON."
)

def build_core_blank_prompt(first_sentence: str) -> str:
    return (
        "Make ONE blank-inference question from the FIRST SENTENCE below.\n\n"
        "[FIRST SENTENCE]\n" + first_sentence + "\n\n"
        "[RULES]\n"
        "- core_blank_target = an EXACT substring of the first sentence (copy it character-for-character) "
        "that the whole point hinges on. At least 3 words.\n"
        "- It must NOT include any word that sits right before/after it (so that blanking it leaves the "
        "rest of the sentence reading exactly like the original — no duplicated word around the blank).\n"
        "- The CORRECT option is a PARAPHRASE of core_blank_target (synonyms / figurative rewording, NOT a copy).\n"
        "- ★ Keep verbatim ONLY (1) genuine proper nouns (people/places/organizations/titles, e.g. 'John Dewey') and (2) fixed scientific/technical terms with no natural synonym (e.g. 'thyroid gland', 'photosynthesis'). Everything else is paraphrased normally — ordinary phrases like 'social and political protest' are NOT protected, reword them. When unsure, paraphrase.\n"
        "- ★★ GRAMMATICAL FIT IS MANDATORY: the correct option MUST keep the SAME grammatical structure as "
        "core_blank_target so the sentence stays grammatical when inserted.\n"
        "   · If the blanked phrase is a CLAUSE (has a subject + finite verb, e.g. 'food and nutrition play the "
        "greatest role'), the paraphrase MUST also be a clause (e.g. 'food and nutrition have the most "
        "significant impact'). NEVER turn a clause into a bare noun phrase.\n"
        "   · If it is a noun phrase, keep a noun phrase.\n"
        "   · TEST: read the full first sentence with the correct option in the blank — if it is not "
        "grammatical, rewrite the option.\n"
        "- Provide 5 options total (index 0-4). The 4 wrong options = opposite meaning or content not stated; "
        "each grammatically fits the blank; not near-duplicates.\n"
        "- This is the ONLY thing you are writing now — focus entirely on this one blank question.\n\n"
        "[OUTPUT JSON]\n"
        '{"core_blank_target": "<exact substring of the first sentence, >=3 words>", '
        '"core_blank_options": ["<opt0>", "<opt1>", "<opt2>", "<opt3>", "<opt4>"], '
        '"core_blank_correct": <0-4>, '
        '"core_blank_explain": "<Korean one-line explanation>"}'
    )

# ════════════════════════════════════════════════════════════════
# 한글 해석 전용 프롬프트 (폴백)
#   Q4 요약문·Q5 주제문은 영문을 2차로 재생성해 덮어쓴다. 그때 한글도 함께 받아야
#   답지의 해석과 정답 영문이 같은 문장이 된다. LLM이 kr 키를 누락하면 이 프롬프트로
#   번역만 따로 한 번 더 부른다. (렌더러가 kr이 비면 해석 줄 자체를 안 찍는다)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# B Q3 복수정답 검증 — 별도 호출 (_s121)
#   지금까지의 검사(summary_check, 역할 라벨)는 전부 **만든 LLM 에게 되묻는** 방식이라
#   대충 채우면 통과했다. 시험을 낸 사람에게 "이 문제 복수정답 아니죠?" 하고 묻는 셈이다.
#   → 문항만 떼어 **다른 호출로 풀린다.** 만든 맥락을 모르므로 학생 입장에 가깝다.
#     답이 정답과 다르거나 "둘 이상 성립"이라고 하면 복수정답이다.
# ════════════════════════════════════════════════════════════════
SOLVE_SYS = (
    "You are a Korean high school student taking an English exam. "
    "Solve the given 요약문 빈칸 문제 honestly. "
    "You have NOT seen this problem before and do not know the intended answer. "
    "Return ONLY JSON."
)


def build_solve_prompt(passage_en: str, summary_template: str, options) -> str:
    """B Q3 문항을 '푸는' 프롬프트. 정답을 알려주지 않는다."""
    _opts = "\n".join(
        f"  {'①②③④⑤'[k]} (A) {o[0]}  …  (B) {o[1]}"
        for k, o in enumerate(options) if isinstance(o, (list, tuple)) and len(o) >= 2)
    return (
        "다음 지문을 읽고 요약문의 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것을 고르시오.\n\n"
        "[지문]\n" + str(passage_en or "").strip() + "\n\n"
        "[요약문]\n" + str(summary_template or "").strip() + "\n\n"
        "[선지]\n" + _opts + "\n\n"
        "## 어떻게 답하나\n"
        "1. 다섯 선지를 **하나씩 요약문에 넣어 읽어라.** 머리로 넘기지 마라.\n"
        "2. 각 선지가 성립하는지 판정한다. (A)와 (B)가 **둘 다** 맞아야 성립이다.\n"
        "3. 성립하는 선지 번호를 전부 적는다.\n"
        "\n"
        "★ 정직하게 답하라. 출제자가 의도한 답을 맞히는 게 목적이 아니다.\n"
        "  **문제가 잘못 만들어졌는지 찾는 것**이 목적이다.\n"
        "  두 개 이상 성립하면 두 개 이상 적어라. 억지로 하나로 좁히지 마라.\n"
        "\n"
        "[OUTPUT JSON]\n"
        '{"per_option": [\n'
        '   {"n": 1, "filled": "<①번을 넣은 완성 문장>", "ok": true/false,\n'
        '    "why": "<성립하면 왜 맞는지, 아니면 어느 칸이 왜 틀렸는지>"},\n'
        '   {"n": 2, ...}, {"n": 3, ...}, {"n": 4, ...}, {"n": 5, ...}\n'
        '],\n'
        ' "valid": [<성립하는 번호들. 예: [2] 또는 [2,4]>]}'
    )


TRANSLATE_SYS = (
    "You translate one English sentence into natural Korean. "
    "Output ONLY valid JSON, no markdown, no text outside JSON."
)


def build_translate_prompt(en_sentence: str) -> str:
    return (
        "Translate the sentence below into natural Korean.\n\n"
        "[SENTENCE]\n" + en_sentence + "\n\n"
        "[RULES]\n"
        "- Translate THIS sentence only — do not summarize, do not add, do not omit.\n"
        "- PLAIN declarative style (~한다/~이다), never honorific (~합니다/~입니다).\n"
        "- Natural Korean that a high school student reads easily.\n\n"
        "[OUTPUT JSON]\n"
        '{"kr": "<Korean translation>"}'
    )

# ════════════════════════════════════════════════════════════════
# A Q3 — 어휘 유형 (수능 30번)
#   기출 7세트 35개 밑줄 실측:
#     · 첫 문장에 밑줄 0/7
#     · 밑줄 위치 평균 61% 지점 — ★ 원문 순서 지문 기준. 우리 A 는 (A)(B)(C)가
#       셔플돼 앞뒤 개념이 없으므로 위치 가중치를 쓰지 않는다(_s97).
#     · 정답 위치 ③2 ④3 ⑤2 (원문 순서 지문 기준. 우리는 셔플돼 있어 ①~⑤ 전부 쓴다)
#     · 밑줄 품사 형용사37% 동사37% 명사17% 부사5%
#     · 정답 품사 형용사4 동사3 — ★ 명사도 정답이 된다(_s97). 2015 수능 30번 ⑤ 'concern',
#       연성 하천공학 'modesty', 바자르 'restrictions'. 기준은 품사가 아니라 방향이다.
# ════════════════════════════════════════════════════════════════
VOCAB_SYS = (
    "You write CSAT-style vocabulary-in-context questions (수능 30번). "
    "Output ONLY valid JSON, no markdown, no text outside JSON."
)


def build_vocab_prompt(paragraphs, blank_phrases=None, want_n: int = 0) -> str:
    """paragraphs: [[label, text], ...] 원문 그대로
       blank_phrases: Q5 빈칸으로 이미 쓰인 구절들 (겹치면 안 됨)
       want_n: 정답을 놓을 자리(3/4/5). 0이면 기존처럼 LLM에게 맡긴다.

    ★ want_n 이 왜 필요한가 — '3, 4, 5 중에서 골라라'라고만 하면 LLM은 매번 3을
      고른다(실측 3/3 전부 ③, _s63에서도 같은 관측). 나열의 첫 항목이고 출력 예시
      JSON도 n:3 을 정답으로 보여주기 때문이다. 코드로 뒤에서 옮기는 것은 불가능하다:
      original/shown 을 교환하면 원문 인덱스와 어긋나 validate_vocab 이 거부하고,
      is_answer 만 옮기면 반의어가 박힌 자리가 오답이 되어 문항이 깨진다(_s68).
      → 자리를 코드가 정해서 내려보내고, 그 자리의 반의어는 LLM이 만든다."""
    # ★★ Q5 빈칸 자리를 지문 안에서 눈에 보이게 표시한다 (_s103).
    #   옛 코드는 원본처럼 보이는 지문에 <BLANK_A> 마커만 박아 두고, 아래에
    #   'Q5 가 쓴 구절' 목록을 따로 보여줬다. 그 목록이 오히려 그 단어들을 눈앞에
    #   갖다놓는 꼴이라 LLM 이 거기서 골랐다(실측: 'relies' 'flexibility' 'available,'
    #   전부 Q5 빈칸 안 단어). 목록은 없애고, 지문에서 그 자리를 못 고르게 막는다.
    def _mask(txt):
        t = str(txt or "")
        t = t.replace("<BLANK_A>", "[[[여기는 Q5 빈칸 — 이 안의 단어는 이미 사라졌다]]]")
        t = t.replace("<BLANK_B>", "[[[여기는 Q5 빈칸 — 이 안의 단어는 이미 사라졌다]]]")
        return t
    body = "\n\n".join(f"({lab}) {_mask(txt)}" for lab, txt in paragraphs)
    if want_n in (1, 2, 3, 4, 5):
        _want_txt = (
            f"  · ★★ 정답은 반드시 {want_n}번 자리다. 다른 번호에 두면 그 항목은 버려진다.\n"
            f"    {want_n}번 자리의 단어를 반의어로 바꾸고, 나머지 네 자리는 동의어로 바꿔라.\n"
            f"    자리를 먼저 정해 놓고 고르는 것이다 — {want_n}번에 놓을 만한 단어를 찾아\n"
            f"    그 문장을 밑줄 자리로 잡아라. 반대말을 못 대는 단어면 다른 문장을 쓴다.\n"
            f"  · ★ 기출은 정답이 ③④⑤에만 있지만 그 규칙은 여기 적용되지 않는다.\n"
            f"    기출은 지문이 원문 순서 그대로라 '앞쪽이 논지를 확인시키고 뒤에서 뒤집는'\n"
            f"    구조가 성립한다. 이 지문은 (A)(B)(C)가 셔플돼 학생이 보는 순서와\n"
            f"    원문 순서가 다르다. 앞뒤 개념이 없으므로 ①~⑤ 어디든 정답이 될 수 있다.\n")
    else:
        _want_txt = (
            "  · 정답은 ①~⑤ 어디든 될 수 있다. 지문이 셔플돼 있어 앞뒤 개념이 없다.\n")
    # ★ blank_phrases 는 더 이상 프롬프트에 싣지 않는다 (_s103).
    #   보여주면 LLM 이 그 안에서 고른다. 자리는 위 지문의 [[[...]]] 표시로 충분하다.
    avoid = ("\n\n★★ 지문에 [[[여기는 Q5 빈칸 …]]] 이라고 적힌 자리가 있다.\n"
             "  그 자리의 단어들은 **이미 다른 문항이 가져가서 학생 시험지에 안 보인다.**\n"
             "  거기서 고르면 밑줄 칠 단어가 없어 그 항목은 버려진다.\n"
             "  [[[...]]] 밖에 **실제로 인쇄돼 있는 단어** 중에서만 다섯을 골라라.\n")
    return (
        "Read the passage and design a 수능 30번 vocabulary question.\n\n"
        "════════════════════════════════════════════════════\n"
        "★★★ 가장 먼저 읽어라 — 다섯 항목 **전부** `antonym` 칸을 채워야 한다.\n"
        "   하나라도 비면 그 문항은 통째로 버려진다. 실측: 세 지문 연속 전멸했다.\n"
        "   antonym = 그 단어(original)의 **반대말 한 단어**.\n"
        "     exciting → dull      convince → dissuade    rely → disregard\n"
        "     adapted  → unsuited  internal → external    greater → lesser\n"
        "   ★ 오답 자리(4개)도 적는다. 시험지에 나가지 않는다 —\n"
        "     '이 자리가 반대말을 댈 수 있는 자리인가'를 확인하는 용도다.\n"
        "   ★★ 적는 것으로 끝이 아니다. **그 반대말을 그 자리에 넣어 읽어라.**\n"
        "     문장이 성립해야 진짜 반대말이다. 안 되면 그 자리를 쓰면 안 된다.\n"
        "       지문 'plays a central role in learning'\n"
        "         role 의 반대말? — 억지로 'non-role' 이라 적어도\n"
        "         'plays a central non-role in learning' 은 말이 안 된다 → 못 쓴다\n"
        "       지문 'must face the consequences'\n"
        "         face 의 반대말? 'avoid' — 'must avoid the consequences' 는 되지만\n"
        "         이건 방향이 뒤집힌 게 아니라 다른 얘기다 → 애매하면 쓰지 마라\n"
        "       지문 'a significant increase'\n"
        "         significant → trivial — 'a trivial increase' 성립 ✓ 방향도 뒤집힘 ✓\n"
        "     ★ 시험: 반대말을 넣은 문장이 **원문과 정반대 주장**이 되는가?\n"
        "       그냥 '다른 말'이 되면 안 된다. 반대여야 한다.\n"
        "   ★ 못 적겠거나 넣어도 성립 안 하면(audience / story / role / face 처럼)\n"
        "     **그 단어를 고르지 마라.** 다른 문장에서 다른 단어를 고른다.\n"
        "   ★★ 품사는 안 가린다. **방향이 있는가**만 본다.\n"
        "     부사도 방향이 있으면 쓸 수 있다.\n"
        "       [O] rarely → frequently        빈도가 뒤집힌다\n"
        "       [O] willingly → reluctantly    태도가 뒤집힌다\n"
        "       [O] naturally → artificially   방식이 뒤집힌다\n"
        "       [X] mentally → ???             영역을 가리킬 뿐 뒤집을 게 없다\n"
        "       [X] understandably → ???       같음\n"
        "       [X] clearly → unclearly?       억지스럽고 방향이 약하다\n"
        "     ★ 기출 정답에 부사가 0개인 건 부사가 금지라서가 아니라\n"
        "       방향 있는 부사가 드물기 때문이다. 있으면 써도 된다.\n"
        "   ★ 정답 자리 하나만 antonym 을 shown 에 그대로 옮긴다.\n"
        "     나머지 넷은 antonym 을 적어만 두고 shown 에는 **동의어**를 쓴다.\n"
        "════════════════════════════════════════════════════\n\n"
        "[PASSAGE]\n" + body + avoid + "\n\n"

        "## 무엇을 만드는가\n"
        "Choose FIVE words in the passage to underline. Four of them keep the meaning the\n"
        "passage intends; ONE is replaced by a word that CONTRADICTS the passage's logic.\n"
        "The student must find the contradictory one.\n\n"

        "## STEP 1 — 논지를 먼저 잡아라\n"
        "State to yourself: what does this passage argue, and which sentence carries it?\n"
        "Every underlined slot must be a word whose direction the argument decides.\n\n"

        "## STEP 2 — 밑줄 자리 고르기 (기출 실측을 따르라)\n"
        "  · NEVER underline a word in the passage's FIRST sentence. 기출 0/7.\n"
        "    That sentence sets the premise; if it wavers, nothing else can be judged.\n"
        "  · 다섯 자리를 지문 전체에 **고르게** 흩어라. 한 문장에 둘은 안 된다.\n"
        "    ★ 뒤쪽에 몰지 마라. 기출이 61% 지점에 몰리는 것은 원문 순서 지문이라\n"
        "      '앞에서 확인시키고 뒤에서 뒤집는' 구조가 성립하기 때문이다.\n"
        "      이 지문은 (A)(B)(C)가 셔플돼 앞뒤 개념이 없다. 처음부터 끝까지 균등하게.\n"
        "  · Underline only words whose OPPOSITE would change the argument:\n"
        "      GOOD  significant / diminished / undermines / drives / absolute / insignificant\n"
        "            justifies / abandon / permanently / modesty / careful / least\n"
        "      BAD   tasks / time / people / products / efforts / doing / extent\n"
        "            (concrete nouns and vague words have no meaningful opposite)\n"
        "  · Grammatical make-up should follow 기출: about two adjectives, two verbs,\n"
        "    one noun. Avoid adverbs.\n"
        "  · ★ NEVER underline a sentence-initial connective or discourse marker —\n"
        "    Similarly, Conversely, However, Moreover, Therefore, Instead, Meanwhile,\n"
        "    Nevertheless, Consequently, Perhaps, Indeed, Finally.\n"
        "    These signal the flow of argument, not a judgment about meaning in context.\n"
        "    부사는 기출 정답에 한 번도 없었다.\n"
        "  · Never underline the same word twice.\n"
        "  · A sentence-final word is fine — 기출에도 흔하다 (2025 수능 ③ 'uncomfortable.').\n"
        "    Give 'original' exactly as it appears in the passage, punctuation included.\n\n"

        "## STEP 3 — 정답 자리와 반의어\n"
        + _want_txt +

        "  · ★★ 정답 자리의 기준은 품사가 아니라 '반대말이 있는가' 다.\n"
        "    고르기 전에 스스로 물어라 — 이 단어의 반대말을 댈 수 있는가?\n"
        "    못 대면 그 자리는 쓸 수 없다. 반전할 수가 없기 때문이다.\n"
        "      [O] 형용사  significant ↔ trivial / inhabitable ↔ uninhabitable\n"
        "      [O] 동사    undermines ↔ reinforces / rely ↔ disregard / abandon ↔ retain\n"
        "      [O] 명사    modesty ↔ arrogance / majority ↔ minority / presence ↔ absence\n"
        "                  기출에도 명사가 쓰인다 — modesty(연성 하천공학), restrictions(바자르),\n"
        "                  concern(상황윤리), necessity. 방향만 있으면 명사도 정답이 된다.\n"
        "      [X] story / tasks / readers / line — 반대말 자체가 없다\n"
        "      [X] capriciously / arbitrarily     — 부사는 기출 정답에 없다\n"
        "    실측 실패: 정답 자리에 'story.' 를 골라 바꿀 말이 없어 'story.' → 'story.' 를\n"
        "    냈고, 그 지문이 3회 재시도 끝에 통째로 누락됐다.\n"
        "    ★ 명사를 꺼리지 마라. 2015 수능 30번 정답이 'concern'(명사)이고,\n"
        "      연성 하천공학은 'modesty', 바자르는 'restrictions' 였다.\n"
        "      품사가 아니라 '이 문맥에서 방향을 뒤집을 수 있는가'가 기준이다.\n"
        "  · Replace it with a word of OPPOSITE direction — not a random word, not a\n"
        "    near-synonym. The sentence must still read grammatically.\n"
        "  · ★★ **철자로 답이 보이면 안 된다.** 정답 자리의 반의어는 원문어와\n"
        "    글자가 확연히 달라야 한다. 학생이 논지를 안 읽고 눈으로 찾으면 실패다.\n"
        "    [X] inhabitable → uninhabitable   ('un-' 이 붙은 게 바로 보인다)\n"
        "    [X] possible → impossible         [X] regular → irregular\n"
        "    [X] agree → disagree              [X] likely → unlikely\n"
        "    [O] inhabitable → barren          [O] possible → futile\n"
        "    [O] regular → erratic             [O] agree → object\n"
        "    ★ 어근이 다른 말을 쓰면 자연히 해결된다.\n"
        "    ★★ 오답 자리(①②④⑤)에는 **절대 반의어를 넣지 마라.** 정답이 둘이 된다.\n"
        "      그 자리의 antonym 칸에 적은 말을 shown 에 그대로 옮기면 안 된다는 뜻이다.\n"
        "      실측 실패: 원문 'inhabitable' 을 오답 자리에서 'habitable' 로 보여줬다\n"
        "      — in- 을 떼어 반대 뜻이 됐다.\n"
        "      [X] inhabitable → habitable   [O] inhabitable → livable\n"
        "  · ★ 정답 자리의 치환어도 나머지 넷과 같은 수준·같은 문체여야 한다.\n"
        "    정답만 유난히 어렵거나 쉬우면 학생이 내용을 안 읽고 그것부터 찍는다.\n"
        "    기출은 정답과 오답의 난이도가 구분되지 않는다 — 그게 30번의 핵심이다.\n"
        "  · ★ NEVER use a word that merely LOOKS or SOUNDS similar (affect/effect,\n"
        "    adapt/adopt, comprise/compose, principal/principle). That tests spelling,\n"
        "    not reading. The wrong word must be a genuine ANTONYM whose meaning the\n"
        "    passage's logic rules out. 1회독 교재도 같은 규칙을 쓴다.\n"
        "  · ★ The contradiction must be provable from ONE of these three, exactly as 기출:\n"
        "      (a) THE NEXT SENTENCE — 3/7 세트\n"
        "          'lead to ④ stronger motivation' → next sentence says 'lower task motivation'\n"
        "          'accepts variability as ⑤ insignificant' → next says 'presumed important'\n"
        "      (b) PARALLEL OR CAUSAL LINK INSIDE THE SAME SENTENCE — 2/7\n"
        "          'diminished emphasis on dialogue AND a ③ significant emphasis on song'\n"
        "            → 'and' binds the direction; it must be 'reduced'\n"
        "          'not to ask for very ④ low prices BECAUSE not an absolute necessity'\n"
        "            → 'because' supplies the reason; it must be 'high'\n"
        "      (c) THE PASSAGE'S OVERALL THESIS — 2/7\n"
        "          thesis: external control harms autonomy\n"
        "            → '④ drives the acquisition of self-responsibility' reverses it\n"
        "    Name which of (a)(b)(c) you used, and quote the exact evidence.\n\n"

        "## ★★ 유의어·반의어는 사전이 아니라 '이 문맥에서' 판정한다\n"
        "이것이 평가원 30번의 핵심이고, 여기서 가장 많이 틀린다.\n"
        "\n"
        "### 사전적 동의어가 아니라 '이 문장에서 바꿔 넣어도 논지가 그대로인 말'\n"
        "  · 사전에서 같은 뜻이어도 이 문맥에 안 맞으면 못 쓴다.\n"
        "      'a critical moment'(결정적) 를 'a judgmental moment'(비판적) 로 바꾸면\n"
        "      사전상 critical 의 한 뜻이지만 이 문장에서는 딴 말이 된다.\n"
        "  · 반대로 사전에 유의어로 안 실려 있어도, 이 문맥에서 같은 일을 하면 쓸 수 있다.\n"
        "      'the brain uses multiple routes' 의 'uses' → 'draws on'\n"
        "  · 바꿔 넣고 그 문장을 소리 내어 읽어라. 앞뒤 문장과의 관계가 그대로여야 한다.\n"
        "\n"
        "### 반의어도 '이 문맥에서 논지를 뒤집는 말'\n"
        "  · 사전적 반대말을 찾는 게 아니다. 그 자리에 넣었을 때 **지문이 하는 주장과\n"
        "    어긋나는** 말을 찾는 것이다.\n"
        "      지문: 뇌는 신호가 막히면 다른 경로로 갈아탄다 (= 유연함)\n"
        "      원문 'flexibility' → 'rigidity'\n"
        "      rigidity 가 flexibility 의 사전적 반대라서가 아니라, 그 자리에 넣으면\n"
        "      '갈아탄다'는 지문의 주장과 정면으로 어긋나기 때문이다.\n"
        "  · ★ 문장 자체는 여전히 문법적이고 그럴듯해야 한다. 읽다가 걸리면 안 된다.\n"
        "    걸려서 아는 게 아니라, **앞뒤를 대조해서** 알아야 한다.\n"
        "  · 그래서 '반대말이 없는 단어'는 못 쓴다 — story / tasks / readers 는\n"
        "    무엇으로 바꿔도 논지가 뒤집히지 않는다. 방향이 없기 때문이다.\n"
        "\n"
        "### 출력 직전에 다섯 자리를 전부 대입해 읽어라\n"
        "  네 자리: 바꿔 넣어도 지문의 주장이 그대로인가? 아니면 그 자리를 다시 잡아라.\n"
        "  한 자리: 바꿔 넣으면 지문의 주장과 어긋나는가? 안 어긋나면 정답이 성립 안 한다.\n\n"

        "## STEP 4 — 나머지 네 자리도 패러프레이즈하라\n"
        "  ★ Do NOT leave the four correct slots as the passage's own words. Replace each\n"
        "    with a SYNONYM that keeps the meaning. Otherwise a student can skip them as\n"
        "    'words that were already there' and only weigh the odd one out.\n"
        "      passage 'salient'    → shown 'conspicuous'\n"
        "      passage 'assess'     → shown 'gauge'\n"
        "      passage 'careful'    → shown 'circumspect'\n"
        "  ★ The synonym must fit the sentence grammatically and keep the same part of speech.\n"
        "  ★ Likewise, never swap in a look-alike word here either. A synonym must share\n"
        "    the MEANING, not the spelling.\n"
        "  ★ If the original carries punctuation ('uncomfortable.'), keep it on the synonym\n"
        "    ('disconcerted.') so the sentence still reads correctly.\n"
        "  ★★ shown 은 original 과 반드시 달라야 한다. 같으면 그 항목은 버려진다.\n"
        "    특히 정답 자리가 같으면 '틀린 단어'가 아예 없어 문항이 성립하지 않는다.\n"
        "    실측 실패: 정답 3번 'story.' → 'story.' — 바뀐 게 없어 3회 재시도 끝에\n"
        "    그 지문이 통째로 누락됐다.\n"
        "    출력 직전 다섯 항목을 훑어 original != shown 인지 하나씩 확인하라.\n"
        "  ★ YOU choose every word here — both the four synonyms and the one antonym.\n"
        "    Pick words that a CSAT student would plausibly accept in that slot; the point is\n"
        "    that only the argument, not the word's surface, reveals which one is wrong.\n\n"

        "## STEP 5 — 어휘 수준과 문체: 평가원 30번 그대로\n"
        "\n"
        "### 어떤 종류의 말인가 — 이게 목록보다 중요하다\n"
        "  · ★ 주제어가 아니라 **판단어**를 쓴다.\n"
        "    지문이 무엇에 관한 글인지 알려주는 말(perception, territory, algorithm)이\n"
        "    아니라, 그것을 어떻게 평가·규정하는지를 담은 말이다.\n"
        "      [O] significant / diminished / undermines / justifies / absolute / abandon\n"
        "\n"
        "  ★★★ 자리를 고르는 방법 — **다섯 자리 전부에 반대말을 실제로 써 봐라.**\n"
        "  이건 목록으로 외울 일이 아니다. 후보 단어를 놓고 이렇게 적어 보면 바로 갈린다.\n"
        "\n"
        "      significant  → 반대말 trivial        O  쓸 수 있다\n"
        "      undermines   → 반대말 reinforces     O\n"
        "      concern      → 반대말 indifference   O  (명사여도 방향이 있다)\n"
        "      modesty      → 반대말 arrogance      O\n"
        "      dissuade     → 반대말 persuade       O  (어미가 특이해도 방향이 있다)\n"
        "\n"
        "      audience     → 반대말 ???           X  못 쓴다 → 이 자리는 버려라\n"
        "      story        → 반대말 ???           X\n"
        "      region       → 반대말 ???           X\n"
        "      brain        → 반대말 ???           X\n"
        "      process      → 반대말 ???           X\n"
        "\n"
        "  ★ 출력의 `antonym` 칸에 그 반대말을 **반드시 적어라.** 다섯 자리 전부다.\n"
        "    (정답 자리는 실제로 그 반대말을 shown 에 넣고, 오답 자리는 적기만 한다 —\n"
        "     적을 수 있는 자리인지 확인하는 용도다.)\n"
        "  ★ 반대말을 못 적겠으면 **그 문장을 밑줄 자리로 쓰지 마라.** 다른 문장을 고른다.\n"
        "    실측 실패: 'audience' 가 밑줄로 나갔다. 사람·집단·구체 사물은 방향이 없어\n"
        "    무엇으로 바꿔도 논지가 뒤집히지 않는다.\n"
        "      [X] perception / territory / readers / process — 방향이 없다\n"
        "  · 라틴·프랑스 어원의 학술적 추상어. 수능 필수어휘 범위 안에서 고른다.\n"
        "  · 파생명사(-ity, -tion, -ness)보다 형용사·동사 어근형이 자연스럽다.\n"
        "    기출 밑줄 품사도 형용사 37% 동사 37%로 74%가 이 둘이다.\n"
        "\n"
        "### 난이도 — 다섯 개가 고르게\n"
        "  수능 수준      significant, absolute, internal, permanent, similar, necessary,\n"
        "                 assess, justify, abandon, restore, undermine, diminish\n"
        "  반 단계 위      discernible, contingent, provisional, marginal, redundant,\n"
        "                 tenable, latent, incremental, adverse, salient, tacit,\n"
        "                 conspicuous, circumspect, gauge, forfeit, curtail\n"
        "  ★ 이 목록은 '수준의 눈금'이지 고르는 통이 아니다. 문맥에 맞는 말을 쓰되\n"
        "    난이도가 이 범위 안에 들어오게 하라.\n"
        "  ★★ 다섯 개의 난이도가 고르게 깔려야 한다. 하나만 유난히 어려우면 학생이\n"
        "    내용을 안 읽고 그것부터 의심한다 — 정답이 어휘 난이도로 새어나간다.\n"
        "    각 단에서 두셋씩 섞어라. 다섯 다 어려우면 인위적이고, 다섯 다 쉬우면\n"
        "    변별이 안 된다.\n"
        "\n"
        "### 문체 — 학술 산문. 이런 말은 쓰지 않는다\n"
        "  [X] 구어체·구동사      get, make it, stuff, a lot of, kind of, come up with\n"
        "  [X] 과장·감정어        amazing, terrible, awesome, devastating, crucial하게 남발\n"
        "  [X] 지문 밖 전문용어    지문이 다루지 않는 분야의 술어를 끌어오지 마라\n"
        "  [X] 최신 조어·유행어    학술 산문에 안 쓰는 말\n"
        "  ★ 밑줄 단어를 넣은 문장을 읽었을 때, 원래 지문과 같은 결로 읽혀야 한다.\n"
        "    갑자기 문체가 튀면 그 자리가 표가 난다.\n"
        "\n"
        "### ★★ original 은 지문에서 '베껴' 적어라 — 여기서 가장 많이 버려진다\n"
        "  · ★★★ original 은 네가 기억해서 쓰는 게 아니다. **지문의 그 자리를 눈으로 보고\n"
        "    글자를 그대로 옮겨** 적는 것이다. 구두점·대소문자까지 포함해서다.\n"
        "    실측 실패 — 지문에 없는 단어를 적어 그 항목이 통째로 버려졌다:\n"
        "      지문 'internal features such as eyes'  →  original 에 'external' 이라고 적음\n"
        "      지문 'vicarious functioning.'          →  original 에 'flexibility' 라고 적음\n"
        "    비슷한 뜻의 다른 단어를 쓰면 안 된다. 그 자리에 **실제로 인쇄된** 글자여야 한다.\n"
        "  · para 와 idx 를 적기 전에, 그 단락을 공백으로 끊어 세어 확인하라.\n"
        "    idx 가 틀려도 코드가 근처에서 찾아 보정하지만, original 자체가 지문에 없으면\n"
        "    찾을 방법이 없다.\n"
        "  · ★★ **그 자리에 넣었을 때 문장이 읽히는가**를 머릿속으로 확인하라.\n"
        "    어휘 문제는 단어 하나만 바꾸는 것이다. 그 하나가 문장을 깨면 안 된다.\n"
        "      지문 'the brain depends on external features'\n"
        "        [O] relies   the brain relies on ...   읽힌다\n"
        "        [X] rely     the brain rely on ...     수일치가 깨진다\n"
        "      지문 'It is extraordinarily adapted to a changing world'\n"
        "        [O] adjusted   [X] adjust   (is + 과거분사 자리다)\n"
        "    ★ 어미가 달라 보여도 읽히면 정상이다.\n"
        "      지문 'no matter how exciting it was'  →  [O] compelling\n"
        "      (exciting 도 compelling 도 형용사다. -ing 로 끝난다고 동사가 아니다)\n"
        "  · 원문에 구두점이 붙어 있으면 치환어에도 붙인다\n"
        "    ('uncomfortable.' → 'disconcerted.').\n"
        "  · ★★ 구동사(phrasal verb)의 동사만 바꾸지 마라. 뒤에 남은 전치사가 어긋난다.\n"
        "    실측 실패: 원문 'rely **on** landmarks' 에서 rely 만 disregard 로 바꿨다.\n"
        "      [X] disregard on landmarks   — disregard 는 타동사라 on 을 안 받는다.\n"
        "          학생이 뜻을 몰라도 문법만으로 소거한다.\n"
        "      [O] ignore ... 처럼 전치사 없이 쓰는 말이면 그 자리를 아예 고르지 마라.\n"
        "      [O] 'rely on' 자리를 쓰려면 같은 전치사를 받는 말로 — 'depend'(on),\n"
        "          'count'(on), 'lean'(on) 처럼.\n"
        "    ★ 밑줄 단어 뒤에 전치사가 붙어 있으면(on/to/for/with/in/at/from),\n"
        "      치환어도 그 전치사를 자연스럽게 받는 말이어야 한다.\n"
        "  · 첫 글자 대소문자도 맞춘다. 문장 첫 단어를 소문자로 내면 문장이 깨진다.\n"
        "  ★ 다섯 자리는 서로 다른 단어여야 한다 — **굴절형도 같은 단어로 본다.**\n"
        "    'depends' 와 'depend' 를 둘 다 밑줄 치지 마라. 학생 눈에는 같은 말이다.\n"
        "  · 선지는 밑줄 단어 하나만 보인다 — 구가 아니라 한 단어다.\n\n"

        "## OUTPUT — 자리는 '몇 번째 단락, 몇 번째 단어'로 정확히 지목하라\n"
        "  ★ why 는 정답 항목에만, 40자 이내 한 줄로 쓴다. 답지에 한 줄로 들어가므로\n"
        "    문단으로 늘어놓지 마라. 오답 4개의 why 는 비워도 된다.\n"
        "  ★ vocab_explain 은 빈 문자열로 둔다 — 답지에 이미 근거·이유·나머지 선지가 나온다.\n"
        "  Count words by splitting on spaces, starting at 0, within that paragraph only.\n"
        "  'original' must be the passage's word at that exact index, punctuation included\n"
        "  as it appears.\n"
        + (f"  ★ 출력 직전 확인: is_answer 가 true 인 항목의 n 이 {want_n} 인가?\n"
           f"    아래 예시 JSON 은 n:3 을 정답으로 보여주지만 그건 형식 예시일 뿐이다.\n"
           if want_n in (1, 2, 3, 4, 5) else "")
        + ("\n"
           "## ★★ 출력 직전 자가점검 — 코드가 안 보는 것들이다. 네가 봐야 한다.\n"
           "\n"
           "  1. **다섯 자리가 서로 다른 문장에 있는가?**\n"
           "     한 문장에 둘을 치면 그 문장만 유난히 촘촘해져 눈에 띈다.\n"
           "     문장 수가 다섯보다 적으면 어쩔 수 없지만, 그 전엔 최대한 흩어라.\n"
           "\n"
           "  2. **철자만 비슷한 말로 바꾸지 않았는가?**\n"
           "     [X] affect → effect / adapt → adopt / principal → principle\n"
           "     [X] complement → compliment / assure → ensure\n"
           "     이건 독해가 아니라 철자 암기를 묻는 것이다. 뜻으로 갈리게 하라.\n"
           "     (단 부정 접두사는 따로 금지 — 아래 4번)\n"
           "\n"
           "  3. **antonym 다섯 개를 전부 적었는가?**\n"
           "     못 적은 자리가 있으면 그 문장을 버리고 다른 문장을 골라라.\n"
           "     시험: 'audience' 의 반대말? — 못 댄다. 그러면 쓰지 마라.\n"
           "\n"
           "  4. **부정 접두사만 붙이거나 떼지 않았는가?**\n"
           "     [X] inhabitable ↔ uninhabitable / possible ↔ impossible\n"
           "     철자로 답이 새어나간다. 어근이 다른 말을 써라.\n"
           "\n"
           "  5. **다섯 자리가 서로 다른 단어인가?** 굴절형도 같은 단어다\n"
           "     ('depends' 와 'depend' 를 둘 다 치지 마라).\n"
           "\n"
           "  6. **다섯 개의 난이도가 고른가?** 정답만 유난히 어려우면\n"
           "     학생이 내용을 안 읽고 그것부터 찍는다.\n")
        + "\n"

        "★★ 다섯 항목 **전부** antonym 을 채워라. 하나라도 비면 버려진다.\n"
        "   그 단어의 반대말 한 단어. 못 적으면 그 자리를 쓰지 마라.\n\n"
        '{"vocab_items": [\n'
        '   {"n": 1, "para": 0, "idx": 12, "original": "exciting",\n'
        '    "antonym": "dull",  "shown": "thrilling", "is_answer": false,\n'
        '    "why": "도입부의 흡인력을 말하는 자리"},\n'
        '   {"n": 2, "para": 1, "idx": 5,  "original": "internal",\n'
        '    "antonym": "external", "shown": "intrinsic", "is_answer": false,\n'
        '    "why": "..."},\n'
        '   {"n": 3, "para": 1, "idx": 22, "original": "convince",\n'
        '    "antonym": "dissuade",\n'
        '    "shown": "dissuade",\n'
        '    "is_answer": true,\n'
        '    "evidence_type": "next_sentence | same_sentence | thesis",\n'
        '    "evidence": "<quote the exact words that prove the contradiction>",\n'
        '    "why": "<왜 틀렸는지 한 줄. 40자 이내. 예: \'초반에 주의를 끌면 독자를 설득한다는 인과와 정반대\'>"},\n'
        '   {"n": 4, "para": 2, "idx": 8,  "original": "adapted",\n'
        '    "antonym": "unsuited", "shown": "adjusted", "is_answer": false,\n'
        '    "why": "..."},\n'
        '   {"n": 5, "para": 2, "idx": 30, "original": "greater",\n'
        '    "antonym": "lesser", "shown": "larger", "is_answer": false,\n'
        '    "why": "..."}\n'
        '],\n'
        ' "vocab_explain": ""}'
    )


# ════════════════════════════════════════════════════════════════
# A Q5 — 빈칸영작 자리 선정 (수능 32~34번 빈칸 논리를 영작에 적용)
#   기출 23문항 실측:
#     · 빈칸 위치: 결론 39% / 논지핵심 17% / 두괄식 주제문 17% / 전환점 12%
#     · 평균 위치 0.66 (지문 2/3 지점) — 첫 문장 17%, 마지막 문장 30%
#     · 단어 수 4~11, 평균 6.9
#     · 근거 위치: 뒤 문장 14회 / 앞 문장 10회 / 글 전체·예시 9회
#   → 평가원은 '단어 수'가 아니라 '논지가 착지하는 자리'를 먼저 고른다.
# ════════════════════════════════════════════════════════════════
Q5_BLANK_SYS = (
    "You select fill-in-the-blank spans for a Korean CSAT-style writing task. "
    "Output ONLY valid JSON, no markdown, no text outside JSON."
)


def build_q5_blank_prompt(paragraphs) -> str:
    """paragraphs: [[label, text], ...] 원문 그대로 (마커 없음)

    ★ _s93에서 '위치 지목(starts_with/ends_with)'을 폐기하고 구절을 통째로 받는다.

    _s70에서 지목 방식으로 바꾼 이유는 창작이었다 — LLM이 논지를 자기 말로 요약해
    원문에 없는 구절('The territory known as Bir Tawil')을 만들었다. 그런데 그건
    '원문에 없으면 버린다' 검사 하나로 막을 일이었고, validate_llm_q5_spans 에
    이미 그 검사가 있다. 구조를 바꿀 필요가 없었다.

    대신 지목 방식은 길이 통제를 잃었다. 시작 두 단어와 끝 두 단어만 짚으면 그 사이가
    몇 단어인지 LLM 눈에 안 보인다. 실측: 5시도 전부 폴백, 반려 24건 중 17건이
    길이·구두점. word_count 자기신고를 넣어도 11건 전부 실제와 어긋났다
    (6단어라 적고 19단어를 잡음). LLM은 단어를 못 센다.

    구절을 직접 쓰게 하면 길이가 눈에 보인다. 세는 게 아니라 보는 것이라 된다.
    그리고 '몇 단어'가 아니라 '한 문법 단위'를 기준으로 주면 저절로 그 길이가 된다.
    (수능 32~34번 23문항 115선지 실측: 3~12단어, 평균 7.4, 중앙값 7, 6~8단어가 62%,
     12단어 초과 0개)
    """
    lines = []
    n_para = len(paragraphs)
    for pi, (lab, txt) in enumerate(paragraphs):
        body_txt = re.sub(r"\s+", " ", str(txt or "")).strip()
        nw = len(body_txt.split())
        lines.append("=" * 62)
        lines.append(f"### para = {pi}   (label: {lab})   [{nw} words]")
        lines.append("=" * 62)
        lines.append(body_txt)
        lines.append("")
    lines.append("=" * 62)
    lines.append(f"\u203b \uc774 \uc9c0\ubb38\uc740 \uc704 {n_para}\uac1c \ub2e8\ub77d\uc73c\ub85c \ub098\ub284\ub2e4 (para = 0 ~ {n_para - 1}).")
    lines.append("   blank_A \uc640 blank_B \ub294 \uc11c\ub85c \ub2e4\ub978 \ub450 \ub2e8\ub77d\uc5d0\uc11c \uace8\ub77c\uc57c \ud55c\ub2e4.")
    lines.append("=" * 62)
    body = "\n".join(lines)
    return (
        "Pick TWO phrases from this passage to blank out for a word-order writing task.\n\n"
        "[PASSAGE]\n" + body + "\n"

        "## 이 문제가 무엇인가\n"
        "학생은 빈칸 뚫린 지문과 뒤섞인 단어 보기를 받고, 원문 표현을 그대로 되맞춘다.\n"
        "그러니 네가 고르는 구절은 지문에 인쇄된 글자 그대로여야 한다.\n"
        "요약하거나 다듬으면 학생이 맞출 방법이 없어 그 문항은 버려진다.\n"
        "  [X] 'The territory known as Bir Tawil'  — 지문에 없는 표현을 지어낸 실제 사례\n"
        "  [O] 지문을 눈으로 훑어 그 자리 글자를 그대로 옮겨 적는다\n\n"

        "## ★★ 네 몫과 코드 몫 — 무엇을 판단하고 무엇을 맡길 것인가\n"
        "네가 하는 일은 하나다. **논지가 어디에 실려 있는가**를 판단하는 것.\n"
        "그건 의미 판단이라 코드가 못 한다.\n"
        "\n"
        "  네 몫  · 어느 자리를 뚫을 것인가 (STEP 2 의 (A)(B)(C)(D))\n"
        "         · 어디서 시작하고 어디서 끊을 것인가 (성분 하나 / 남는 게 단서)\n"
        "         · 근거가 되는 옆 문장을 짚는 것\n"
        "\n"
        "  코드 몫 · 그 구절이 지문에 글자 그대로 있는가\n"
        "         · 되넣으면 원문이 복원되는가\n"
        "         · (A)(B)가 서로 다른 단락인가, 등장 순서가 맞는가\n"
        "         · 보기(word bank) 생성 — 네가 만들 필요 없다\n"
        "\n"
        "★ **상한은** 세느라 애쓰지 마라. 성분 하나를 잡으면 12단어를 넘을 수가 없다.\n"
        "★ **하한은 반드시 세라.** 이건 성분 규칙으로 안 걸린다.\n"
        "  'designed to'(2단어)는 성분을 둘 문 게 아니라 **끝까지 안 잡은** 것이다.\n"
        "  성분 이름을 못 대는데도 짧다는 걸 스스로 못 알아챈다. 세야만 안다.\n"
        "  네 개를 넘는지만 보면 된다 — 그건 세기 쉽다. (실측 실패: 'designed to')\n"
        "★ 보기를 만들지 마라. 코드가 정답 구절에서 자동 생성한다.\n"
        "★ 대신 **지문에 인쇄된 글자를 그대로 옮기는 것**만은 네가 해야 한다.\n"
        "  한 글자라도 다르면 코드가 찾지 못해 그 항목이 버려진다.\n\n"

        "## STEP 1 — 논지를 먼저 잡아라\n"
        "이 지문은 무엇을 주장하는가. 그 주장이 어느 문장에 실려 있는가.\n"
        "두괄식인가 미괄식인가. 전환점(But / However / in fact / Thus)은 어디인가.\n\n"

        "## STEP 2 — 어느 자리를 뚫는가\n"
        "\n"
        "### 원칙 하나\n"
        "기출 23문항을 뜯어보면 공통점이 하나다.\n"
        "**빈칸을 지우고 읽었을 때, 남은 뼈대만으로 '여기에 무엇이 와야 하는지'가 결정된다.**\n"
        "네 가지 방식으로 그렇게 만든다. 이 중 하나에 해당하는 자리만 골라라.\n"
        "\n"
        "### (A) 구문이 틀을 만든다 — 빈칸 밖에 뼈대가 남는다\n"
        "  If such laws forbid them to do something..., then the law cannot be ______.\n"
        "      If~then 의 then절. 조건은 지문에 남고 귀결만 뚫는다.\n"
        "  he affirms not only his aesthetic judgment, ... but the conviction that ______.\n"
        "      not only A but B 의 B. A가 지문에 남아 대비 축이 된다.\n"
        "  The more affected one is, ______.\n"
        "      the비교급~the비교급의 뒷절. 앞절이 방향을 지정한다.\n"
        "  But even imperfect possession of it ______ of being 'stimulus-driven'\n"
        "      뒤에 of구가 남아 '무엇으로부터'가 확정된다.\n"
        "  ★ 문장을 통째로 뚫으면 이 뼈대가 사라진다. 성분 하나만 뚫어야 뼈대가 남는다.\n"
        "\n"
        "### (B) 옆 문장이 같은 말을 다시 한다\n"
        "  As writers, we have to ______; in effect, we have to imagine both halves\n"
        "  of a virtual conversation.        ← 세미콜론 뒤가 빈칸의 패러프레이즈\n"
        "  But tags aren't ______. It matters who is doing what, and tags don't capture this.\n"
        "                                    ← 뒤 문장이 빈칸을 풀어 설명한다\n"
        "  Thus, being creative ______. In fact, the ability to generate creative ideas\n"
        "  is essentially useless if these ideas die a silent death.   ← In fact 로 재진술\n"
        "\n"
        "### (C) 인과 연결어가 방향을 지정한다\n"
        "  No wonder it is difficult for most people to ______ in a group discussion.\n"
        "      앞의 인용(입술이 침묵해도 손끝이 지껄인다)의 귀결. No wonder 가 인과를 건다.\n"
        "  And how she views the street ______. That's why we find so many citizens\n"
        "  arguing past one another.        ← That's why 앞이 원인. 빈칸이 그 원인이다.\n"
        "\n"
        "### (D) 두괄식 — 첫 문장을 뚫고 뒤 전체가 근거가 된다\n"
        "  Centralized, formal rules can ______.\n"
        "      뒤에 야구 규칙·악보·법인 설립 예시가 줄줄이 나온다. 그 전부가 근거.\n"
        "  Prior to photography, ______.\n"
        "      뒤에 '그림은 제작이 오래 걸리고 운반이 어렵고 유일본이었다'가 이유로 붙는다.\n"
        "\n"
        "### 뚫으면 안 되는 자리\n"
        "  · 도입부 배경 설명 — 근거가 될 옆 문장이 없다\n"
        "  · 예시 나열 중 한 항목 — 그 항목만의 근거가 없다\n"
        "  · 앞뒤 어디에도 같은 말을 다시 하는 문장이 없는 자리\n"
        "★ why 에 네 자리가 (A)(B)(C)(D) 중 무엇인지, 그리고 근거가 되는 옆 문장을\n"
        "  그대로 옮겨 적어라. 못 옮기면 그 자리는 쓰면 안 된다.\n\n"

        "## STEP 3 — 얼마나 뚫는가 ★★ 4~12단어. 여기서 가장 많이 틀린다\n"
        "\n"
        "### 상한은 세지 마라. 하한은 반드시 세라.\n"
        "★ 12단어를 넘는지는 '성분 하나인가'만 보면 저절로 걸린다 — 세지 않아도 된다.\n"
        "★ 그러나 **4단어 미만은 반드시 직접 확인하라.** 보기 단어가 두세 개뿐이면\n"
        "  배열 문제가 성립하지 않는다. 실측 실패: 'designed to'(2단어)를 골라 버려졌다.\n"
        "기출 빈칸을 뜯어보면 전부 문장 성분 하나다. 두 절에 걸친 것이 하나도 없다.\n"
        "  · 술부 하나        — 'Centralized, formal rules can ______.'\n"
        "  · 보어 하나        — 'then the law cannot be ______.'\n"
        "  · to부정사구 하나  — 'it is difficult for most people to ______ in a discussion.'\n"
        "  · that절 내용 하나 — 'the conviction that ______.'\n"
        "  · 주절 하나        — 'Prior to photography, ______.'\n"
        "★ 이렇게 끊으면 길이는 저절로 맞는다. 실측 3~12단어, 평균 7.4, 중앙값 7,\n"
        "  6~8단어가 62%. 12단어를 넘는 기출 빈칸은 115개 중 0개다.\n"
        "\n"
        "### 길어지는 이유는 딱 하나 — 성분을 두 개 이상 물었을 때다\n"
        "  원문: But if it's a cloudy day, they may instead rely on landmarks\n"
        "  [X] if it's a cloudy day, they may instead rely on landmarks   (11)\n"
        "      조건절 + 주절. 성분 두 개를 물었다.\n"
        "  [O] may instead rely on landmarks   (5)\n"
        "      주절의 술부 하나. 조건절은 지문에 남아 문맥 단서가 된다.\n"
        "\n"
        "  원문: very few of your readers would make it to your dramatic conclusion\n"
        "  [X] very few of your readers would make it to your dramatic conclusion   (12)\n"
        "      주어부 + 술부.\n"
        "  [O] make it to your dramatic conclusion   (6)\n"
        "      술부 하나. 주어는 지문에 남는다.\n"
        "\n"
        "### 짧아지는 이유 — 성분을 끝까지 안 잡았을 때다\n"
        "  원문: The human perceptual system is designed to recognize objects\n"
        "        under changing illuminations, situations, and contexts.\n"
        "  [X] designed to                                         (2)\n"
        "      to부정사가 목적어도 없이 잘렸다. 보기 단어 두 개로는 배열이 안 된다.\n"
        "  [O] designed to recognize objects                       (4)\n"
        "      to부정사구. 'under changing illuminations...' 는 지문에 남아 단서가 된다.\n"
        "★ 성분 하나가 4단어도 안 되면 그 자리는 쓰지 말고 다른 곳을 골라라.\n"
        "\n"

        "### ★★ 어디까지 뚫나 — 빈칸 밖에 남는 것이 단서다\n"
        "기출은 수식어구를 다 물지 않는다. 뒤를 남겨 학생에게 단서를 준다.\n"
        "  frees a person from the burden  |  of being 'stimulus-driven'\n"
        "      the burden OF ~ 를 한복판에서 쪼갰다. 'of'가 남아\n"
        "      '무엇으로부터 벗어나게 하는가'라는 꼴이 확정된다.\n"
        "  conceal what they mean and feel  |  in a face-to-face group discussion\n"
        "      전치사구를 밖에 남겼다. 어디에서 일어나는 일인지가 지문에 인쇄돼 있다.\n"
        "  human beings must participate  |  in order for it to exist at all\n"
        "      부사구를 밖에 남겼다.\n"
        "★ 그러니 구를 중간에서 끊어도 된다. **뒤에 남는 것이 단서가 되는 지점**에서 끊어라.\n"
        "  뒤에 남길 게 없으면(문장 끝) 그냥 문장 끝까지 뚫으면 된다.\n\n"

        "## STEP 4 — 구두점\n"
        "### 쉼표·세미콜론·콜론은 빈칸 안에 있어도 된다\n"
        "보기에는 쉼표가 앞 단어에 붙어 제시된다 — 'original,' 'signals,' 처럼.\n"
        "그래서 학생이 쉼표 자리를 알 수 있다. 억지로 피하지 마라.\n"
        "  원문: Egypt understandably claimed the original, straight border of 1899\n"
        "  [O] claimed the original, straight border of 1899\n"
        "      보기: 1899 / claimed / straight / original, / of / border / the\n"
        "  ★ 절대 하지 말 것: 쉼표만 지워서 내는 것.\n"
        "  [X] claimed the original straight border of 1899\n"
        "      쉼표가 사라져 지문에 없는 문자열이 된다. 학생이 되맞출 수 없어 버려진다.\n"
        "      (실측 실패 — 거부 사유를 받고 쉼표만 지워서 다시 낸 사례가 있다)\n"
        "\n"
        "### 다만 이 둘은 안 된다\n"
        "  · 문장 경계(. ! ?)를 넘지 마라. 빈칸은 한 문장 안에 있어야 한다.\n"
        "  · 쉼표로 끝내지 마라. 끝 구두점은 빈칸 밖에 남긴다.\n"
        "      [X] 'dwindle and trail off,'   [O] 'dwindle and trail off'\n\n"

        "## STEP 5 — 경계 ★ 시작과 끝의 기준이 다르다 (기출 23개 실측)\n"
        "\n"
        "### 시작 — 거의 안 가린다\n"
        "기출 정답 빈칸 23개 중 7개(30%)가 기능어로 시작한다. 피하지 마라.\n"
        "  [O] the less similarity is required for the thing to appear\n"
        "  [O] a justification for converting nonforests into forests\n"
        "  [O] is often counterproductive in cases of conflict\n"
        "  [O] not in the perception of the figure but in its rational representation\n"
        "★ 막아야 하는 것은 **앞 절과 이어붙는 접속사·관계사**뿐이다.\n"
        "  그걸로 시작하면 빈칸 앞이 잘린 것처럼 읽힌다.\n"
        "  [X] 'as it would mean giving up a greater prize'   (as 로 시작)\n"
        "  [X] 'and then the reader loses interest'           (and 로 시작)\n"
        "  [X] 'which the collector cannot predict'           (which 로 시작)\n"
        "\n"
        "### 끝 — 엄격하다\n"
        "★ 기능어로 끝나는 기출 빈칸은 **23개 중 0개**다. 예외가 없다.\n"
        "  반드시 내용어(명사·동사·형용사·부사)로 끝나라.\n"
        "  [X] 'to your'   [X] 'over the course of'   [X] 'is required for the'\n"
        "  [O] 'to your dramatic conclusion'  [O] 'over the course of your writing'\n"
        "  · ★★ 하이픈 복합어로 끝내지 마라. 뒤에 꾸밀 명사가 반드시 따라온다.\n"
        "      [X] straight line of the twenty-second   — twenty-second parallel 을 쪼갬\n"
        "      [X] cited by a well-known                [X] produced a long-term\n"
        "      [O] straight line of the twenty-second parallel\n"
        "      [O] a well-known scholar of history      [O] produced a long-term effect\n"
        "  · 고유명사도 중간에서 쪼개지 마라 (Bir Tawil, Hala'ib Triangle).\n"
        "  · 관용구를 중간에서 끊지 마라\n"
        "      [X] trail off over the course  — over the course of 를 쪼갬\n\n"

        "## STEP 6 — (A)와 (B)\n"
        "  · ★ blank_A 의 para 와 blank_B 의 para 가 서로 달라야 한다.\n"
        "    같은 단락에 몰면 지문 한쪽만 비고 나머지가 온전해 읽기 균형이 깨진다.\n"
        "  · 후보는 para = 0 (intro, 주어진 글) 부터 마지막 단락까지 전부다.\n"
        "  · (A)가 (B)보다 지문에서 먼저 나와야 한다.\n"
        "  · 두 구절이 같은 논리 단계를 반복하지 않게 — (A)는 전제·전환, (B)는 결론.\n"
        "  · 각 구절은 그 단락 안에서 한 번만 나오는 표현이어야 한다.\n\n"

        "## 출력 직전 — 네가 쓴 구절을 다시 읽어라\n"
        "세지 말고 보라. 여섯 가지만 확인한다.\n"
        "  1. 이 구절이 지문에 글자 그대로 있는가? 지문에서 찾아 대조하라.\n"
        "  2. 이 자리가 (A)(B)(C)(D) 중 무엇인가? 넷 다 아니면 다른 자리를 골라라.\n"
        "  3. 근거가 되는 옆 문장을 그대로 옮겨 적을 수 있는가? 못 하면 못 쓰는 자리다.\n"
        "  4. 성분 하나인가, 두 개를 물었는가? 두 개면 뒤쪽 하나만 남겨라.\n"
        "  5. ★ 단어를 세어라. 4개 이상인가? 12개 이하인가?\n"
        "     세 개 이하면 성분을 끝까지 안 잡은 것이다 — 목적어·수식어까지 넣어라.\n"
        "  6. 마침표·물음표를 넘었는가? 넘었으면 한 문장 안으로 줄여라.\n"
        "     (쉼표는 넘어가도 된다 — 보기에 붙어 나간다)\n"
        "  7. blank_A.para 와 blank_B.para 가 다른 숫자인가?\n\n"

        "[OUTPUT JSON]\n"
        '{"blank_A": {"para": <n>, "text": "<지문에서 그대로 옮긴 구절>",\n'
        '             "unit": "<술부 | 보어 | to부정사구 | that절 | 주절 | 명사구>",\n'
        '             "slot_type": "<A구문틀 | B재진술 | C인과연결어 | D두괄식>",\n'
        '             "evidence": "<근거가 되는 옆 문장을 지문에서 그대로>"},\n'
        ' "blank_B": {"para": <n>, "text": "...", "unit": "...",\n'
        '             "slot_type": "...", "evidence": "..."}}'
    )


# ════════════════════════════════════════════════════════════════
# B Q1 — 문장 삽입 자리 선정 (수능 38·39번)
#   기존: 코드가 '가운데 문장부터' 하나씩 떼어보고 복원되면 채택 — 위치만 봤다.
#   변경: LLM이 '자리를 확정하는 단서가 있는 문장'을 고르고, 코드는 복원·간격만 검증.
#   삽입 문제의 본질은 위치가 아니라, 뺀 문장에 그 자리를 지목하는 단서가 있는가다.
# ════════════════════════════════════════════════════════════════
INSERT_SYS = (
    "You select the sentence to remove for a Korean CSAT sentence-insertion question. "
    "Output ONLY valid JSON, no markdown, no text outside JSON."
)


def build_insert_prompt(sentences) -> str:
    """sentences: 원문 문장 리스트 (분리된 상태)"""
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "Choose ONE sentence to lift out of this passage for a sentence-insertion question.\n\n"
        "[PASSAGE — numbered by sentence]\n" + numbered + "\n\n"

        "## 이 문제가 무엇인가\n"
        "The chosen sentence is shown in a box above the passage. The passage is printed with\n"
        "five numbered gaps. Students must decide which gap the sentence came from.\n"
        "★ It only works if the sentence CARRIES ITS OWN ADDRESS — something inside it that\n"
        "  points back to one specific preceding sentence, and forward to what follows.\n\n"

        "## STEP 1 — 자리를 확정하는 단서를 찾아라 (이것이 선정 기준의 전부다)\n"
        "Scan every sentence and ask: if I lifted this out, what inside it tells a reader\n"
        "exactly where it belongs? A good candidate has at least ONE strong anchor:\n\n"
        "  (a) DEMONSTRATIVE — this / these / such / that + 명사\n"
        "      'This shift forced designers to rethink materials.'\n"
        "      → 'this shift'가 앞 문장의 무엇을 받는지 명확해야 한다.\n"
        "  (b) PRONOUN — it / they / he / she, 지시 대상이 앞 문장에 하나뿐일 때\n"
        "      'They rarely produce anything of lasting value.'\n"
        "  (c) CONNECTIVE — However / Instead / Therefore / In contrast / For example / Yet\n"
        "      역접·인과·예시 신호어가 앞 문장과의 관계를 지정한다.\n"
        "  (d) DEFINITE ARTICLE — the + 앞에서 처음 소개된 명사\n"
        "      'The problem with this approach is cost.' → 앞에서 approach가 소개돼야 한다.\n"
        "  (e) TIME / SEQUENCE — Later / Then / Afterward / Once this happens\n\n"
        "★ 단서가 하나도 없는 문장은 고르지 마라. 어느 자리에 넣어도 말이 되어 정답이 성립하지 않는다.\n"
        "★ 단서가 앞을 가리키기만 하고 뒤와 안 이어지면 약하다. 앞뒤 양쪽에 걸리는 문장이 가장 좋다.\n\n"

        "## STEP 2 — 자리가 하나로 확정되는지 검증하라\n"
        "Take your chosen sentence out, then try inserting it at EVERY other gap.\n"
        "  · If it reads acceptably at two or more gaps, the item is broken — pick another sentence.\n"
        "  · If it reads correctly at exactly one gap, that is your answer.\n"
        "★ 특히 확인할 것: 그 문장을 뺀 뒤 앞뒤 문장이 바로 이어져도 자연스러운가?\n"
        "  자연스럽다면 그 문장은 '없어도 되는' 문장이라 삽입 문제가 안 된다.\n"
        "  뺐을 때 논리에 구멍이 생겨야 한다 — 지시어가 받을 대상이 사라지거나, 논리가 건너뛰거나.\n\n"

        "## STEP 3 — 피해야 할 문장\n"
        "  · 지문 첫 문장 — 도입이라 앞을 가리킬 대상이 없다.\n"
        "  · 독립적인 일반 진술 — 어디 놓아도 말이 되는 문장.\n"
        "  · 단순 예시 나열의 중간 항목 — 순서가 자유로워 자리가 확정되지 않는다.\n"
        "  · 너무 짧은 문장(5단어 미만) — 단서를 담을 여유가 없다.\n\n"

        "## OUTPUT\n"
        "  index: 뺄 문장의 번호 (위 목록의 [n])\n"
        "  anchor_type: a/b/c/d/e 중 어느 단서를 썼는지\n"
        "  anchor: 그 단서에 해당하는 표현 (문장 안의 실제 어구)\n"
        "  refers_to: 그 단서가 가리키는 앞 문장의 내용 (몇 번 문장의 무엇인지)\n"
        "  why_unique: 다른 자리에 넣으면 왜 어색한지 한 문장\n\n"
        '{"index": <number>,\n'
        ' "anchor_type": "<a|b|c|d|e>",\n'
        ' "anchor": "<the exact phrase that anchors it>",\n'
        ' "refers_to": "<what in the preceding sentence it points to>",\n'
        ' "why_unique": "<why no other gap works>"}'
    )
