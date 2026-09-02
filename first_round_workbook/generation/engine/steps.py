"""1회독 전용 step 생성기 (async) — pipeline.py step1~8 에서 복사/이식.

- 캐시(load/save)는 제거 → orchestrator.run_step 이 관장.
- passage_dir 제거 (로컬 캐시 없음, DB 캐시는 run_step).
- LLM 호출은 llm.client (async SDK) 사용.
- 문장 분리는 utils.text.

(pipeline 원본 step1~8 은 당분간 유지 — 나중 삭제)
"""
import re
import json
import random
import asyncio
import logging
from typing import List, Dict

from llm.client import call_claude_json_async
from llm.prompt_service import render_prompt, SYS_JSON, SYS_JSON_KR
from utils.text import split_sentences, merge_short_dialogue, _is_dialogue

logger = logging.getLogger(__name__)


# Stage 5 글의 흐름 (순서 + 삽입)의 5지 선다 - 고정
ORDER_TABLE = [
    ("① (A)-(C)-(B)", ("A", "C", "B")),
    ("② (B)-(A)-(C)", ("B", "A", "C")),
    ("③ (B)-(C)-(A)", ("B", "C", "A")),
    ("④ (C)-(A)-(B)", ("C", "A", "B")),
    ("⑤ (C)-(B)-(A)", ("C", "B", "A")),
]

# Stage 6 빈칸 추론에 사용되는 원 마커.
CIRCLE_NUMBERS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫"]
# step6 내용일치(10개) 셔플용 — 구 pipeline._CIRCLE_NUMS 와 동일(10개)
_CIRCLE_NUMS = CIRCLE_NUMBERS[:10]


# ============================================================
# STEP 1: 기본 분석 (어휘 + 번역 + 핵심문장)
#   프롬프트 템플릿($passage, $sent_count, $numbered_sentences) = DB(prompt_templates)
# ============================================================
async def step1_basic_analysis(passage: str, sentences: List[str], prompt_template: str, user_translations: List[str], full_translation: str = "") -> Dict:
    # 문장 분리·대화문 병합은 orchestrator 에서 1회 수행해 주입(sentences) — 전 step 공유
    sent_count = len(sentences)

    # 가공된 문장 리스트를 프롬프트에 명시적으로 전달
    numbered_sentences = "\n".join([f"[문장{i+1}] {s}" for i, s in enumerate(sentences)])
    prompt = render_prompt(prompt_template, passage=passage, sent_count=sent_count, numbered_sentences=numbered_sentences)

    data = await call_claude_json_async(SYS_JSON_KR, prompt, max_tokens=4096)

    # 🔒 문장 분리는 항상 주입된 결과 사용 (AI가 문장을 합치거나 쪼개는 것 방지)
    data["sentences"] = sentences

    # 번역은 항상 DB(user_translations) 사용 — LLM 번역 절대 미사용
    data["sentence_translations"] = user_translations
    data["translation"] = full_translation
    if len(user_translations) != sent_count:
        logger.warning("step1: 번역 %d줄 ≠ 영어 %d문장 (DB 값 그대로 사용함) - split_sentences(), merge_short_dialogue() 확인 요망", len(user_translations), sent_count)

    # Stage1 - step2 데이터 생성(vocab에서 영단어/유의어/뜻 추출)
    import copy

    step2_data = copy.deepcopy(data["vocab"]) # List 복사본 사용

    def data_from_vocab(get_item:str) -> List:
        v = []
        for i in range(5):
            v.append(step2_data.pop(random.randint(0, len(step2_data)-1))[get_item])

        return v

    data["test_a"] = data_from_vocab("word")
    data["test_b"] = data_from_vocab("word")
    data["test_c"] = data_from_vocab("meaning")

    return data


# ============================================================
# STEP 2~8: async 이식 예정 (stub)
# ============================================================
async def step2_order(passage: str, prompt_template: str, sentences: List[str]) -> Dict:
    """Lv.5 순서배열 + 문장삽입. 단락 위치 매칭 성공까지 최대 3회 재시도, 삽입 지문은 코드로 재구성.
    프롬프트 변수: $passage, $sentences (json 문자열).
    """
    logger.info("step2: Stage5 생성 시작")
    prompt = render_prompt(prompt_template, passage=passage, sentences=json.dumps(sentences, ensure_ascii=False))

    # 단락 위치 매칭이 성공할 때까지 최대 3회 재시도
    max_step2_retries = 3
    data = None
    for attempt in range(1, max_step2_retries + 1):
        try:
            candidate = await call_claude_json_async(SYS_JSON, prompt, max_tokens=4096)
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
                    break
                logger.warning("step2: 시도 %d/%d 위치 매칭 실패 %s → 재시도", attempt, max_step2_retries, test_positions)
            else:
                logger.warning("step2: 시도 %d/%d 단락 개수 이상 → 재시도", attempt, max_step2_retries)
        except Exception as e:
            logger.warning("step2: 시도 %d/%d 예외 %s", attempt, max_step2_retries, str(e)[:80])
        await asyncio.sleep(1)

    # if data is None:
    #     # 마지막 시도: 그래도 받아서 진행 (_generate_order_choices에서 또 검증)
    #     data = await call_claude_json_async(SYS_JSON, prompt, max_tokens=4096)
    #     logger.warning("step2: %d회 재시도 모두 실패, 마지막 결과로 진행", max_step2_retries)

    # Stage5-A: order_paragraphs → [label, text] 형태로 변환
    if data.get("order_paragraphs") and isinstance(data["order_paragraphs"][0], dict):
        data["order_paragraphs"] = [[p["label"], p["text"]] for p in data["order_paragraphs"]]

    # 문장 삽입 문제(Stage5-B): API 결과를 신뢰하지 않고 항상 원문 문장으로 재구성
    data["full_order_blocks"] = []

    for i, s in enumerate(sentences):
        data["full_order_blocks"].append([chr(ord('A') + i), s])

    insert_sentence = data["insert_sentence"]
    _build_sentence_insertion_problem(insert_sentence=insert_sentence, sentences=sentences, data=data)

    # 순서 배열 문제(Stage5-C): 순서 선지 코드 생성 + 3단락 라벨/정답 결정
    _generate_order_block_shuffled(data=data)
    _generate_order_choices(data=data, passage=passage)

    return data


async def step3_blank(passage: str, prompt_template: str) -> Dict:
    """Lv.6 빈칸 추론. 생성 후 선지 순서를 셔플(정답 위치 고정 방지).
    프롬프트 변수: $passage.
    """
    logger.info("step3: 빈칸 추론 생성 시작")
    prompt = render_prompt(prompt_template, passage=passage)
    data = await call_claude_json_async(SYS_JSON, prompt, max_tokens=3000)
    _shuffle_blank_options(data)
    return data


async def step4_topic(passage: str, prompt_template: str) -> Dict:
    """Lv.7 주제 찾기. 프롬프트 변수: $passage."""
    logger.info("step4: 주제 찾기 생성 시작")
    prompt = render_prompt(prompt_template, passage=passage)
    data = await call_claude_json_async(SYS_JSON, prompt, max_tokens=3000)
    data["topic_passage"] = passage
    return data


async def step5_grammar(passage: str, prompt_template: str, sentences: List[str], grammar_addendum: str = "") -> Dict:
    """Lv.8 어법: 8-1 괄호형(grammar_bracket) + 8-2 서술형(grammar_error) 생성.
    프롬프트 변수: $sent_count, $passage, $bracket_count, $bracket_dist_lines.
    sentences: orchestrator 에서 가공(문장분리+대화문병합)해 주입 — 전 step 공유.
    grammar_addendum: Supabase grammar_points 텍스트(service에서 주입) → 시스템프롬프트 보강.
    """
    logger.info("step5: 어법 생성 시작")
    sent_count = len(sentences)
    word_count = len(passage.split())

    # 지문 길이에 따라 최소 괄호 수 동적 계산
    if word_count <= 80:
        min_brackets = 2
    elif word_count <= 120:
        min_brackets = 3
    else:
        min_brackets = 8

    bracket_count = sent_count  # 문장당 1개 → 합계 = 총 문장 수
    # 8-1 괄호 분배: AI 호출 전에 어느 문장에 몇 개 출제할지 코드가 결정
    bracket_dist = _distribute_brackets(sent_count, bracket_count, max_per=1)
    bracket_dist_lines = "\n".join(
        f'- 문장 {i}번 ("{sentences[i][:60]}{"..." if len(sentences[i]) > 60 else ""}") → {bracket_dist[i]}개'
        for i in range(sent_count)
    )
    logger.info("step5: %d단어/%d문장 → 최소 %d개, 권장 %d개, 분배 %s",
                word_count, sent_count, min_brackets, bracket_count, bracket_dist)

    prompt = render_prompt(prompt_template, sent_count=sent_count, passage=passage,
                        bracket_count=bracket_count, bracket_dist_lines=bracket_dist_lines)

    # 시스템프롬프트에 grammar_points(어법 함정 규칙) 주입 (service에서 로드해 전달)
    sys_prompt_stage7 = SYS_JSON + ("\n\n" + grammar_addendum if grammar_addendum else "")

    async def _ai_call():
        """LLM 호출 + 8-1 triples → 괄호 지문 문자열 조립을 한 번에."""
        d = await call_claude_json_async(sys_prompt_stage7, prompt, max_tokens=4000)
        triples = d.get("grammar_bracket_passage", [])
        bracket_str, bracket_answers = _assemble_bracket_passage(triples, sentences)
        d["grammar_bracket_passage"] = bracket_str
        d["grammar_bracket_answers"] = bracket_answers
        d["grammar_bracket_count"] = len(bracket_answers)
        return d

    data = await _ai_call()

    # 🔒 8-2 grammar_error_passage 검증 — 문장 수 불일치 시 1회 재생성
    err_text = data.get("grammar_error_passage", "")
    if err_text and len(split_sentences(err_text)) != sent_count:
        logger.warning(
            "step5: grammar_error 문장 수 %d != %d → 재시도",
            len(split_sentences(err_text)), sent_count
            )
        data = await _ai_call()

    orig_len = len(re.sub(r'\s+', '', passage))
    err_gen = data.get("grammar_error_passage", "")
    if err_gen:
        err_sents_list = split_sentences(err_gen)
        if len(err_sents_list) > sent_count:
            logger.warning("step5: grammar_error %d문장 > %d → 트림", len(err_sents_list), sent_count)
            data["grammar_error_passage"] = " ".join(err_sents_list[:sent_count]).strip()
        elif len(err_sents_list) < sent_count:
            logger.warning("step5: grammar_error %d문장 < %d → 원문 사용", len(err_sents_list), sent_count)
            data["grammar_error_passage"] = passage

        # 길이 체크: 원문 대비 20% 초과면 내용 추가된 것 → 재시도
        err_clean_len = len(re.sub(r'\s+', '', data.get("grammar_error_passage", "")))
        if orig_len > 0 and err_clean_len > orig_len * 1.2:
            logger.warning("step5: grammar_error 길이 %d >> %d (>20%%) → 재시도", err_clean_len, orig_len)
            retry_data = await _ai_call()
            retry_text = retry_data.get("grammar_error_passage", "")
            retry_clean_len = len(re.sub(r'\s+', '', retry_text))
            if retry_clean_len <= orig_len * 1.2:
                data["grammar_error_passage"] = retry_text
                data["grammar_error_answers"] = retry_data.get(
                    "grammar_error_answers", data.get("grammar_error_answers", []))
                logger.info("step5: grammar_error 재시도 성공 (길이 %d)", retry_clean_len)
            else:
                logger.warning("step5: grammar_error 재시도도 김 (%d) → 기존 유지", retry_clean_len)

    # 서술형: 무의미 항목 제거 (error == original 인 경우)
    raw_errors = data.get("grammar_error_answers", [])
    valid_errors = [a for a in raw_errors
                    if isinstance(a, dict) and a.get("error", "").strip() != a.get("original", "").strip()]
    if len(valid_errors) != len(raw_errors):
        logger.warning("step5: 서술형 무의미 항목 %d개 제거", len(raw_errors) - len(valid_errors))
        for i, a in enumerate(valid_errors):
            a["num"] = i + 1
        data["grammar_error_answers"] = valid_errors

    # 서술형 error_count 를 실제 answers 길이로 보정
    data["grammar_error_count"] = len(data.get("grammar_error_answers", []))

    # 8-2 오류 0개면 재시도 (원문 그대로 나온 경우)
    if data["grammar_error_count"] == 0:
        logger.warning("step5: 8-2 오류 0개(원문 그대로) → 재시도")
        for attempt in range(1, 4):
            retry_data = await _ai_call()
            retry_errors = retry_data.get("grammar_error_answers", [])
            if len(retry_errors) >= 3:
                data["grammar_error_passage"] = retry_data.get(
                    "grammar_error_passage", data.get("grammar_error_passage", ""))
                data["grammar_error_answers"] = retry_errors
                data["grammar_error_count"] = len(retry_errors)
                logger.info("step5: 8-2 재시도 성공 (%d개 오류)", len(retry_errors))
                break
            logger.info("step5: 8-2 재시도 %d 실패 (%d개)", attempt, len(retry_errors))

    # 8-1 괄호 형태론 검증: 둘 다 정답인 괄호를 올바른 출제로 교체 (형태론 로직 — 별도 헬퍼)
    _fix_ambiguous_brackets(data, passage, sentences)

    return data


async def step6_vocab_content(passage: str, prompt_template: str) -> Dict:
    """Lv.9 어휘심화(Part A 괄호/Part B 유의어) + 내용일치(kr/en). 프롬프트 변수: $passage."""
    logger.info("step6: 어휘/내용일치 생성 시작")
    prompt = render_prompt(prompt_template, passage=passage)

    data = await call_claude_json_async(SYS_JSON_KR, prompt, max_tokens=6000)

    # 내용일치 10개 미만이면 1회 재시도
    kr_count = len(data.get("content_match_kr", []))
    en_count = len(data.get("content_match_en", []))
    if kr_count < 10 or en_count < 10:
        logger.warning("step6: content_match 부족 (kr=%d, en=%d) → 재시도", kr_count, en_count)
        data2 = await call_claude_json_async(SYS_JSON_KR, prompt, max_tokens=6000)
        if len(data2.get("content_match_kr", [])) >= kr_count:
            data["content_match_kr"] = data2.get("content_match_kr", data.get("content_match_kr", []))
            data["content_match_kr_answer"] = data2.get("content_match_kr_answer", data.get("content_match_kr_answer", []))
        if len(data2.get("content_match_en", [])) >= en_count:
            data["content_match_en"] = data2.get("content_match_en", data.get("content_match_en", []))
            data["content_match_en_answer"] = data2.get("content_match_en_answer", data.get("content_match_en_answer", []))
    (
        data["content_match_kr"],
        data["content_match_kr_answer"],
        data["content_match_kr_wrong"],
        data["content_match_kr_wrong_trans"]) = _shuffle_content_match(
            data.get("content_match_kr", []),
            data.get("content_match_kr_answer", []),
            data.get("content_match_kr_wrong", []),
            data.get("content_match_kr_wrong_trans", []),
    )

    # Part B choices 안에서 정답 위치 랜덤화
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
    (
        data["content_match_en"],
        data["content_match_en_answer"],
        data["content_match_en_wrong"],
        data["content_match_en_wrong_trans"]) = _shuffle_content_match(
            data.get("content_match_en", []),
            data.get("content_match_en_answer", []),
            data.get("content_match_en_wrong", []),
            data.get("content_match_en_wrong_trans", []),
    )

    # 9-1 Part A 5개 미만이면 재시도 (최소 5개 강제, 최대 4회)
    actual_parta = data.get("vocab_parta_answers", [])
    if len(actual_parta) < 5:
        logger.warning("step6: Part A %d개 < 5개 최소 기준 → 재시도", len(actual_parta))
        for attempt in range(1, 5):
            data2 = await call_claude_json_async(SYS_JSON_KR, prompt, max_tokens=6000)
            parta2 = data2.get("vocab_parta_answers", [])
            if len(parta2) >= 5:
                data["vocab_parta_answers"] = parta2
                data["vocab_advanced_passage"] = data2.get("vocab_advanced_passage", data.get("vocab_advanced_passage", ""))
                actual_parta = parta2
                logger.info("step6: Part A 재시도 성공 → %d개", len(parta2))
                break
            logger.info("step6: Part A 재시도 %d 실패 (%d개)", attempt, len(parta2))
        if len(actual_parta) < 5:
            logger.warning("step6: Part A 최종 %d개 - 재시도 모두 실패", len(actual_parta))
    data["vocab_parta_count"] = len(actual_parta)

    # 9-1 Part A 정답 좌우 랜덤 shuffle (각 괄호 개별 50% 확률)
    va_passage = data.get("vocab_advanced_passage", "")
    va_answers = data.get("vocab_parta_answers", [])
    if va_passage and va_answers:
        result_va = va_passage
        for ans in va_answers:
            num = ans.get("num", "")
            if random.random() < 0.5:
                pat = re.compile(r'\(' + str(num) + r'\)\[([^\]]+)\]')

                def do_swap_va(m, n=num):
                    parts = [p.strip() for p in m.group(1).split(' / ')]
                    return f'({n})[{parts[1]} / {parts[0]}]' if len(parts) == 2 else m.group(0)

                result_va = pat.sub(do_swap_va, result_va)
        data["vocab_advanced_passage"] = result_va

    return data


async def step7_writing(sentences: List[str], sentence_translations: List[str]) -> Dict:
    """Lv.10 영작(스크램블). LLM 없음 — 문장을 단어 단위로 섞어 배열 문제 생성.
    (프롬프트 불필요)
    """
    logger.info("step7: 영작 생성 시작")
    # 대화문 여부 확인
    is_dialogue = _is_dialogue(sentences)

    # 한국어 문장: sentence_translations 그대로 사용 (Step1에서 sentences와 개수 맞춰짐)
    kr_sentences = (sentence_translations or [])[:len(sentences)]
    while len(kr_sentences) < len(sentences):
        kr_sentences.append(f"문장 {len(kr_sentences)+1}")

    writing_items = []
    for i, eng in enumerate(sentences):
        words = eng.split()
        kr = kr_sentences[i] if i < len(kr_sentences) else f"문장 {i+1}"

        # 대화문에서 6단어 이하 문장: scramble 안 하고 원문 그대로
        if is_dialogue and len(words) <= 6:
            writing_items.append({"korean": kr, "scrambled": eng, "answer": eng})
            continue

        if is_dialogue:
            # 대화문: 합쳐진 문장을 원래 문장 단위로 분리, 6단어 이하는 원문 그대로
            sub_sents = re.split(r'(?<=[.!?])\s+', eng)
            scramble_parts = []
            speaker_prefix = ""
            speaker_match = re.match(r'^([A-Z][a-z]+\s*:\s*)', eng)
            if speaker_match:
                speaker_prefix = speaker_match.group(1)

            for si, sub in enumerate(sub_sents):
                sub_words = sub.split()
                if len(sub_words) <= 6:
                    scramble_parts.append(sub)
                else:
                    sub_prefix = ""
                    sub_text = sub
                    sub_speaker = re.match(r'^([A-Z][a-z]+\s*:\s*)', sub)
                    if sub_speaker:
                        sub_prefix = sub_speaker.group(1)
                        sub_text = sub[len(sub_prefix):]

                    proc = sub_text.split()
                    if proc and proc[0][0].isupper() and proc[0] not in ['I', 'I,']:
                        if not (len(proc[0]) > 1 and proc[0][1:].islower() and any(c.isupper() for c in proc[0])):
                            proc[0] = proc[0][0].lower() + proc[0][1:]
                    if proc and proc[-1].endswith(('.', '!', '?')):
                        proc[-1] = proc[-1][:-1]
                    shuffled = proc.copy()
                    random.shuffle(shuffled)
                    scramble_parts.append(sub_prefix + ' / '.join(shuffled))

            scrambled = ' '.join(scramble_parts)
            writing_items.append({"korean": kr, "scrambled": scrambled, "answer": eng})
            continue

        # 비대화문: 단어 단위 셔플
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

        writing_items.append({"korean": kr, "scrambled": scrambled, "answer": eng})

    return {"writing_items": writing_items}

def _generate_order_choices(data: Dict, passage: str = ""):
    """order_paragraphs 3단락 원문 위치 파악 → ORDER_TABLE 5개 중 정답 무작위 선택 → 라벨 역산."""
    paras = data.get("order_paragraphs", [])
    if len(paras) != 3:
        raise ValueError(f"STAGE 5 | 단락 개수 이상(3개 필요): {len(paras)}개")

    norm_passage = re.sub(r'\s+', ' ', passage)

    def _find_pos(text):
        snippet = re.sub(r'\s+', ' ', text.strip())
        words = snippet.split()
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

    original_order = sorted(range(3), key=lambda i: positions[i])
    answer_str, correct = random.choice(ORDER_TABLE)

    labels = [None] * 3
    for k in range(3):
        labels[original_order[k]] = correct[k]

    new_paras = [[labels[i], paras[i][1]] for i in range(3)]
    new_paras.sort(key=lambda x: x[0])

    data["order_paragraphs"] = new_paras
    data["order_answer"] = answer_str


def _generate_order_block_shuffled(data: Dict):
    """full_order_blocks 전체 문장 배열(심화) 셔플 + 정답 순서 기록."""
    blocks = data.get("full_order_blocks", [])
    if len(blocks) >= 2:
        n = len(blocks)
        alpha = [chr(65 + i) for i in range(n)]
        random.shuffle(alpha)
        new_blocks = [[alpha[i], blocks[i][1]] for i in range(n)]
        correct_order = "→".join([f"({alpha[i]})" for i in range(n)])
        data["full_order_answer"] = correct_order
        new_blocks.sort(key=lambda x: x[0])
        data["full_order_blocks"] = new_blocks


def _build_sentence_insertion_problem(insert_sentence: str, sentences: List[str], data: Dict):
    """
    문장 삽입 문제(Stage5-B) 재구성

    원문에서 insert_sentence 1개를 빼고, 남은 문장 사이에 ①~⑤ 마커를 배치.
    뺀 문장의 원래 자리가 정답. data['insert_answer'], data['insert_passage'] 를 채운다.
    (원문 문장으로 직접 조립하므로 지문 축약/변형이 구조적으로 불가능)
    """

    markers = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

    sentences = list(sentences)

    if len(insert_sentence) == 0:
        logger.warning("클로드의 응답 값 중 하나인 핵심 문장(insert_sentence)가 없음. step2의 처음인 claude call 재시도 필요")
        return

    # 1. 삭제할 문장 찾기 — 대화문/일반 지문을 _is_dialogue 로 분기.
    target = insert_sentence.strip()
    if _is_dialogue(sentences):
        # 대화문: insert_sentence 가 한 화자의 turn(여러 문장 합본)일 수 있음.
        #        → 각 문장이 insert 안에 '포함'되면 삭제(여러 문장 매칭 허용).
        matched_indexes = [i for i, s in enumerate(sentences) if s.strip() in target]
    else:
        # 일반 지문: insert 는 단일 문장 → 공백 무시 '정확' 매칭(오매칭 방지).
        matched_indexes = [i for i, s in enumerate(sentences) if s.strip() == target]
        if not matched_indexes:                       # 정확 실패 시 '포함'으로 보조
            matched_indexes = [i for i, s in enumerate(sentences) if s.strip() in target]

    if matched_indexes:
        # 경우 1: 매칭 성공 — 매칭된 문장들을 모두 삭제.
        #         정답 위치(삽입 지점)는 첫 매칭 문장의 인덱스.
        insertion_index = matched_indexes[0]
        answer_sentence = insert_sentence.strip()
        for index in reversed(matched_indexes):   # 뒤에서부터 삭제해야 인덱스가 밀리지 않음
            del sentences[index]
    else:
        # 경우 2: 매칭 실패(LLM 패러프레이즈·특수문자 등) — 크래시 방지 fallback.
        #         중앙 문장을 삽입 대상으로 삼아 삭제.(정답도 그 문장으로 맞춤)
        insertion_index = len(sentences) // 2
        answer_sentence = sentences[insertion_index]   # 삭제 전에 실제 문장 확보
        logger.warning("step2: insert_sentence 원문 매칭 실패 → %d번째 문장으로 대체", insertion_index)
        del sentences[insertion_index]

    # 2. 남은 문장들 앞에 마커(①②③…) 붙이기. 마지막 문장 뒤에는 마커 하나 더.
    for i in range(len(sentences)):
        sentences[i] = f"( {markers[i]} ) {sentences[i]}"
        if i == (len(sentences) - 1):
            sentences[i] = f"{sentences[i]} ( {markers[i + 1]} )"

    # 3. 지문 문자열로 합치고 정답 지정. (정답 마커 = 삽입 지점 인덱스의 마커)
    data["insert_passage"] = " ".join(sentences)
    data["insert_answer"] = f"{markers[insertion_index]} {answer_sentence}"


    ################################################################
    # insert_index = -1
    # for index, sentence in enumerate(original_sentences):
    #     if re.sub(r'\s+', ' ', sentence.strip()) == insert_normalized:
    #         insert_index = index
    #         break
    # if insert_index == -1:
    #     insert_prefix = ' '.join(insert_normalized.split()[:8])
    #     for index, sentence in enumerate(original_sentences):
    #         if insert_prefix in re.sub(r'\s+', ' ', sentence.strip()):
    #             insert_index = index
    #             break
    # if insert_index == -1:
    #     insert_index = len(original_sentences) // 2
    #     data["insert_sentence"] = original_sentences[insert_index]
    #     logger.warning("step2: insert_sentence 원문에서 못 찾음 → %d번째 문장 사용", insert_index)

    # remaining_sentences = [s for i, s in enumerate(original_sentences) if i != insert_index]
    # remaining_count = len(remaining_sentences)
    # markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
    # correct_position = min(insert_index, remaining_count)

    # # 마커 5개를 남은 문장 사이에 균등 배치
    # if remaining_count >= 5:
    #     interval = remaining_count / 5
    #     marker_positions = [int(interval * (i + 0.5)) for i in range(5)]
    # else:
    #     marker_positions = list(range(min(5, remaining_count + 1)))

    # # 정답 위치가 반드시 포함되도록 조정
    # if correct_position not in marker_positions and marker_positions:
    #     closest_index = min(range(len(marker_positions)),
    #                         key=lambda x: abs(marker_positions[x] - correct_position))
    #     marker_positions[closest_index] = correct_position

    # marker_positions = sorted(set(marker_positions))
    # unused_positions = [i for i in range(remaining_count + 1) if i not in marker_positions]
    # while len(marker_positions) < 5 and unused_positions:
    #     marker_positions.append(unused_positions.pop(0))
    # marker_positions.sort()

    # answer_index = marker_positions.index(correct_position) if correct_position in marker_positions else 2
    # data["insert_answer"] = f"{answer_index + 1}"

    # # 지문 재구성: 남은 문장 사이에 마커 삽입
    # rebuilt_parts = []
    # marker_positions_set = set(marker_positions)
    # marker_cursor = 0
    # for position in range(remaining_count + 1):
    #     if position in marker_positions_set and marker_cursor < 5:
    #         rebuilt_parts.append(markers[marker_cursor])
    #         marker_cursor += 1
    #     if position < remaining_count:
    #         rebuilt_parts.append(remaining_sentences[position])
    # while marker_cursor < 5:
    #     rebuilt_parts.append(markers[marker_cursor])
    #     marker_cursor += 1

    # insert_passage = " ".join(rebuilt_parts).strip()

    # # ( ④ )와 ( ⑤ )가 연속으로 나오면 ( ⑤ ) 제거
    # insert_passage = re.sub(r'\(\s*[④④]\s*\)(.{0,30})\(\s*[⑤⑤]\s*\)', r'( ④ )\g<1>', insert_passage)

    # data["insert_passage"] = insert_passage
    # logger.info("step2: 삽입 지문 재구성 완료 (정답 %s번, 마커 위치 %s)",
    #             data["insert_answer"], marker_positions)


def _shuffle_blank_options(data: Dict) -> None:
    """빈칸 선지 순서를 랜덤 셔플(정답 위치 고정 방지) + 정답/오답/오답해석 라벨 재매핑."""
    options = data.get("blank_options", [])
    if not (options and len(options) >= 2):
        return

    correct_labels = set(data.get("blank_correct", []))
    wrong_labels = set(data.get("blank_wrong", []))
    ai_wrong_labels = data.get("blank_wrong", [])
    ai_wrong_translations = data.get("blank_wrong_translation", [])

    # 번호 제거한 선지 텍스트 (셔플 전 순서)
    option_texts = [re.sub(r'^[①-⑫]\s*', '', opt).strip() for opt in options]

    def _text_of(label: str) -> str | None:
        if label in CIRCLE_NUMBERS and CIRCLE_NUMBERS.index(label) < len(option_texts):
            return option_texts[CIRCLE_NUMBERS.index(label)]
        return None

    correct_texts = [t for t in (_text_of(c) for c in correct_labels) if t is not None]
    wrong_texts = [t for t in (_text_of(w) for w in wrong_labels) if t is not None]

    # 셔플
    shuffled_texts = option_texts.copy()
    random.shuffle(shuffled_texts)

    # 새 번호 부여 + 정답/오답 재매핑
    new_options, new_correct, new_wrong = [], [], []
    for index, text in enumerate(shuffled_texts):
        label = CIRCLE_NUMBERS[index]
        new_options.append(f"{label} {text}")
        if text in correct_texts:
            new_correct.append(label)
        elif text in wrong_texts:
            new_wrong.append(label)

    # 오답 한글 해석 → 셔플된 라벨로 매핑
    translation_by_label = {}
    for old_label, translation in zip(ai_wrong_labels, ai_wrong_translations):
        text = _text_of(old_label)
        if text is None:
            continue
        new_label = CIRCLE_NUMBERS[shuffled_texts.index(text)]
        translation_by_label[new_label] = re.sub(r'^[①-⑫]\s*', '', translation).strip()

    new_wrong_translations = [
        f"{label} {translation_by_label[label]}"
        for label in new_wrong
        if label in translation_by_label
    ]

    data["blank_options"] = new_options
    data["blank_correct"] = new_correct
    data["blank_wrong"] = new_wrong
    data["blank_wrong_translation"] = new_wrong_translations
    logger.info("step3: 빈칸 선지 셔플 완료 (정답 %s / 오답해석 %d건 매핑)",
                new_correct, len(new_wrong_translations))


def _distribute_brackets(sent_count: int, total: int, max_per: int = 2) -> List[int]:
    """각 문장에 0~max_per개 무작위 분배. counts[i] 합계 = min(total, sent_count*max_per)."""
    counts = [0] * sent_count
    pool = list(range(sent_count)) * max_per
    random.shuffle(pool)
    for i in pool[:min(total, len(pool))]:
        counts[i] += 1
    return counts


def _fix_ambiguous_brackets(data: Dict, passage: str, sentences: List[str]) -> None:
    """8-1 괄호 형태론 검증: 둘 다 정답이 되는 괄호(help+to/ing·지각동사·병렬 등)를
    올바른 출제로 교체하거나 제거. data['grammar_bracket_*'] 를 in-place 수정.
    (pipeline.py:1365-1966 형태론 블록 충실 복사 — 로직 무수정)
    """
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
                    logger.info(f"  🚫 8-1 괄호({num_str}) 제거: [{a_raw} / {b_raw}] (둘다가능동사 뒤 to/ing)")
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
                logger.info(f"  🔧 8-1 괄호({num}) 교체: [{correct} / {wrong}] ({reason})")
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
                    logger.info(f"  🔢 괄호 재번호 완료: {list(renumber_map.items())}")
            
            logger.info(f"  ✅ 둘 다 정답 괄호 {len(fixed_nums)}개 처리 완료 (남은 괄호: {actual_after}개)")
    

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
                logger.info(f"  WARNING: grammar_error_passage appears duplicated, trimming...")
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
                    logger.info(f"  🚫 지시대명사 복원({num_str}): [{bracket_content}] → those")
                elif 'these that' in passage.lower() or 'these who' in passage.lower():
                    final_bp = re.sub(r'\(' + num_str + r'\)\[[^\]]+\]', 'these', final_bp)
                    removed_demo.append(int(num_str))
                    logger.info(f"  🚫 지시대명사 복원({num_str}): [{bracket_content}] → these")
        if removed_demo:
            data["grammar_bracket_passage"] = final_bp
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_demo]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp))
            logger.info(f"  ✅ 지시대명사 {len(removed_demo)}개 복원 완료")

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
                    logger.info(f"  🚫 '바로 옆' 자리 괄호({num_str}) 제거: 앞 단어='{check_word}' → '{correct_word}'")
        
        if removed_na:
            data["grammar_bracket_passage"] = final_bp_na
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_na]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_na))
            logger.info(f"  ✅ '바로 옆' 자리 {len(removed_na)}개 제거 완료")

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
                    logger.info(f"  🚫 '바로 옆' 자리 괄호({num_str}) 제거: 앞 단어='{check_word}' → '{correct_word}'")
        
        if removed_na:
            data["grammar_bracket_passage"] = final_bp_na
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_na]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_na))
            logger.info(f"  ✅ '바로 옆' 자리 {len(removed_na)}개 제거 완료")

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
                    logger.info(f"  🚫 의미 차이 괄호({num_str}) 제거: {reason} → '{correct_word}'")
        if removed_neg:
            data["grammar_bracket_passage"] = final_bp_neg
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_neg]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_neg))
            logger.info(f"  ✅ 의미 차이 괄호 {len(removed_neg)}개 제거 완료")

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
                    logger.info(f"  🚫 추가 차단 괄호({num_str}) 제거: {reason} → '{correct_word}'")
        if removed_extra:
            data["grammar_bracket_passage"] = final_bp_extra
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_extra]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_extra))
            logger.info(f"  ✅ 추가 차단 괄호 {len(removed_extra)}개 제거 완료")

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
                    logger.info(f"  🚫 '뒤에 by' 능/수동 괄호({num_str}) 제거: 뒤='{after_text[:30]}' → '{correct_word}'")
        if removed_by:
            data["grammar_bracket_passage"] = final_bp_by
            data["grammar_bracket_answers"] = [a for a in data.get("grammar_bracket_answers", []) if a.get("num") not in removed_by]
            data["grammar_bracket_count"] = len(re.findall(r'\(\d+\)\[', final_bp_by))
            logger.info(f"  ✅ '뒤에 by' 능/수동 {len(removed_by)}개 제거 완료")

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
                    logger.info(f"  🚫 whom/who 괄호 제거({num_str})")
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


def _assemble_bracket_passage(triples, sentences: List[str]):
    """AI가 반환한 [[원문장, 정답, 오답], ...] 를 grammar_bracket_passage 문자열 +
    grammar_bracket_answers 리스트로 변환. 50% 확률로 정답 좌우 swap.

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
        logger.info("step5: 괄호 추가한 문장 확인: %s", bracketed[idx])
        answers.append({"num": n, "answer": ans, "wrong": wrong})

    return " ".join(bracketed), answers


def _shuffle_content_match(items, answers, ai_wrong, ai_wrong_trans):
    """content_match_{kr|en}을 셔플하면서 answer/wrong/wrong_trans 라벨을 동시 동기화.

    Returns: (new_items, new_answer, new_wrong, new_wrong_trans)
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
