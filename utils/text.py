"""텍스트 처리 유틸 — 영어 지문 문장 분리/대화문 병합.

pipeline.py 원본(split_sentences / _is_dialogue / _merge_short_dialogue)에서 복사.
(pipeline 원본은 당분간 유지 — 나중에 이 모듈로 통합 예정)

    from utils.text import split_sentences, merge_short_dialogue
"""
import re
from typing import List, Dict


def split_sentences(text: str) -> List[str]:
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
        r'(?<=[.!?])\s+(?=[“”"]?[A-Z])|(?<=[.!?][“”"])\s+(?=[“”"]?[A-Z])',
        protected
    ) if s.strip()]

    # 3.5단계: 따옴표 내부가 너무 길면 추가 slice (짝 보정 X)
    def inner_slice_long_quote(sentence: str, min_words: int = 6) -> list[str]:
        if '§QSEP§' not in sentence:
            return [sentence]

        parts = sentence.split('§QSEP§')

        def wc(seg: str) -> int:
            cleaned = seg.strip().strip('"“”')  # §DOT§는 일부러 복원 X
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


def _is_dialogue(sentences: List[str]) -> bool:
    """대화문 지문인지 판별: 문장의 20% 이상이 '이름:' 패턴으로 시작하면 대화문"""
    if len(sentences) < 3:
        return False
    speaker_count = sum(1 for s in sentences if re.match(r'^[A-Z][a-z]+\s*:', s))
    return speaker_count / len(sentences) >= 0.2


def merge_short_dialogue(sentences: list, min_words: int = 6) -> list:
    """대화문에서 짧은 문장(6단어 이하) 합치기 (구 _merge_short_dialogue)
    규칙:
    - 같은 화자의 다음 문장과 합침
    - 화자가 바뀌면 합치지 않음
    - 대화문이 아니면 원본 그대로 반환
    """
    if not _is_dialogue(sentences) or len(sentences) < 2:
        return sentences

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

    return final
