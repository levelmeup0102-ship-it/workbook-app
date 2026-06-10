"""
seosul/prompts.py
유형별 출제 프롬프트. seosul_types(지시문/조건)와 grammar_points(어법 화이트리스트)를
주입하여 LLM이 '즉석 판단'이 아니라 'DB 근거'로만 출제하게 만든다.
모든 프롬프트는 검증기가 파싱할 수 있도록 엄격한 JSON만 출력하도록 요구한다.
"""
import json
from typing import List, Dict

_COMMON = """너는 한국 고등학교 영어 내신 서술형 출제 전문가다.
아래 지문(문장별 번호 부여)에서 지정된 문장만 사용해 문제를 만든다.
반드시 JSON만 출력한다. 설명/마크다운/코드펜스 금지."""

def _numbered(sentences: List[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def prompt_SA(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SA] {spec['instruction']}
조건: 보기 단어를 모두 사용. 어형 변형 및 단어 중복 가능.
대상 문장(이 문장들에서만 빈칸 2곳을 만든다): {target_idx}

지문:
{_numbered(sentences)}

출력 JSON 스키마:
{{"type":"SA","answers":{{"A":"<원문에서 떼어낸 어구>","B":"..."}},
 "bogi":["<섞인 단어들>"],
 "blanks":{{"A":{{"sent":<번호>,"tpl":"<해당문장에서 정답자리를 {{{{A}}}}로 치환>","original":"<원문문장>"}},
            "B":{{"sent":<번호>,"tpl":"...","original":"..."}}}}}}
규칙: 정답 A,B는 서로 다른 문장에 위치. bogi는 A,B 정답 토큰의 어간만 모아 섞되 중복은 1개로 표기."""


def prompt_SC(sentences, spec) -> str:
    return f"""{_COMMON}

[유형 SC] {spec['instruction']}
조건: 보기 어구를 변형 없이 모두 한 번씩만 배열. (A)와 (B)는 요약문의 서로 다른 절에 둔다.
구조는 매번 달라야 한다(직전 사용 구조 회피). 후보: the 비교급~the 비교급 / Those who~ / Not only~but / It is~that / By v-ing / Despite~ / so~that.

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SC","summary":"<(A)자리는 {{{{A}}}}, (B)자리는 {{{{B}}}} 인 2문장 요약>",
 "answers":{{"A":"...","B":"..."}},"bogi":["..."],"structure":"<사용한 구조명>"}}"""


def prompt_SD(sentences, target_idx, allowed_gp: List[dict]) -> str:
    wl = "\n".join(f"- #{g['id']} [{g['category']}] {g['name']} :: 함정={g.get('trap_warning','')}"
                   for g in allowed_gp)
    return f"""{_COMMON}

[유형 SD] 다음 지문에서 어법상 틀린 곳을 만든다. 밑줄/번호 표시 없음.
대상 문장(여기에만 오류 삽입, 빈칸 문장 제외): {target_idx}
반드시 아래 화이트리스트의 포인트로만 출제. 목록에 없는 어법은 금지(즉석 판단 금지).
서로 다른 category로, 주어-동사는 수식어구로 거리를 벌려라. 관사/어휘혼동/둘 다 맞는 문법 금지.

화이트리스트:
{wl}

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SD","errors":[
  {{"sent":<번호>,"wrong":"<오류형>","right":"<정답형>","gp_id":<화이트리스트 id>,"why":"<근거 한 줄>"}}
]}}"""


def prompt_SE(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SE] {spec['instruction']}
핵심 주장·소재 단어 위주로 {len(target_idx)}곳을 빈칸. 보기에는 '원형'을 주고 학생이 품사를 변형하게 한다.
대상 문장: {target_idx}

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SE","bogi":["<원형들>"],
 "blanks":[{{"label":"C","sent":<번호>,"base":"<원형>","answer":"<변형형>","note":"<품사 변화>"}}]}}
규칙: answer는 base에서 품사 형태가 바뀐 형태여야 하며 base와 달라야 한다."""
