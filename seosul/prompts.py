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
규칙: 정답 A,B는 서로 다른 문장/절에 위치.
- 정답 A,B는 '원문에 실제로 있는 연속된 어구를 그대로' 떼어내라. 한 단어도 바꾸지 마라(빈칸에 정답을 도로 넣으면 원문과 글자까지 똑같아야 한다).
- bogi에는 정답에 쓰인 모든 단어의 '원형(사전 기본형)'을 넣어라. 관사(a/the)·전치사(in/to/of 등) 같은 기능어도 정답에 쓰였다면 빠짐없이 bogi에 포함하라.
- 정답은 오직 bogi의 단어들로만 구성하라. bogi에 없는 새 단어(특히 관사·전치사)를 정답에 추가하지 마라.
- 어형 변형은 허용되나, come→came 같은 '불규칙' 변형이 필요한 자리는 빈칸으로 고르지 마라(규칙 변형만).
- (A)와 (B)는 본문에서 '서로 겹치지 않는 다른 어구'여야 한다. 한쪽이 다른쪽에 포함되면 안 된다(예: (A)='over food a mate', (B)='a mate' 금지).
- ★ 난이도: 두 빈칸 (A)+(B)의 정답에 쓰인 단어 수 '합계'가 최소 10개, 최대 20개가 되도록 '충분히 긴 연속 어구'를 골라라. 너무 짧으면(합 10개 미만) 더 긴 어구로 다시 잡아라. (보기 단어가 10~20개가 되어야 함)"""


def prompt_SC(sentences, spec) -> str:
    return f"""{_COMMON}

[유형 SC] {spec['instruction']}
조건: 보기 어구를 변형 없이 모두 한 번씩만 배열. (A)와 (B)는 요약문의 서로 다른 절에 둔다.
- 같은 단어를 정답에서 두 번 쓰면, 보기에도 그 단어를 '그 횟수만큼' 똑같이 넣어라(because/and/the 등 중복 주의). 가급적 같은 단어를 반복하지 않는 간결한 요약으로 만들어라.
구조는 매번 달라야 한다(직전 사용 구조 회피). 후보: the 비교급~the 비교급 / Those who~ / Not only~but / It is~that / By v-ing / Despite~ / so~that.

지문:
{_numbered(sentences)}

출력 JSON (모든 키 필수 — bogi를 절대 빠뜨리지 마라):
{{"type":"SC","summary":"<(A)자리는 {{{{A}}}}, (B)자리는 {{{{B}}}} 인 2문장 요약>",
 "answers":{{"A":"...","B":"..."}},"bogi":["...","..."],"structure":"<사용한 구조명>"}}
주의: summary의 빈칸 자리에는 '(A)','(B)' 같은 라벨이나 밑줄(___)을 쓰지 말고, 오직 {{{{A}}}} {{{{B}}}} placeholder만 정확히 한 번씩 넣어라(라벨과 placeholder를 같이 쓰면 빈칸이 두 번 찍힌다)."""


def prompt_SD(sentences, target_idx, allowed_gp: List[dict]) -> str:
    if allowed_gp:
        wl = "\n".join(f"- [{g.get('category','')}] {g['name']} :: 함정={g.get('trap_warning','')}"
                       for g in allowed_gp[:40])
    else:
        wl = ("- [일치] 주어-동사 거리 수일치\n"
              "- [관계사] 관계대명사 vs 관계부사 자리\n"
              "- [관계사] what/that/which 구분, the sense/fact + 동격 that\n"
              "- [준동사] 능동/수동 분사 (v-ing vs p.p.)\n"
              "- [준동사] to-v vs v-ing, 전치사+동명사(in v-ing)\n"
              "- [형용사/부사] 형용사 자리 vs 부사 자리\n"
              "- [태] 능동 vs 수동\n- [도치] 도치 구문 수일치")
    return f"""{_COMMON}

[유형 SD] 다음 지문에서 어법상 틀린 곳을 만든다. 밑줄/번호 표시 없음.
대상 문장(여기에만 오류 삽입, 빈칸 문장 제외): {target_idx}
★ '깨끗한' 어법 오류만 2곳 또는 3곳 만들어라(각각 다른 문장·다른 category).
   3개째가 억지스럽거나 금지 유형밖에 안 남으면 '2개만' 만들어라(억지로 3개 채우지 마라).
아래 '권장 유형'에서만 골라 출제하고, 각 오류에 그 category를 반드시 적어라.
서로 다른 category로 출제하고, 주어-동사는 수식어구로 거리를 벌려라.

★ 절대 금지(출제하면 안 됨): 관사(a/an/the) 오류, 어휘·철자 혼동(affect/effect, affecting/unaffected, rise/raise, lie/lay 등 '뜻이 다른 단어 바꿔치기'), 둘 다 맞는 문법(지각/사역/help+원형 vs to V 등). 이런 건 만들지 마라. 특히 단어를 '반대뜻/다른뜻 단어'로 바꾸는 건 어법이 아니라 어휘라서 금지다.

권장 유형:
{wl}

지문:
{_numbered(sentences)}

출력 JSON (gp_id는 쓰지 말 것, category로 표기):
{{"type":"SD","errors":[
  {{"sent":<번호>,"wrong":"<바뀌는 한 단어만(문장 전체 금지)>","right":"<고친 한 단어>","category":"<권장유형 category>","why":"<무엇을 무엇으로 고치는지 포함한 한 줄 설명. 예: 'after 뒤이므로 engage→engaging'>"}}
]}}
주의: wrong/right에는 '실제로 바뀌는 그 단어'만 넣어라(문장이나 긴 어구를 통째로 넣지 마라). 설명(why)에 교정 근거와 '원형→고친형'을 간단히 적어라."""


def prompt_SE(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SE] {spec['instruction']}
핵심 주장·소재 단어 위주로 {len(target_idx)}곳을 빈칸. 보기에는 '원형'을 주고 학생이 품사를 변형하게 한다.
대상 문장: {target_idx}

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SE","bogi":["<원형들 + 오답용 원형들(섞어서)>"],
 "blanks":[{{"label":"C","sent":<번호>,"base":"<원형>","answer":"<변형형>","note":"<품사 변화>"}}]}}
규칙:
- answer는 base에서 품사 형태가 바뀐 형태여야 하며 base와 달라야 한다.
- ★ 오답(distractor): bogi에는 정답으로 쓰는 원형들 외에, '지문에 실제로 등장하는 다른 내용어'(정답과 무관한 단어)를 원형으로 추가해, 보기 총 개수가 빈칸 개수의 약 1.5배가 되게 하라(예: 빈칸 4개 → 보기 6개). 오답 단어도 정답들과 똑같이 '원형'으로 적어 섞어라(어느 게 정답인지 티 나지 않게)."""
