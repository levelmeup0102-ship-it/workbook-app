"""
seosul/prompts.py
유형별 출제 프롬프트. seosul_types(지시문/조건)와 grammar_points(어법 화이트리스트)를
주입하여 LLM이 '즉석 판단'이 아니라 'DB 근거'로만 출제하게 만든다.
모든 프롬프트는 검증기가 파싱할 수 있도록 엄격한 JSON만 출력하도록 요구한다.
"""
import json
from typing import List, Dict

_COMMON = """너는 한국 고등학교 영어 내신 서술형 출제 전문가다.
출제 스타일은 평가원 출제 패턴을 따른다.
아래 지문(문장별 번호 부여)에서 지정된 문장만 사용해 문제를 만든다.
반드시 JSON만 출력한다. 설명/마크다운/코드펜스 금지."""

def _numbered(sentences: List[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def prompt_SA(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SA] {spec['instruction']}
조건: 보기 단어를 '모두 한 번씩만' 사용. 어형 변형은 가능하되 같은 보기를 두 번 쓰지 마라.
★★ 보기 단어 개수 == (A)+(B) 정답 단어 개수. 정확히 1:1로 맞춰라.
   정답에 the가 두 번 나오면 그 자리는 빈칸으로 고르지 마라(중복 불가).
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
- bogi에는 정답에 쓰인 모든 단어의 '원형(사전 기본형)'을 '한 번씩만' 넣어라. 관사(a/the)·전치사(in/to/of 등) 같은 기능어도 정답에 쓰였다면 빠짐없이 bogi에 포함하라.
- ★ 같은 단어가 정답에서 두 번 이상 필요한 어구는 빈칸으로 고르지 마라(보기와 정답이 1:1이어야 한다).
- 정답은 오직 bogi의 단어들로만 구성하라. bogi에 없는 새 단어(특히 관사·전치사)를 정답에 추가하지 마라.
- 어형 변형은 허용되나, come→came 같은 '불규칙' 변형이 필요한 자리는 빈칸으로 고르지 마라(규칙 변형만).
- (A)와 (B)는 본문에서 '서로 겹치지 않는 다른 어구'여야 한다. 한쪽이 다른쪽에 포함되면 안 된다(예: (A)='over food a mate', (B)='a mate' 금지).
- ★ 난이도: 두 빈칸 (A)+(B)의 정답에 쓰인 단어 수 '합계'가 최소 10개, 최대 20개가 되도록 '충분히 긴 연속 어구'를 골라라. 너무 짧으면(합 10개 미만) 더 긴 어구로 다시 잡아라. (보기 단어가 10~20개가 되어야 함)"""


def prompt_SC(sentences, spec) -> str:
    return f"""{_COMMON}

[유형 SC] {spec['instruction']}
조건: 보기 어구를 변형 없이 모두 한 번씩만 배열. (A)와 (B)는 요약문의 서로 다른 절에 둔다.

★★ 수능 40번 요약문 형식으로 쓴다.
- 지문 전체의 '원인→결과' 또는 '속성→귀결' 논리를 한 문장으로 압축하라.
  첫 문장이나 마지막 문장을 바꿔 쓴 것이면 안 된다.
- 요약문 전체 25~35단어. (A)는 원인·속성 쪽, (B)는 결과·기능 쪽에 둔다.
- ★ (A)(B) 각각 최소 5단어, 권장 6~9단어. 합계 12~18단어.
- ★ (A)와 (B) 사이에 원문 텍스트가 최소 4단어 있어야 한다.
- ★ 지문 표현을 그대로 베끼지 말고 한 단계 추상화하라.
  (지문 'precisely controlled' → 요약 'the controllability of production')
- ★ 구체적 소재 명사를 빈칸으로 떼지 마라. 논리를 지고 있는 어구를 떼라.

- 같은 단어를 정답에서 두 번 쓰면, 보기에도 그 단어를 '그 횟수만큼' 똑같이 넣어라
  (because/and/the 등 중복 주의).

★★ answers 와 bogi 에는 '영어만' 넣어라. 한글 설명·라벨·괄호주석을 정답이나 보기에
   절대 섞지 마라(구조 설명은 오직 structure 필드에만 적는다).

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
★★ 어법 오류를 '정확히 3곳' 만들어라(각각 다른 문장·다른 category).
   검증 과정에서 일부가 폐기될 수 있어 여유분이 필요하다. 최종 출제는 2곳으로 확정된다.
   3곳을 못 채우면 문항 자체가 사라지므로, 아래 '좋은 예시' 수준으로 반드시 3곳을 만들어라.

═══════════════════════════════════════════════════════════
★★★ 대원칙 — "거리 벌리기" (이것 하나가 난이도를 결정한다)
═══════════════════════════════════════════════════════════
수능·평가원 29번(어법)의 오답 선지는 '형태만 봐서는 못 고르고 문장 구조를 따져야'
만들어진다. 그 비결은 판단 근거와 오류 지점 사이에 '거리'를 두는 것이다.

  ✗ the problem are            (주어 바로 뒤 → 누가 봐도 보임)
  ✓ sites devoted to sales often posts   (주어와 동사 사이에 수식어구 → 헷갈림)

  ✗ can tapping               (조동사 바로 뒤 → 당연)
  ✓ Hardly did soft drinks seemed to fit (도치 + 거리)

  ✗ differently way           (부사가 바로 뒤 명사 수식 → 눈에 띔)
  ✓ made possibly by 를 피하고, 수식 대상과 떨어진 자리를 골라라

판단 근거(주어·선행사·수식 대상·병렬 짝)와 오류 지점이 '바로 붙어 있으면'
그 오류는 절대 출제하지 마라. 사이에 전치사구·관계절·삽입구를 끼워 넣어라.

═══════════════════════════════════════════════════════════
★★★ 좋은 오류 실제 예시 — 이 수준으로 만들어라
═══════════════════════════════════════════════════════════
[수일치 — 거리 벌리기]
  ✓ sites devoted to sales often posts   (주어 sites ↔ 동사 사이 수식어구)
  ✓ designs that ... often lacks         (선행사와 동사를 멀리)
  ✓ Age, experience, and environment all plays  (복합주어인데 단수동사)
  ✓ energy ... each year have remained   (주어와 동사가 멀어 헷갈림)
  ✓ only half uses                       (of consumers 생략 → 복수인데 단수동사)

[준동사 — 본동사가 필요한 자리]
  ✓ ideology stressing the importance    (stressed 필요 — 정동사 자리)
  ✓ animation creating an image          (creates 필요 — 정동사 자리)

[태/분사 — 능동 vs 수동]
  ✓ found throwing away                  (thrown 필요 — 수동)
  ✓ me cover in flour                    (covered 필요 — 지각동사+분사)
  ✓ Internet sites which maintained by   (are 누락)

[관계사 — 삽입구·전치사로 착각 유도]
  ✓ said, quite dramatically, which the rule  (삽입구 뒤라 명사절 that을 which로 착각)
  ✓ a saying which                       (동격 that vs which)
  ✓ the main reason which people show up (1형식에 관계대명사 삽입)
  ✓ it's not the work we do what inspires (강조구문 what→that)
  ✓ dictator of that we will pass        (전치사 뒤 that vs what)

[지칭]
  ✓ wedded to them                       (the hypothesis → it인데 them으로)

[도치]
  ✓ never it is the ultimate             (부정어구 도치 → never is it)

[관용·구문]
  ✓ the world where we live in           (관계부사 + 전치사 중복)
  ✓ It takes 시간 becoming                (to become이 맞음)
  ✓ devoted to produce                   (전치사 to → producing)
  ✓ encourage O indulging                (encourage + O + to V)

═══════════════════════════════════════════════════════════
★★★ 절대 금지 (출제하면 검증기가 자동 폐기한다 — 처음부터 만들지 마라)
═══════════════════════════════════════════════════════════
1) 관사(a/an/the) 오류.  ✗ for athlete to learn / a informations
2) 어휘·철자 혼동(어법이 아니라 어휘).
   ✗ affect↔effect / avoid↔above / principal↔principle / rise↔raise / lie↔lay
   ★ affect / effect / affecting / unaffected 는 어떤 형태로도 출제 금지.
3) 둘 다 맞는 문법.
   ✗ 지각동사 + O + V / V-ing (see him run / see him running 둘 다 맞음)
   ✗ help + O + V / to V     ✗ 사역동사 수동태 + V / to V
   ✗ 자·타동사 양용 (shift / increase / decrease / change / move / open / close)
   → 틀린 출제포인트여도 어색하지 않으면 출제 금지.
4) 병렬 구조에서 '바로 옆' 항목을 바꾸는 오류.
   ✗ mildly and wet / blended and accumulating / grow and scaling
5) 조동사·동사·to 바로 뒤를 원형이 아닌 형태로 바꾸는 오류.
   ✗ can tapping / may planting / continue work / to being able
6) 근접 수일치 — 주어 명사 '바로 뒤' 동사.
   ✗ the rate increase / the problem are / larger males is
7) 동사 바로 뒤 which로 명사절 that 묻기.
   ✗ heard which it takes / indicate which the majority / find which the computer
8) 선행사 바로 앞 + what.
   ✗ factors what get us / farm what would grow / styles what are
9) 바로 뒤 수식.  ✗ differently way / True listen
10) 전치사 관용 쓰임.  ✗ interested at / information in foods / protection from↔against
    (단, 전치사+관계대명사 in which / to which 는 좋은 출제 포인트)
11) 복합관계대명사 뒤 that 중복.  ✗ whatever that he was selling
12) 문법 패턴이 공식처럼 뻔한 것.  ✗ for anyone trusting / the only reason for + to V

★★ 오류는 반드시 '서로 다른 문장'에 하나씩 배치하라. 같은 sent 번호를 두 번 쓰면
   뒤엣것은 자동 폐기되어 문제가 성립하지 않는다. errors 배열의 sent 값은 전부 달라야 한다.
★★ 같은 category를 두 번 쓰지 마라(수일치 2개, 관계사 2개 연속 금지).
★ 고친형(right)은 '해당 대상 문장 원문에 글자 그대로 존재하는 단어'여야 한다
  (검증기가 본문에서 그 단어를 찾아 오답으로 바꿔치기하므로, 원문에 없으면 통째로 폐기된다).

DB 권장 유형:
{wl}

지문:
{_numbered(sentences)}

출력 JSON (gp_id는 쓰지 말 것, category로 표기):
{{"type":"SD","errors":[
  {{"sent":<번호>,"wrong":"<바뀌는 한 단어만(문장 전체 금지)>","right":"<원문에 그대로 있는 고친 한 단어>","category":"<category>","why":"<문법 규칙 이름만. 15자 이내>"}}
]}}
주의: wrong/right에는 '실제로 바뀌는 그 단어'만 넣어라(문장이나 긴 어구를 통째로 넣지 마라).
★★ why 작성 규칙 — 학생이 읽는 답지에 그대로 나가는 문장이다.
- '문법 규칙 이름'만 짧게 적어라. 예: '주어-동사 수일치', '전치사 + 동명사', '정동사 자리', '수동 분사'.
- 문장으로 쓰지 마라. 화살표(→)나 단어 대조를 why 안에 넣지 마라(코드가 따로 붙인다).
- '오류', '삽입', '출제', '일부러', '의도적', '정답은', '만들었' 같은 출제자 시점 표현을 절대 쓰지 마라."""


def prompt_SE(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SE] {spec['instruction']}
핵심 주장·소재 단어 위주로 {len(target_idx)}곳을 빈칸. 보기에는 '원형'을 주고 학생이 품사를 변형하게 한다.
대상 문장: {target_idx}

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SE","bogi":["<원형들 + 오답용 원형들(섞어서)>"],
 "blanks":[{{"label":"C","sent":<번호>,"base":"<원형>","base_pos":"<동사/명사/형용사/부사 중 하나>","answer":"<변형형>","note":"<품사 변화>"}}]}}
규칙(어기면 검증기가 그 빈칸을 통째로 폐기한다 — 처음부터 지켜라):
- ★ answer는 '대상 문장 원문에 글자 그대로 등장하는 그 단어'여야 한다. 본문에 없는 형태를 지어내지 마라(없으면 빈칸이 안 뚫려 폐기됨). 즉 '본문에 이미 파생형으로 쓰인 단어'를 골라 그 원형을 base로 제시하는 방식이다.
- ★ base_pos 에는 원형(base)의 품사를 반드시 적어라: 동사 / 명사 / 형용사 / 부사 중 하나. 이 값으로 굴절 허용 여부가 갈리므로 정확히 적어라.
- ★ base가 '동사'이면 굴절도 허용한다. produce→produces(3인칭), adhere→adhering(동명사), prove→proved(과거/과거분사) 모두 정답으로 인정된다.
- ★ base가 동사가 아니면 굴절은 금지다. photograph→photographs(명사 복수), strong→stronger(비교급)는 출제하지 마라.
- ★ 어느 경우든 base와 answer의 품사가 '완전히 같은 파생'은 금지다(예: race→racialization 은 명사→명사라 불가).
- 좋은 예: educate→education(동→명), simple→simply(형→부), nation→national(명→형), create→creative(동→형).
- answer는 base와 반드시 달라야 한다(무변형 금지).
- base는 사전 원형으로 적어라(answer의 어근과 일치해야 함).
- ★ 오답(distractor): bogi에는 정답 원형들 외에 '지문에 실제로 등장하는 다른 내용어'(정답과 무관)를 원형으로 추가해, 보기 총 개수가 빈칸 개수의 약 1.5배가 되게 하라(예: 빈칸 4개 → 보기 6개). 오답도 '원형'으로 적어 섞어라(정답 티 안 나게)."""
