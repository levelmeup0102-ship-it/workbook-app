"""
seosul/prompts.py
유형별 출제 프롬프트. seosul_types(지시문/조건)와 grammar_points(어법 화이트리스트)를
주입하여 LLM이 '즉석 판단'이 아니라 'DB 근거'로만 출제하게 만든다.
모든 프롬프트는 검증기가 파싱할 수 있도록 엄격한 JSON만 출력하도록 요구한다.

★ 생성 순서 (generator.generate_set이 이 순서로 호출한다)
   1) SA 본문 빈칸 영작      → (A)(B) 자리 확정
   2) SE 어휘 품사 변형      → (C)(D)(E) 자리 확정
   3) SD 어법 틀린 곳        → 남은 문장에서 '구문 재작성'
   4) SC 요약문 빈칸         → 위 결과를 보고 지문 전체 요약
   앞 단계가 점유한 문장은 used_sents로 뒤 단계에 전달되어 겹침이 원천 차단된다.
"""
import json
from typing import List, Dict

_COMMON = """너는 한국 고등학교 영어 내신 서술형 출제 전문가다.
출제 스타일은 평가원 출제 패턴을 따른다.
아래 지문(문장별 번호 부여)에서 지정된 문장만 사용해 문제를 만든다.
반드시 JSON만 출력한다. 설명/마크다운/코드펜스 금지."""


def _numbered(sentences: List[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def _used_note(used_sents) -> str:
    """앞 단계가 이미 점유한 문장 안내. 겹치면 검증기가 통째로 폐기한다."""
    u = sorted(int(x) for x in (used_sents or []))
    if not u:
        return ""
    return (f"\n★★ 이미 다른 문항이 사용한 문장: {u}\n"
            f"   이 문장들은 이미 빈칸이 뚫려 있다. 절대 고르지 마라(고르면 통째로 폐기된다).\n")


# =========================================================
#  1) SA - 본문 빈칸 배열영작
# =========================================================
def prompt_SA(sentences, target_idx, spec) -> str:
    return f"""{_COMMON}

[유형 SA] {spec.get('instruction','')}
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
- (A)와 (B)는 본문에서 '서로 겹치지 않는 다른 어구'여야 한다. 한쪽이 다른쪽에 포함되면 안 된다.
- ★ 난이도: 두 빈칸 (A)+(B)의 정답 단어 수 '합계'가 최소 10개, 최대 20개가 되도록 '충분히 긴 연속 어구'를 골라라."""


# =========================================================
#  2) SE - 어휘 품사 변형
# =========================================================
def prompt_SE(sentences, target_idx, spec, used_sents=()) -> str:
    return f"""{_COMMON}

[유형 SE] {spec.get('instruction','')}
핵심 주장·소재 단어 위주로 {len(target_idx)}곳을 빈칸. 보기에는 '원형'을 주고 학생이 품사를 변형하게 한다.
대상 문장: {target_idx}
{_used_note(used_sents)}
지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SE","bogi":["<원형들 + 오답용 원형들(섞어서)>"],
 "blanks":[{{"label":"C","sent":<번호>,"base":"<원형>","base_pos":"<동사/명사/형용사/부사 중 하나>","answer":"<변형형>","note":"<품사 변화>"}}]}}
규칙(어기면 검증기가 그 빈칸을 통째로 폐기한다 - 처음부터 지켜라):
- ★ answer는 '대상 문장 원문에 글자 그대로 등장하는 그 단어'여야 한다. 본문에 없는 형태를 지어내지 마라(없으면 빈칸이 안 뚫려 폐기됨). 즉 '본문에 이미 파생형으로 쓰인 단어'를 골라 그 원형을 base로 제시하는 방식이다.
- ★ base_pos 에는 원형(base)의 품사를 반드시 적어라: 동사 / 명사 / 형용사 / 부사 중 하나. 이 값으로 굴절 허용 여부가 갈리므로 정확히 적어라.
- ★ base가 '동사'이면 굴절도 허용한다. produce→produces(3인칭), adhere→adhering(동명사), prove→proved(과거/과거분사) 모두 정답으로 인정된다.
- ★ base가 동사가 아니면 굴절은 금지다. photograph→photographs(명사 복수), strong→stronger(비교급)는 출제하지 마라.
- ★ 어느 경우든 base와 answer의 품사가 '완전히 같은 파생'은 금지다(예: race→racialization 은 명사→명사라 불가).
- ★ 하이픈 복합어(routine-type, distraction-free)는 파생이 아니므로 출제하지 마라.
- 좋은 예: educate→education(동→명), simple→simply(형→부), nation→national(명→형), create→creative(동→형).
- answer는 base와 반드시 달라야 한다(무변형 금지).
- base는 사전 원형으로 적어라(answer의 어근과 일치해야 함).
- ★ 오답(distractor): bogi에는 정답 원형들 외에 '지문에 실제로 등장하는 다른 내용어'(정답과 무관)를 원형으로 추가해, 보기 총 개수가 빈칸 개수의 약 1.5배가 되게 하라(예: 빈칸 4개 → 보기 6개). 오답도 '원형'으로 적어 섞어라."""


# =========================================================
#  3) SD - 어법 틀린 곳 고치기 (★ 구문 재작성 방식)
# =========================================================
def prompt_SD(sentences, target_idx, allowed_gp: List[dict], used_sents=()) -> str:
    if allowed_gp:
        wl = "\n".join(f"- [{g.get('category','')}] {g['name']} :: 함정={g.get('trap_warning','')}"
                       for g in allowed_gp[:30])
    else:
        wl = ("- [준동사] 정동사 자리 vs 준동사 자리\n"
              "- [준동사] 능동/수동 분사 (v-ing vs p.p.)\n"
              "- [수일치] 주어-동사 거리 수일치\n"
              "- [관계사] that/what/which 구분, 전치사+관계대명사")
    return f"""{_COMMON}

[유형 SD] 지정된 문장의 '구문을 바꿔' 어법 오류 문항을 만든다. 밑줄/번호 표시 없음.
대상 문장(여기서만 출제): {target_idx}
{_used_note(used_sents)}
═══════════════════════════════════════════════════════════
★★★ 핵심 - 단어 하나를 바꾸는 게 아니라 '구문을 갈아끼운다'
═══════════════════════════════════════════════════════════
실제 수능·평가원 어법 문항은 원문 문장을 그대로 두고 단어만 바꾸지 않는다.
문장의 구조 자체를 다른 구문으로 재작성한 뒤, 그 새 구조에서 오류를 만든다.

  원문  : the cognitive load imposed by their environment
  재작성: the cognitive load the environment imposing on them
          → 수동 분사구를 '목적격 관계대명사 생략 + 정동사' 구조로 바꿨다.
            그 자리는 정동사(imposes)가 와야 하는데 imposing으로 출제.
          wrong=imposing  right=imposes  category=준동사

  원문  : Bar, who directs the Center, reports that ...
  재작성: Bar, who directs the Center, has found this, report that ...
          → 정동사를 분사구문 자리로 바꿔 reporting이 맞는 구조를 만들었다.
          wrong=report  right=reporting  category=준동사

  원문  : a house located in the city
  재작성: a house locating in the city
          → 수동(located)이어야 하는 자리.  wrong=locating  right=located

  원문  : led to the loss of many programs
  재작성: led to lose many programs
          → 전치사 to 뒤라 동명사.  wrong=lose  right=losing

★ 재작성 문장은 원문과 '의미가 같아야' 하고, 원문 단어를 대부분 그대로 써야 한다
  (단어의 70% 이상 공유). 새 내용을 지어내지 마라. 구조만 바꾼다.

═══════════════════════════════════════════════════════════
★★★ 출제 축 - 이 비중으로 만들어라
═══════════════════════════════════════════════════════════
[1순위 · 절반] 준동사 ↔ 정동사  (가장 많이 나오는 축)
  - 분사구 ↔ 관계절 변환 후 정동사/준동사 판단
  - 목적격 관계대명사 생략으로 동사가 두 개처럼 보이게
  - 능동 분사(v-ing) vs 수동 분사(p.p.)
  - 전치사 + 동명사 / to부정사 vs 동명사
  예: stressing→stressed / throwing→thrown / receiving→received / housing→houses

[2순위 · 1/4] 먼 수일치
  - 주어와 동사 사이에 전치사구·관계절을 끼워 '거리를 벌린' 수일치만 출제
  예: sites devoted to sales often posts / Age, experience, and environment all plays
  ✗ 주어 바로 뒤 동사는 금지 (the problem are / larger males is)

[3순위 · 1/4] 관계사
  - that / what / which 구분, 전치사 + 관계대명사(in which / to which)
  - 선행사가 있으면 that·which, 없으면 what
  예: said, quite dramatically, which the rule (삽입구 뒤 that을 which로)
      it's not the work we do what inspires (강조구문 what→that)
      the world where we live in (관계부사 + 전치사 중복)

═══════════════════════════════════════════════════════════
★★★ 절대 금지 (검증기가 자동 폐기한다 - 처음부터 만들지 마라)
═══════════════════════════════════════════════════════════
1) 관사(a/an/the) 오류.
2) 어휘·철자 혼동. affect↔effect / rise↔raise / lie↔lay / principal↔principle
   ★ affect / effect / affecting / unaffected 는 어떤 형태로도 출제 금지.
3) 둘 다 맞는 문법. 지각동사·help 뒤 원형 vs to V / 사역동사 수동태 / 자·타동사 양용(increase, change, move, shift)
4) 병렬 구조에서 '바로 옆' 항목을 바꾸는 오류.
5) 조동사·to 바로 뒤를 원형이 아닌 형태로.  ✗ can tapping / to being able
6) 근접 수일치 - 주어 명사 '바로 뒤' 동사.  ✗ the rate increase / the problem are
7) 동사 바로 뒤 which로 명사절 that 묻기.  ✗ heard which it takes
8) 선행사 바로 앞 what.  ✗ factors what get us
9) 바로 뒤 수식.  ✗ differently way
10) 전치사 관용 쓰임.  ✗ interested at (단, in which / to which 는 좋은 출제 포인트)

★★ 오류는 '정확히 3곳' 만들어라. 검증에서 일부가 폐기되므로 여유분이 필요하다
   (최종 출제는 2곳으로 확정된다). 3곳 모두 서로 다른 문장, 서로 다른 category.

권장 유형(DB):
{wl}

지문:
{_numbered(sentences)}

출력 JSON:
{{"type":"SD","errors":[
  {{"sent":<번호>,
    "original":"<그 번호 원문 문장 그대로>",
    "rewritten":"<구문을 바꾸고 오류를 넣은 문장. 이 문장이 지문에 그대로 실린다>",
    "wrong":"<rewritten 안에 실제로 들어 있는 틀린 단어 하나>",
    "right":"<학생이 고쳐 써야 할 올바른 단어 하나>",
    "category":"<준동사 / 수일치 / 관계사 중 하나>",
    "why":"<문법 규칙 이름만. 15자 이내>"}}
]}}
주의:
- rewritten에는 wrong이 '글자 그대로' 들어 있어야 한다(없으면 폐기).
- wrong/right는 '한 단어'만. 문장이나 긴 어구를 넣지 마라.
- rewritten의 wrong을 right로 바꾸면 문법적으로 완전한 문장이 되어야 한다.
★★ why 작성 규칙 - 학생이 읽는 답지에 그대로 나간다.
- '문법 규칙 이름'만 짧게. 예: '정동사 자리', '수동 분사', '주어-동사 수일치', '전치사 + 동명사'.
- 문장으로 쓰지 마라. 화살표(→)나 단어 대조를 why 안에 넣지 마라(코드가 붙인다).
- '오류', '삽입', '출제', '일부러', '의도적', '정답은', '만들었' 같은 출제자 시점 표현 금지."""


# =========================================================
#  4) SC - 요약문 빈칸 (맨 마지막)
# =========================================================
def prompt_SC(sentences, spec, prior_note: str = "") -> str:
    pn = f"\n[참고] 이 지문에는 이미 다음 문항이 출제되어 있다:\n{prior_note}\n" if prior_note else ""
    return f"""{_COMMON}

[유형 SC] {spec.get('instruction','')}
조건: 보기 어구를 변형 없이 모두 한 번씩만 배열. (A)와 (B)는 요약문의 서로 다른 절에 둔다.
{pn}
★★ 수능 40번 요약문 형식으로 쓴다.
- 지문 전체의 '원인→결과' 또는 '속성→귀결' 논리를 한 문장으로 압축하라.
  첫 문장이나 마지막 문장을 바꿔 쓴 것이면 안 된다.
- 요약문 전체 25~35단어. (A)는 원인·속성 쪽, (B)는 결과·기능 쪽에 둔다.
- ★ (A)(B) 각각 최소 5단어, 권장 6~9단어. 합계 12~18단어.
- ★ (A)와 (B) 사이에 원문 텍스트가 최소 4단어 있어야 한다.
- ★ 지문 표현을 그대로 베끼지 말고 한 단계 추상화하라.
  (지문 'precisely controlled' → 요약 'the controllability of production')
- ★ 구체적 소재 명사를 빈칸으로 떼지 마라. 논리를 지고 있는 어구를 떼라.
- 같은 단어를 정답에서 두 번 쓰면, 보기에도 그 단어를 '그 횟수만큼' 똑같이 넣어라.

★★ answers 와 bogi 에는 '영어만' 넣어라. 한글 설명·라벨·괄호주석을 정답이나 보기에
   절대 섞지 마라(구조 설명은 오직 structure 필드에만 적는다).

지문:
{_numbered(sentences)}

출력 JSON (모든 키 필수 - bogi를 절대 빠뜨리지 마라):
{{"type":"SC","summary":"<(A)자리는 {{{{A}}}}, (B)자리는 {{{{B}}}} 인 2문장 요약>",
 "answers":{{"A":"...","B":"..."}},"bogi":["...","..."],"structure":"<사용한 구조명>"}}
주의: summary의 빈칸 자리에는 '(A)','(B)' 같은 라벨이나 밑줄(___)을 쓰지 말고, 오직 {{{{A}}}} {{{{B}}}} placeholder만 정확히 한 번씩 넣어라."""
