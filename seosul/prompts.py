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

═══════════════════════════════════════════════════════════
★★★ 어느 문장을 고를 것인가 - 수능 빈칸추론과 같은 기준
═══════════════════════════════════════════════════════════
아무 문장이나 뚫지 마라. 글의 논리를 지고 있는 문장만 고른다.
아래 후보 중에서 '논리적으로 짝이 되는 두 문장'을 직접 골라라.

  (A) 글의 주장·결론이 직접 드러난 문장   ← 수능 빈칸이 뚫는 자리
  (B) 그 주장을 뒷받침하거나 뒤집는 문장   ← 원인 / 근거 / 대조

  두 문장은 서로 '원인↔결과', '주장↔근거', '통념↔반박' 처럼 이어져야 한다.
  아무 관계 없는 두 문장을 고르면 안 된다.

✗ 고르면 안 되는 문장
  - 도입부 배경 설명 (In recent years… / Since ancient times…)
  - 예시·일화만 있는 문장 (For example… / Imagine… / One study found…)
  - 숫자·고유명사 나열 문장
  - 마지막 마무리 인사말 성격 문장

고를 수 있는 문장 번호: {target_idx}

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
- ★ 분량: (A)와 (B) 각각 4~7단어, 합계 8~14단어. 핵심 문장 두 개를 동시에 뚫는 것이므로
  너무 길게 떼면 문장이 통째로 비어 읽을 수가 없다. 문장의 '논리를 지고 있는 부분'만 짧게 떼라."""


# =========================================================
#  2) SE - 어휘 품사 변형
# =========================================================
def prompt_SE(sentences, target_idx, spec, used_sents=(), avoid_text="",
              title_only: bool = False) -> str:
    """제목·결론 빈칸 — 본문에서 단어를 찾아 어형을 바꿔 채우기.

    ★ 지문을 건드리지 않는다. 지문 밖에 '한 줄'을 새로 쓰고 거기에 빈칸 하나를 판다.
    ★★ 핵심 규칙 두 줄 (선생님 확정)
        어근  : 본문에 '있어야' 한다   → 학생이 찾을 수 있어야 하므로
        정답  : 본문에 '없어야' 한다   → 있으면 그냥 베껴 쓰게 되어 변형 판단이 사라진다
    """
    av = (f"\n★ 아래는 이미 나간 다른 문항의 정답·문장이다. 네가 쓰는 한 줄에 그대로 넣지 마라.\n"
          f"   답이 새어 나가고, 학생이 같은 내용을 두 번 읽게 된다.\n"
          f"   {avoid_text}\n") if avoid_text else ""
    # 요약문 빈칸 영작이 이미 지문을 한 문장으로 눌러 놨으면 결론 요약형을 금지한다.
    # 안 막으면 같은 일을 두 번 시키는 시험지가 된다(실제 사고: 2번 문장이 4번 정답을 풀어 줌).
    fr = ("\n★★ 이번에는 frame=\"title\" 만 쓴다. frame=\"summary\" 는 금지다.\n"
          "   요약문 빈칸 영작이 이미 이 지문을 한 문장으로 요약해 놓았다.\n"
          "   네가 또 요약 문장을 쓰면 학생이 같은 일을 두 번 하게 된다.\n"
          "   명사구 제목으로 써라 — 마침표 없이, Title Case 로.\n") if title_only else ""
    return f"""{_COMMON}

[유형 SE] 이 지문을 압축한 '한 줄'을 새로 쓰고, 그 안에 빈칸 1개를 만든다.
★ 지문 본문은 절대 건드리지 마라. 빈칸은 오직 네가 쓴 그 한 줄 안에만 있다.
{fr}{av}
═══════════════════════════════════════════════════════════
★★★ 프레임 - 지문에 맞는 쪽을 네가 골라라
═══════════════════════════════════════════════════════════
frame="title"   제목형
  · 고3 평가원 제목추론(24번) 선지 문체. 명사구로 쓰고 마침표를 찍지 않는다.
  · Title Case. 콜론(:)으로 두 덩이로 끊는 형태를 기본으로 한다.
  · 10~16단어. 비유·대조·역설을 써도 좋고, 물음표로 끝나도 된다.
  · 지문이 '하나의 주제'로 딱 떨어질 때 쓴다.
  예: The Hidden Cost of {{{{C}}}}: How Farming Villages Became the Cradle of Epidemic Disease

frame="summary"  결론 요약형
  · 지문의 '결론부 한두 문장'을 합쳐 다시 쓴 완결된 문장. 마침표로 끝낸다.
  · 22~40단어. 주변 표현은 전부 바꾸되 빈칸 자리만 남긴다.
  · 지문이 '원인 → 결과'처럼 흐름을 타서 한 줄 제목으로 누르기 애매할 때 쓴다.
  예: The New Stone Age marked an important turning point in human history, as the
      advancement of {{{{C}}}} increased population density, leading to the formation of
      towns and cities and eventually to the rise of civilizations.

★ 어느 쪽이든 지문의 '결론·주장'을 담아야 한다. 도입부 배경이나 예시를 옮기지 마라.
★ 지문 문장을 그대로 베껴 오지 마라. 반드시 다시 써라.

═══════════════════════════════════════════════════════════
★★★ 빈칸 규칙 - 어기면 검증기가 통째로 폐기한다
═══════════════════════════════════════════════════════════
1) 빈칸은 '정확히 1개'. 문장 안에 {{{{C}}}} 를 한 번만 넣어라.
2) base(어근)는 '본문에 글자 그대로 있는 단어'여야 한다. 학생이 본문에서 찾아야 하므로.
3) ★ answer(정답)는 '본문에 없는 형태'여야 한다.
   본문에 있으면 학생이 그대로 베껴 쓰게 되어 이 문항이 성립하지 않는다.
   예) 본문에 cultivating 이 있고 cultivation 은 없다  →  base=cultivating, answer=Cultivation  (O)
       본문에 growth 가 이미 있다                      →  answer=growth 는 금지        (X)
4) answer 는 한 단어. 띄어쓰기·하이픈 금지.
5) base 와 answer 는 품사가 달라야 한다(파생). 단순 복수·시제 변화는 금지.
   좋은 예: cultivating→Cultivation(동→명), domesticating→Domestication(동→명),
            settled→Settlement(동→명), diverse→Diversity(형→명), able→Ability(형→명)
6) 네가 쓴 문장 안에 answer 나 base 가 (빈칸 말고) 또 나오면 안 된다. 답이 노출된다.

지문:
{_numbered(sentences)}

출력 JSON:
{{{{"type":"SE","frame":"<title 또는 summary>",
 "title":"<{{{{C}}}} 를 정확히 한 번 포함한 한 줄>",
 "blanks":[{{{{"label":"C","base":"<본문에 있는 단어 그대로>","base_pos":"<동사/명사/형용사/부사>",
             "answer":"<본문에 없는 파생형 한 단어>","note":"<품사 변화. 예: 동사 → 명사>"}}}}]}}}}"""


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
