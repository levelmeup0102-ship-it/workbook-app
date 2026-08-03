"""Stage 7-1 Step 1 (Lv.8-1 어법 괄호형) 프롬프트 템플릿.

[수정 이력]
- rule 6: "문장당 정확히 1개" → 목표 개수만 제시, 문장별 할당 폐기
- <distribution>: "합계 = 총 문장 수" → target/min 방식, {bracket_dist_lines} 폐기
- <lv82_rules>: "최소 5개 목표" → 8-1 triple 개수와 동일하게 동기화
"""

PROMPT_TEMPLATE = """다음 영어 지문으로 어법 문제 2종류를 생성합니다.

<task>
- Lv.8-1 괄호형: (원문장, 정답, 오답) triple 리스트. 괄호 번호와 위치는 코드가 처리.
- Lv.8-2 서술형: 원문에 오류만 삽입한 수정본 + 오류 답안 리스트.
- ★ 두 문제의 출제 개수는 반드시 동일하다.
</task>

<source sentence_count="{sent_count}">
{passage}
</source>

<output_rules>
<rule id="1">출력은 &lt;output_format&gt; 안의 JSON만. 설명·주석·markdown·앞뒤 텍스트 금지.</rule>
<rule id="2">grammar_bracket_passage는 list of triples: [["원문 문장", "정답 단어/구", "오답 단어/구"], ...]</rule>
<rule id="3">각 triple[0]은 sentences[i]와 글자 단위 동일 (공백·구두점·대소문자 보존).</rule>
<rule id="4">triple[1] (정답)은 triple[0] 안에 ★유일하게 한 번만★ 등장하는 토큰/구. 코드가 첫 등장 자리에 괄호를 박으므로, 중복 등장 시 의도 외 자리가 감싸짐.</rule>
<rule id="5">괄호 번호 (N), 좌우 정답 위치는 코드가 결정. AI는 단어쌍만 제공.</rule>
<rule id="6">triple 총 개수는 &lt;distribution&gt;의 target이 ★목표★이며 강제가 아니다. 문장별 할당은 없다 — 괄호가 0개인 문장이 있는 것이 정상이다. 후보가 없으면 개수를 채우지 말고 적게 출력한다.</rule>
<rule id="7">한 문장에서 뽑는 triple은 최대 1개.</rule>
</output_rules>

<patterns>
<intro>원문을 훑어 아래 trigger를 찾으면 후보 풀에 등록. 후보 = (문장, 정답표현, 오답표현, 등급, 출제포인트↔근거단어 거리). 이 단계는 내부 수행이며 출력하지 않음.</intro>

<grade name="S" priority="highest">
<pattern id="S1">"in/on/by/for/at/of/to + which" 또는 명사 + which 직후 → 전치사+which vs which (뒤 절 완전성)</pattern>
<pattern id="S2">think/believe/say/know/feel/find/realize/show/suggest + that절 → that vs what (선행사 무 + 뒤 절 완전 → that)</pattern>
<pattern id="S3">fact/idea/belief/news/opinion/notion/claim + that절 → 동격 that vs which (추상명사 + 완전 절)</pattern>
<pattern id="S4">so + 형/부 + that 또는 such + 명 + that → 결과절 that (which 불가)</pattern>
<pattern id="S5">한 문장에 본동사 부재 의심 자리 → 본동사 vs 준동사 (V vs V-ing/V-ed)</pattern>
<pattern id="S6">문장 첫머리 "V-ing, S+V" 또는 "V-ed, S+V" → 분사구문 vs 명령문/본동사</pattern>
<pattern id="S7">주격관계대명사 who/which/that + 동사 (선행사가 5단어 이상 떨어진 경우만)</pattern>
</grade>

<grade name="A">
<pattern id="A1">명사 + V-ing/V-ed 후치수식 → 능동/수동</pattern>
<pattern id="A2">같은 어근 형용사/부사 쌍 (accurate/accurately, clear/clearly, careful/carefully)</pattern>
<pattern id="A3">1형식 동사(matter/function/work/exist/happen/occur/appear) + 부사</pattern>
<pattern id="A4">2형식 동사(be/become/seem/look/feel/sound/taste) + 형용사 보어</pattern>
<pattern id="A5">decide/agree/hope/want/plan/promise/refuse/expect/manage/fail/choose/learn/offer + to V</pattern>
<pattern id="A6">enjoy/finish/mind/avoid/suggest/deny/postpone/consider/admit/recommend + V-ing</pattern>
<pattern id="A7">couldn't help but + 동사원형</pattern>
<pattern id="A8">one of + 복수명사</pattern>
</grade>

<grade name="B">
<pattern id="B1">주어 ↔ 동사 사이 5단어 이상 수식어구가 끼인 수일치</pattern>
<pattern id="B2">학문명 -ics(mathematics/physics/economics/politics/statistics/ethics/linguistics) + 단수동사</pattern>
<pattern id="B3">A number of + 복수동사 / The number of + 단수동사</pattern>
<pattern id="B4">Either A or B / Neither A nor B / Not only A but also B의 동사 일치 (B에 일치)</pattern>
</grade>
</patterns>

<forbidden>
<intro>아래 서명 중 하나라도 매칭되면 후보 풀에서 즉시 제거. 점수 무관 즉시 탈락.</intro>

<filter id="C1" name="둘 다 정답 — 어법 성립 X">
- start/continue/begin/love/like/hate + [to V / V-ing]
- 주어 자리 [To V / V-ing]
- help (+ 목적어) + [V / to V]
- see/watch/hear/feel/notice + 목적어 + [V / V-ing]
- 사역동사 make/let/have + 목적어 + [V] (have만 p.p. 가능)
- and/or 병렬 to 생략 (to A and [B / to B])
- 목적격 관계대명사 [who / whom]
- 목적격 관계대명사 [생략 / 사용]
- 주격/목적격 자리 [that / which]
- 관계부사 자리 why/where/when 교차, those/these 위치 변경
</filter>

<filter id="C2" name="어휘·의미 차이 — 어법 아님">
- 시제: [is/was], [has/had], [do/did], [goes/went], [studies/studied]
- 진행 vs 단순: [is studying/studies], [was running/ran]
- 부정/긍정: [can/cannot], [could/couldn't], [has/hasn't], [is/isn't]
- 동의어 형/부: [varied/various], [different/diverse], [big/large], [many/numerous]
- 접두사 어휘: [accurate/inaccurate], [possible/impossible], [legal/illegal], [fair/unfair]
- 추상/불가산명사 + s: [advice/advices], [information/informations], [well-being/well-beings]
- 고유명사 + s: [Walthamstow/Walthamstows]
</filter>

<filter id="C3" name="바로 옆 힌트 — 거리 ≤ 1 토큰">
- 조동사 + [V / V-ing], 조동사 + [V / V-ed]
- have/has/had + [p.p. / V]
- be동사 + [V-ing / V], be동사 + [p.p. / V]
- by 행위자 직전 [능동 / 수동]
- 주어 명사/대명사 + [동사 단수 / 복수] (★ 일반명사 동일: this [causes/causing], Harold [confirms/confirm] 금지)
- until/when/if + 주어 + [동사 수일치]
- be + 부사 + [V-ing / V-ed] (명백한 분사 자리)
</filter>

<filter id="C4" name="정답이 선택지에 없음">
- 원문에 없는 표현을 정답으로 제시 금지
- those that/those who의 those 빠진 [that/where] 같은 쌍 금지
</filter>

<filter id="C5" name="성별 불명확 대명사">
- 인물 성별 특정 불가 시 [his/her], [him/her] 금지
</filter>
</forbidden>

<distance_check>
<formula>거리 = (출제 포인트) ↔ (정답을 결정하는 가장 가까운 근거 단어) 사이 토큰 수.</formula>
<note>부사·관사·전치사도 토큰. 부사 1개만 끼어도 "바로 옆"으로 간주.</note>
<rule>거리 &lt; 2 인 후보는 아래 예외만 허용. 그 외는 제거.</rule>
<exceptions>
- S5 본동사 vs 준동사
- S6 분사구문
- A1 분사 후치수식 능/수동
- S2/S3/S4 (절 완전성 판단)
- A2~A4 형용사 vs 부사 (1형식/2형식 식별)
- B2 학문명 -ics
- A7 couldn't help but
</exceptions>
</distance_check>

<distribution target="{bracket_count}" min="3">
<note>target은 ★목표 개수★일 뿐이다. 총 문장 수와는 무관하며, 문장별 할당량은 존재하지 않는다.</note>
<note>한 문장에 최대 1개. 괄호가 0개인 문장이 있는 것이 정상이며, 그 문장도 원문 그대로 전부 출력한다.</note>
<note>후보가 부족하면 min(3개)까지 내려도 된다. 3개도 못 채우면 채워진 만큼만 출력한다.</note>
<note>★ 개수를 맞추려고 &lt;forbidden&gt;에 걸리는 자리를 채택하는 것은 절대 금지. 개수 부족이 언제나 더 낫다.</note>
</distribution>

<selection_procedure>
<step n="1">&lt;patterns&gt;에서 trigger 패턴 스캔 → 후보 풀 등록</step>
<step n="2">후보를 (등급 S&gt;A&gt;B) → (거리 큰 순) → (수능형 난이도 높은 순) 정렬</step>
<step n="3">&lt;forbidden&gt; C1~C5 금지 필터 + &lt;distance_check&gt; 통과 → 상위 후보부터 채택</step>
<step n="4">triple[1]이 triple[0] 안에 ★유일하게 한 번만★ 등장하는지 확인. 중복 시 다음 후보로.</step>
<step n="5">같은 문장에서 이미 채택된 triple이 있으면 건너뛴다 (문장당 최대 1개).</step>
<step n="6">target 개수에 도달하거나 후보가 소진되면 종료. 남은 문장은 괄호 없이 그대로 둔다.</step>

<priority>
★ 매우 중요 — 누락 허용 원칙
- 분배 개수는 ★목표값★. 강제 아님.
- 후보 없거나 모두 금지 필터에 걸리면 그 문장 triple 생략.
- ★ 개수 부족 &lt; 잘못된 어법 출제 ★ — 무리해서 채우지 말 것.
- 어휘 문제, 의미 차이, 바로 옆 힌트, 둘 다 정답 자리로 개수 메우는 것 절대 금지.
- 후보가 target보다 적다는 이유로 출력을 지연하거나 재검토를 반복하지 말 것. 통과한 후보만 즉시 출력한다.
</priority>
</selection_procedure>

<good_examples>
<example pattern="S1">
<sentence>The house in which he lives is old.</sentence>
<correct>in which</correct>
<wrong>which</wrong>
<reason>lives는 자동사+부사구 → 뒤 절 완전 → in which 필요</reason>
</example>

<example pattern="S2">
<sentence>I believe what he said was honest.</sentence>
<correct>what</correct>
<wrong>that</wrong>
<reason>said 뒤 목적어 빠진 불완전한 절 → what</reason>
</example>

<example pattern="B2">
<sentence>The aesthetics of a new project is too often considered irrelevant.</sentence>
<correct>is</correct>
<wrong>are</wrong>
<reason>aesthetics는 학문명 단수 취급</reason>
</example>

<example pattern="A1">
<sentence>I ordered the dish called kibbeling at the stand.</sentence>
<correct>called</correct>
<wrong>calling</wrong>
<reason>dish가 부르는 행위의 대상 → 수동 p.p.</reason>
</example>

<example pattern="S5">
<sentence>The animation, displayed on the wall, creates vivid images of war.</sentence>
<correct>creates</correct>
<wrong>creating</wrong>
<reason>주절 본동사 자리. displayed는 분사구.</reason>
</example>
</good_examples>

<bad_examples>
<example violation="C3"><sentence>He [helps / help] her study.</sentence><reason>He 바로 뒤 수일치, 1초 컷</reason></example>
<example violation="C1"><sentence>We started [studying / to study] hard.</sentence><reason>start + V-ing/to V 둘 다 정답</reason></example>
<example violation="C2"><sentence>It [is / was] famous in the 1960s.</sentence><reason>시제 차이, 어법 아님</reason></example>
<example violation="C2"><sentence>This formula is [accurate / inaccurate].</sentence><reason>접두사 어휘 차이</reason></example>
<example violation="C1"><sentence>the museums [that / which] attract visitors</sentence><reason>주격 관계대명사 자리에서 that=which</reason></example>
<example violation="C3"><sentence>is too often [considered / considering] irrelevant</sentence><reason>be + 부사 + 분사 패턴</reason></example>
<example violation="C3"><sentence>this [causes / causing] problems</sentence><reason>명사주어 바로 뒤, 1초 컷</reason></example>
</bad_examples>

<no_candidate_sentences>
<intro>아래에 해당하는 문장은 후보 스캔 대상이 아니다. triple을 만들려 시도하지 말고 건너뛴다.</intro>
<case>본동사·절이 없는 문장 (예: "A shark? A lion? Or, maybe a bear?")</case>
<case>6단어 이하의 단문</case>
<case>절이 하나뿐이고 주어-동사가 인접해 C3에만 해당하는 문장</case>
<case>&lt;patterns&gt; S/A/B 어느 trigger에도 걸리지 않는 문장</case>
</no_candidate_sentences>

<final_verification>
<intro>각 triple 출력 직전 모든 항목을 점검. 하나라도 실패면 그 triple 폐기 후 다른 자리로 교체. 교체할 자리가 없으면 그냥 개수를 줄인다.</intro>
<check>triple[0] = sentences[i] 글자 단위 동일</check>
<check>triple[1] 정답이 triple[0] 안에 실제 토큰으로 ★유일하게 1번만★ 존재</check>
<check>triple[2] 오답은 해당 문맥에서 정답으로 인정되지 않음</check>
<check>정답/오답 판단에 ★문법적 근거 1개 이상★ (문장 구조, 절 완전성, 수식 관계, 의미역, 동사 성질 등)</check>
<check>단순 형태 단서만으로 1초 만에 풀리는 자리가 아님</check>
<check>&lt;forbidden&gt; C1~C5 어느 서명에도 매칭되지 않음</check>
<check>거리 ≥ 2 (또는 &lt;distance_check&gt; 예외)</check>
<check>&lt;patterns&gt; S/A/B 등급 trigger 중 하나에 정확히 해당</check>
<check>같은 문장에서 두 번 뽑지 않았음</check>
</final_verification>

<lv82_rules>
<rule>★ 오류 개수 = grammar_bracket_passage의 triple 개수와 ★정확히 동일★ (8-1과 8-2는 항상 같은 개수)</rule>
<rule>8-1에서 채택한 것과 ★같은 문장의 같은 문법 포인트★를 오류로 삽입한다</rule>
<rule>원문 {sent_count}개 문장 모두 포함, 글자 단위 보존 (오류 삽입 부분 외 100% 동일)</rule>
<rule>한 문장당 최대 1개 오류 (8-1과 동일 기준)</rule>
<rule>&lt;forbidden&gt; C1~C5 + &lt;distance_check&gt; + 수능형 판단 가능성 모두 통과</rule>
<rule>어휘/의미 차이/시제 차이/바로 옆 힌트형 오류 금지</rule>
<rule>8-1 triple이 3개면 오류도 3개다. 개수를 늘리려고 다른 자리를 추가하지 말 것</rule>
<rule>grammar_error_count = grammar_error_answers 길이 = grammar_bracket_passage 길이</rule>
</lv82_rules>

<output_format>
{{
  "grammar_bracket_passage": [
    ["원문 문장", "정답 단어/구", "오답 단어/구"]
  ],
  "grammar_error_passage": "오류 포함 전체 지문 (정확히 {sent_count}문장)",
  "grammar_error_count": 실제삽입개수,
  "grammar_error_answers": [
    {{"num": 1, "original": "정답 표현", "error": "오류 표현"}}
  ]
}}
</output_format>"""


# ============================================================
# 호출부 헬퍼 (선택) — bracket_count 계산
#   기존: bracket_count = len(sentences)        ← 문장 수 = 괄호 수
#   변경: 후보 문장 수 기반, 최소 3 / 상한 8
# ============================================================

MIN_BRACKETS = 3
MAX_BRACKETS = 8


def is_grammar_testable(sent: str) -> bool:
    """어법 출제 후보가 될 수 있는 문장인지 대략 판정.

    프롬프트의 <no_candidate_sentences>와 같은 기준. 정밀 판정은 모델이 하고,
    여기서는 명백히 불가능한 문장만 걸러 target 개수를 현실적으로 잡는 용도.
    """
    words = sent.split()
    if len(words) <= 6:
        return False
    # 본동사/조동사/be동사 흔적이 전혀 없으면 제외 (명사구 나열, 의문사 단문 등)
    lowered = " " + sent.lower() + " "
    verbish = any(
        f" {v} " in lowered
        for v in (
            "is", "are", "was", "were", "be", "been", "being", "am",
            "has", "have", "had", "do", "does", "did",
            "can", "could", "will", "would", "may", "might", "should", "must",
        )
    )
    # -ed / -s / -ing 로 끝나는 토큰이 하나라도 있으면 동사 가능성 인정
    inflected = any(
        w.strip(".,;:!?\"'").endswith(("ed", "ing", "s")) for w in words
    )
    return verbish or inflected


def plan_bracket_count(sentences: list[str]) -> tuple[int, list[int]]:
    """(target 개수, 후보 문장 인덱스) 반환.

    호출부 예:
        bracket_count, cand_idx = plan_bracket_count(sentences)
        prompt = PROMPT_TEMPLATE.format(
            sent_count=len(sentences),
            passage=passage,
            bracket_count=bracket_count,
        )
    (bracket_dist_lines 인자는 더 이상 사용하지 않음 — 넘겨도 무시됨)
    """
    cand_idx = [i for i, s in enumerate(sentences) if is_grammar_testable(s)]
    if not cand_idx:
        cand_idx = list(range(len(sentences)))
    target = max(MIN_BRACKETS, min(MAX_BRACKETS, len(cand_idx)))
    target = min(target, len(sentences))  # 문장당 최대 1개이므로 문장 수 초과 불가
    return target, cand_idx
