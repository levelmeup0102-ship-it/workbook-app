#!/usr/bin/env python3
"""
qa_check.py — 워크북 HTML 자동 검증 + 수정
파이프라인에서 import해서 사용하거나, 독립 실행 가능

사용법:
  (1) 파이프라인 내부: from qa_check import fix_single_html, fix_merged_html
  (2) 독립 실행:       python qa_check.py output/합본.html
"""
import re, sys
from pathlib import Path

QA_VERSION = "v1.0"

def _log(msg):
    try:
        print(f"  [QA] {msg}")
    except Exception:
        pass


# ============================================================
# 단건 HTML 검증 (render_pdf 직후, 개별 워크북)
# ============================================================

def _fix_error_count_mismatch(html: str) -> str:
    """서술형 오류 개수("N곳") vs 실제 답안 박스 수 불일치 수정
    
    문제: 지시문에 "5곳 찾아 고치시오"인데 답안 박스는 4개
    원인: Claude가 선언한 개수 ≠ 실제 생성한 오류 수
    수정: 답안 박스 수에 맞게 "N곳" 숫자 변경
    """
    # 서술형 섹션 찾기: "어법상 틀린 부분을 <span class="b">N곳</span>"
    pattern = r'(어법상 틀린 부분을\s*<span class="b">)(\d+)(곳</span>)'
    
    for match in re.finditer(pattern, html):
        declared = int(match.group(2))
        # 이 섹션 이후의 답안 박스 수 카운트 (grammar-num 패턴)
        section_start = match.end()
        # 다음 page break 또는 다음 Stage 제목까지
        next_page = html.find('page-break-after', section_start)
        next_stage = html.find('Stage 8-3', section_start)
        section_end = min(
            next_page if next_page > 0 else len(html),
            next_stage if next_stage > 0 else len(html)
        )
        section = html[section_start:section_end]
        actual = len(re.findall(r'<span class="grammar-num">\(\d+\)</span>', section))
        
        if actual > 0 and actual != declared:
            _log(f"[내용오류] 서술형: 지시문 {declared}곳 → 답안 박스 {actual}개 → {actual}곳으로 수정")
            old = match.group(0)
            new = f'{match.group(1)}{actual}{match.group(3)}'
            html = html.replace(old, new, 1)
    
    return html


def _fix_bracket_count_mismatch(html: str) -> str:
    """괄호형 개수 불일치 수정
    
    지문 속 (N)[A / B] 개수 vs 답안 박스 수 불일치 시 답안 박스 수를 지문에 맞춤
    (이건 템플릿이 grammar_bracket_count로 답안 박스를 생성하므로,
     파이프라인 단계에서 이미 처리됨. HTML 단에서는 검증만)
    """
    # 괄호형 지문에서 실제 (N)[...] 패턴 수
    bracket_passages = re.findall(r'<div class="psg[^"]*"[^>]*>.*?</div>', html, re.DOTALL)
    for psg in bracket_passages:
        brackets_in_text = len(re.findall(r'\(\d+\)\[', psg))
        if brackets_in_text > 0:
            # 같은 페이지의 답안 박스 수 확인은 복잡하므로 로그만
            _log(f"[검증] 괄호형 지문: {brackets_in_text}개 괄호 확인")
    
    return html


def _fix_answer_bogus_entries(html: str) -> str:
    """정답 페이지: 무의미 항목 제거 (오류→정답이 동일한 경우)
    
    예: <p>I was going to talk->I was going to talk</p> 삭제
    """
    # arrow 패턴: "something->something" (양쪽 동일)
    pattern = r'<p>(.+?)->\1</p>'
    matches = re.findall(pattern, html)
    if matches:
        for m in matches:
            _log(f"[내용오류] 정답: 무의미 항목 제거 '{m}->{m}'")
        html = re.sub(pattern, '', html)
    
    return html


def _fix_typo_스스로(html: str) -> str:
    """오타 수정: '스스로' → '스스로' (이미 맞음, 하지만 혹시 '스스로'가 있으면)
    실제로는 이 단어가 맞는 표현이므로, 추가 오타만 체크
    """
    # 현재 특별한 오타 패턴 없음 - 향후 추가
    return html


def fix_single_html(html: str) -> str:
    """단건 HTML 전체 검증 + 수정 (파이프라인 render_pdf에서 호출)"""
    _log(f"단건 검증 시작 (QA {QA_VERSION})")
    
    html = _fix_error_count_mismatch(html)
    html = _fix_bracket_count_mismatch(html)
    html = _fix_answer_bogus_entries(html)
    
    _log("단건 검증 완료")
    return html


# ============================================================
# 합본 HTML 검증 (merge_html_files 직후)
# ============================================================

def _fix_duplicate_ids(html: str) -> str:
    """중복 ID 고유화
    
    문제: lv83-content, lv83-psg가 모든 섹션에서 동일
    영향: 인쇄 시 폰트 자동조절 스크립트가 첫 번째 섹션에만 작동
    수정: lv83-content → lv83-content-1, lv83-content-2, ...
    """
    for base_id in ['lv83-content', 'lv83-psg']:
        count = html.count(f'id="{base_id}"')
        if count > 1:
            _log(f"[코드오류] 중복 ID '{base_id}' {count}개 → 고유화")
            counter = [0]
            def _replace(match, _c=counter):
                _c[0] += 1
                return f'id="{base_id}-{_c[0]}"'
            html = re.sub(f'id="{re.escape(base_id)}"', _replace, html)
    
    return html


def _fix_duplicate_scripts(html: str) -> str:
    """중복 script 블록 → 1개로 통합
    
    문제: 동일 beforeprint/afterprint 스크립트가 섹션 수만큼 반복
    수정: 전부 제거 후 고유 ID를 순회하는 통합 스크립트 1개 삽입
    """
    script_pattern = r'<script>\s*\(function\(\)\{.*?\}\)\(\);\s*</script>'
    scripts = list(re.finditer(script_pattern, html, re.DOTALL))
    
    if len(scripts) <= 1:
        return html
    
    _log(f"[코드오류] 중복 script {len(scripts)}개 → 1개로 통합")
    
    # 전부 제거
    for s in reversed(scripts):
        html = html[:s.start()] + html[s.end():]
    
    # lv83-content-N이 몇 개인지 카운트
    id_count = len(re.findall(r'id="lv83-content-\d+"', html))
    if id_count == 0:
        # 고유화 안 된 상태면 기존 ID 카운트
        id_count = len(re.findall(r'id="lv83-content"', html))
    
    # 통합 스크립트 삽입
    consolidated = f'''<script>
(function(){{
  var A4_PX=1123,MARGIN_PX=96,PRINTABLE=A4_PX-MARGIN_PX*2;
  var MIN_FONT=9,MIN_LH=1.8;
  window.addEventListener('beforeprint',function(){{
    for(var i=1;i<={id_count};i++){{
      var wrap=document.getElementById('lv83-content-'+i);
      var psg=document.getElementById('lv83-psg-'+i);
      if(!wrap||!psg)continue;
      psg._origFont=psg.style.fontSize;
      psg._origLH=psg.style.lineHeight;
      var page=wrap.closest('.page');
      if(!page)continue;
      var pageTop=page.offsetTop;
      var font=11,lh=2.8;
      while(font>=MIN_FONT){{
        psg.style.fontSize=font+'pt';
        psg.style.lineHeight=String(lh);
        if(Math.ceil((page.offsetHeight)/PRINTABLE)<=1)break;
        font-=0.5;lh=Math.max(MIN_LH,lh-0.2);
      }}
      if(font<MIN_FONT){{psg.style.fontSize=psg._origFont;psg.style.lineHeight=psg._origLH;}}
    }}
  }});
  window.addEventListener('afterprint',function(){{
    for(var i=1;i<={id_count};i++){{
      var psg=document.getElementById('lv83-psg-'+i);
      if(!psg||!psg._origFont)continue;
      psg.style.fontSize=psg._origFont;psg.style.lineHeight=psg._origLH;
    }}
  }});
}})();
</script>'''
    
    html = html.replace('</body>', consolidated + '\n</body>')
    return html


def _fix_naming_consistency(html: str) -> str:
    """네이밍 불일치 정규화
    
    문제: 01과는 "01과 01", 02과는 "02과 01번" (번 접미사 불일치)
    수정: "번" 접미사 통일 제거 (과 + 숫자만)
    """
    # 커버 + 헤더에서 "NN과 NN번" → "NN과 NN"
    pattern = r'(\d{2}과\s*\d{2})번'
    matches = re.findall(pattern, html)
    if matches:
        unique = set(matches)
        _log(f"[내용오류] 네이밍 불일치: '번' 접미사 {len(unique)}종 제거")
        for m in unique:
            html = html.replace(f'{m}번', m)
    
    return html


def fix_merged_html(html: str) -> str:
    """합본 HTML 전체 검증 + 수정 (merge_html_files에서 호출)"""
    _log(f"합본 검증 시작 (QA {QA_VERSION})")
    
    # 순서 중요: ID 고유화 → script 통합 (script가 ID를 참조하므로)
    html = _fix_duplicate_ids(html)
    html = _fix_duplicate_scripts(html)
    html = _fix_naming_consistency(html)
    # 합본에도 단건 검증 적용
    html = _fix_error_count_mismatch(html)
    html = _fix_answer_bogus_entries(html)
    
    _log("합본 검증 완료")
    return html


# ============================================================
# 독립 실행: python qa_check.py <파일.html>
# ============================================================
def check_file(filepath: str):
    """HTML 파일을 읽어서 검증 + 수정 후 저장"""
    p = Path(filepath)
    if not p.exists():
        print(f"파일 없음: {filepath}")
        return
    
    html = p.read_text(encoding='utf-8')
    original_len = len(html)
    
    # 합본인지 단건인지 판단
    page_count = html.count('class="page"')
    section_count = html.count('class="cv-c"')  # 커버 페이지 수
    
    print(f"파일: {p.name}")
    print(f"  페이지: ~{page_count}개, 섹션: {section_count}개")
    
    if section_count > 1:
        html = fix_merged_html(html)
    else:
        html = fix_single_html(html)
    
    # 변경사항 있으면 저장
    if len(html) != original_len or html != p.read_text(encoding='utf-8'):
        # 원본 백업
        backup = p.with_suffix('.bak.html')
        if not backup.exists():
            p.rename(backup)
            _log(f"원본 백업: {backup.name}")
        
        p.write_text(html, encoding='utf-8')
        _log(f"수정 완료: {p.name}")
    else:
        _log("수정사항 없음")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"qa_check.py {QA_VERSION}")
        print("사용법: python qa_check.py <워크북.html>")
        print("  단건/합본 자동 판별하여 검증 + 수정")
        sys.exit(1)
    
    for f in sys.argv[1:]:
        check_file(f)
