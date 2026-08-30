"""LLM 응답 JSON 견고 파싱 유틸.

pipeline.py 원본(_parse_json_robust / _fix_json_quotes)에서 복사.
(pipeline 원본은 당분간 유지 — 나중에 이 모듈로 통합 예정)

    from utils.json_parser import parse_json_robust
"""
import re
import json


def parse_json_robust(text: str) -> dict:
    """여러 전략으로 JSON 파싱 시도 (구 pipeline._parse_json_robust)."""
    # 1) 코드블록 제거
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    text = text.strip()

    # 2) 직접 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) JSON 부분만 추출 (가장 바깥 { } 매칭)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 4) 이스케이프 안 된 따옴표 수정: value 안의 " → \"
    try:
        fixed = _fix_json_quotes(text if not match else match.group())
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    # 5) 줄바꿈/탭 이스케이프
    try:
        cleaned = text if not match else match.group()
        cleaned = re.sub(r'(?<=": ")([^"]*?)(?=")',
                         lambda m: m.group(1).replace('\n', '\\n').replace('\t', '\\t'), cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, Exception):
        pass

    raise json.JSONDecodeError("모든 파싱 전략 실패", text[:200], 0)


def _fix_json_quotes(text: str) -> str:
    """JSON 문자열 안의 이스케이프 안 된 따옴표를 수정 (구 pipeline._fix_json_quotes)."""
    result = []
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                rest = text[i + 1:i + 10].lstrip()
                if not rest or rest[0] in ',}]:':
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')
                    continue
        else:
            result.append(ch)
    return ''.join(result)
