"""1단계 테스트 과제 4개와 참고용 자동 점검.

자동 점검 결과는 참고용입니다. 응답 품질은 사람이 직접 읽고 판단하세요.
"""
import json
import re

SYSTEM_PROMPT = "당신은 정확하고 간결하게 답하는 한국어 업무 비서입니다."

DOC = (
    "사내 공지: 2024년 하반기부터 국내 출장 숙박비 한도는 1박 12만 원으로 조정된다. "
    "출장 신청은 출발 3영업일 전까지 전자결재로 제출해야 하며, "
    "영수증은 귀임 후 7일 이내에 정산 시스템에 등록해야 한다."
)

MINUTES = """회의록 (5월 10일)
- 김민수: 다음 주 금요일까지 고객 설문 결과 정리
- 이지은: 신규 서버 견적 3곳 비교 후 월요일 보고
- 박준호: 이번 주 안에 보안 교육 일정 공지"""


def _has_korean(text):
    return bool(re.search(r"[가-힣]", text))


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def check_summary(text):
    ok = _has_korean(text) and ("12만" in text or "120,000" in text)
    return ok, "한국어 응답 + 숙박비 한도(12만 원) 포함 여부"


def check_todo(text):
    names = ["김민수", "이지은", "박준호"]
    found = [n for n in names if n in text]
    return len(found) == len(names), f"담당자 포함: {found}"


def check_json(text):
    data = _extract_json(text)
    if data is None:
        return False, "JSON 파싱 실패"
    expected = {"limit_krw": 120000, "submit_deadline_days": 3, "receipt_deadline_days": 7}
    missing = sorted(set(expected) - set(data))
    if missing:
        return False, f"누락 키: {missing}"
    wrong = {k: data[k] for k in expected
             if str(data[k]).replace(",", "") != str(expected[k])}
    return (not wrong), (f"값 불일치: {wrong}" if wrong else "키와 값 일치")


def check_abstain(text):
    phrases = ["확인할 수 없", "알 수 없", "나와 있지 않", "언급되어 있지 않", "포함되어 있지 않"]
    return any(p in text for p in phrases), "답변 보류 문구 포함 여부"


TASKS = [
    {
        "name": "summary",
        "title": "한국어 문서 요약",
        "prompt": f"다음 문서를 세 문장 이내로 요약하세요.\n\n{DOC}",
        "check": check_summary,
    },
    {
        "name": "todo",
        "title": "회의록 할 일 추출",
        "prompt": f"다음 회의록에서 담당자별 할 일과 기한을 목록으로 정리하세요.\n\n{MINUTES}",
        "check": check_todo,
    },
    {
        "name": "json",
        "title": "JSON 형식 출력",
        "prompt": (
            "다음 문서에서 값을 추출해 JSON 하나만 출력하세요. 설명은 쓰지 마세요.\n"
            "키: limit_krw(숫자), submit_deadline_days(숫자), receipt_deadline_days(숫자)\n\n"
            f"{DOC}"
        ),
        "check": check_json,
    },
    {
        "name": "abstain",
        "title": "근거 없는 질문 답변 보류",
        "prompt": (
            f"아래 문서만 근거로 답하세요.\n\n{DOC}\n\n"
            "질문: 해외 출장 일비는 하루에 얼마입니까?\n"
            "문서에 근거가 없으면 '제공된 문서에서 확인할 수 없습니다'라고 답하세요."
        ),
        "check": check_abstain,
    },
]