"""generation 도메인 요청/응답 모델."""
from typing import Annotated, List, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

# 내부 평탄화 처리용 모델
class GenerateTarget(BaseModel):
    book: str
    unit: str
    passage_id: str
    levels: List[int] | None = None


# unit: `nn강 nn번`에 대한 값을 받을 모델 -> 1개의 unit을 정의한 모델.
class UnitIn(BaseModel):
    unit: NonEmptyStr = Field(
        ...,
        description="강/단원명",
        examples=["22강"],
    )

    passage_ids: List[NonEmptyStr] = Field(
        ...,
        min_length=1,
        description="해당 unit에서 생성할 지문 번호 목록",
        examples=[["02번", "03번", "Gateway"]],
    )

# payload -> 실제 Input으로 들어오는 값 모델
class GenerateIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "book": "26 수특 영어",
                    "units": [
                        {
                            "unit": "22강",
                            "passage_ids": ["01번", "02번", "03번", "Gateway"],
                        },
                        {
                            "unit": "23강",
                            "passage_ids": ["01번", "02번"],
                        },
                    ],
                    "levels": [0, 5, 6, 7, 8, 9, 10],
                }
            ]
        }
    )

    book: NonEmptyStr = Field(
        ...,
        description="교재명. 한 요청은 하나의 book에 대해서만 처리함.",
        examples=["26 수특 영어"],
    )

    units: List[UnitIn] = Field(
        ...,
        min_length=1,
        description="생성할 unit 목록. 각 unit은 여러 passage_id를 가질 수 있음.",
    )

    levels: List[int] | None = Field(
        default=None,
        description=(
            "생성할 level들의 목록."
            "null이면 전체 level 생성, List[int]이면 유저가 선택한 level만 생성. "
            "1개의 요청 안의 모든 지문에 동일하게 적용됨."
        ),
        examples=[[0, 5, 6, 7, 8, 9, 10], None],
    )



# 생성 후, 완성 결과 모델
class GenerateItemOut(BaseModel):
    ok: bool = Field(..., description="결과 생성 성공 여부")
    html: str | None = Field(default=None, description="생성된 HTML")
    filename: str | None = Field(default=None, description="다운로드 파일명")

# 최종 Output 모델
class GenerateOut(BaseModel):
    results: List[GenerateItemOut] = Field(..., description="생성 결과 목록")


# ============================================================
# 대기열(Job) 방식 응답 모델 — 지문이 많아 sync 처리 시 timeout 되는 경우 대비
# ============================================================
class GenerateProgressOut(BaseModel):
    """job 진행 상황 집계."""
    total: int = Field(..., description="전체 task(지문) 수")
    pending: int = Field(..., description="대기 중")
    processing: int = Field(..., description="처리 중")
    completed: int = Field(..., description="완료")
    failed: int = Field(..., description="실패")


class GenerateResponseOut(BaseModel):
    """POST /api/generate 통합 응답. mode=sync(즉시 결과) / mode=job(대기열 등록)."""
    mode: Literal["sync", "job"] = Field(..., description="처리 방식")
    done: bool = Field(..., description="완료 여부(sync는 항상 True)")
    job_id: str | None = Field(default=None, description="job 모드일 때만 발급")
    progress: GenerateProgressOut | None = Field(default=None, description="job 모드 진행상황")
    results: List[GenerateItemOut] | None = Field(default=None, description="sync 모드 결과")


class GenerateJobStatusOut(BaseModel):
    """GET /api/generate/status/{job_id} 응답 — 진행 상황만."""
    job_id: str
    status: str = Field(..., description="job 상태(pending/processing/done 등)")
    done: bool
    progress: GenerateProgressOut


class GenerateJobResultsOut(BaseModel):
    """GET /api/generate/results/{job_id} 응답 — 완료 시 결과(task_index 순)."""
    job_id: str
    status: str
    done: bool
    results: List[GenerateItemOut] | None = Field(default=None, description="done=True일 때만 채움")