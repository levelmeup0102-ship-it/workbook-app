"""generation 라우터 — POST /api/generate.

prefix="/api" + dependencies=[Depends(verify)] 로
엔드포인트마다 _verify(request) 호출하던 것을 대체.
"""

from fastapi import APIRouter, Depends, Request

from core.security import _verify
from . import service
from .schemas import GenerateIn, GenerateOut

router = APIRouter(prefix="/api", tags=["generation"], dependencies=[Depends(_verify)])


@router.post("/generate", response_model=GenerateOut, summary="영어 교재 생성")
async def generate(payload: GenerateIn, request: Request):
        return await service.generate(payload, request.app.state.supabase)

# Input값 평탄화
# def normalize_generate_targets(payload: GenerateIn) -> List[GenerateTarget]:
#         targets: List[GenerateTarget] = []
#         for unit_item in payload.units:
#                 for passage_id in unit_item.passage_ids:
#                         targets.append(
#                                 GenerateTarget(
#                                         book=payload.book,
#                                         unit=unit_item.unit,
#                                         passage_id=passage_id,
#                                         levels=payload.levels
#                                         )
#                                 )
#         return targets

# 평탄화된 값 service층에 전달



# @router.post("/generate", response_model=GenerateOut)
# async def generate(generate_request: GenerateIn, request: Request):

#         tasks = []

#         # 들어온 값 평탄화 -> List[GenerateTarget]
#         targets = normalize_generate_targets(generate_request)

#         service_data = await generate_one(generate_request, request.app.state.supabase)
        

#         semaphore = asyncio.Semaphore(10)

#         async def make_task(task):
#                 async with semaphore:
#                         return await task                

#         for target in targets:
#                 tasks.append(make_task(service.generate(target, request.app.state.supabase)))

#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         res = []
        
#         for result in results:
#                 res.append(GenerateItemOut(
#                         ok=result.ok,
#                         html=result.html,
#                         filename=result.filename
#                         ))
                
#         return GenerateOut(
#         results=res
#         )

# @router.post(
#         "/generate",
#         summary="영어 교재 생성",
#         description="1회독 영어 교재를 생성합니다.",
#         response_model=GenerateOut
#         )
# async def generate(generate_request: GenerateIn, request: Request):
#         """영어 교재를 생성합니다. 1회독 교재 생성 API입니다."""
#         return await service.generate(generate_request, request.app.state.supabase)
