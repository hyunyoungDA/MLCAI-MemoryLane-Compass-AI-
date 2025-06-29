from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

from src.models.schemas import (
    OptimizeRouteRequest,
    OptimizedRouteResult,
    GeminiInfoRequest,
    GeminiInfoResponse,
    PlaceDetails # 각 장소 세부 정보 스키마 
)

from src.services.travel_service import TravelService
from src.utils.visualizer import MapVisualizer
from config.config import config

load_dotenv() # .env 파일 로드

app = FastAPI(
    title="MemoryLane Compass AI",
    description="Optimizes travel routes using various algorithms and provides AI-generated insights for global destinations.",
    version="1.0.0"
)

# 서비스 초기화
travel_service = TravelService()

# 지도 시각화 도구 초기화 config에서 경로 불러옴;(results/maps)
map_visualizer = MapVisualizer(output_dir=config.MAPS_OUTPUT_DIR)

# 루트 엔드포인트: Welcome 페이지 (선택 사항)
@app.get("/", response_class=HTMLResponse, summary="Welcome to MemoryLaneCompassAI")
async def read_root(request: Request):
    """
    MemoryLaneCompassAI의 환영 페이지를 렌더링합니다.
    """
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MemoryLaneCompassAI</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <style>
                body {
                    font-family: 'Inter', sans-serif;
                    background-color: #f0f4f8;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    background-color: #ffffff;
                    padding: 2.5rem;
                    border-radius: 1rem;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    max-width: 90%;
                    width: 600px;
                }
                .title {
                    color: #2c3e50;
                    font-size: 2.5rem;
                    font-weight: bold;
                    margin-bottom: 1rem;
                }
                .description {
                    color: #5d6d7e;
                    font-size: 1.1rem;
                    margin-bottom: 2rem;
                }
                .link {
                    display: inline-block;
                    background-color: #3498db;
                    color: white;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.5rem;
                    text-decoration: none;
                    font-weight: 600;
                    transition: background-color 0.3s ease;
                }
                .link:hover {
                    background-color: #2980b9;
                }
            </style>
        </head>
        <body>
            <div class="container rounded-xl">
                <h1 class="title">Welcome to MemoryLaneCompassAI</h1>
                <p class="description">
                    Optimize your travel routes with advanced algorithms and get insightful AI-generated tips for your journeys. Create new memories and revisit cherished ones.
                </p>
                <a href="/docs" class="link">Explore API Documentation</a>
            </div>
        </body>
        </html>
    """)


# 경로 최적화 요청 및 지도 생성 
@app.post("/optimize-route", 
          response_model=OptimizedRouteResult, 
          summary="Optimize a travel route for given waypoints"
)
async def optimize_route(request_body: OptimizeRouteRequest):
    """
    지정된 알고리즘과 지표를 사용하여 여행 경로를 최적화합니다.
    최적화된 경로에 대한 AI 생성 요약 및 팁을 제공합니다.
    경로 내 각 장소에 대한 Gemini AI 소개도 포함합니다.
    """
    try:
        # Pydantic 모델에 포함된 파라미터는 자동으로 유효성 검사됨
        result = await travel_service.get_optimized_travel_route(request_body)
        
        # 지도 생성 및 저장 (백엔드에서 처리)
        # Waypoint 객체 리스트를 dict 리스트로 변환하여 visualizer에 전달
        waypoints_for_map = [wp.model_dump() for wp in request_body.waypoints] 
        
        # 파일 이름에 알고리즘 이름과 총 비용을 포함하여 고유하게 만듦
        filename_prefix = result.algorithm.replace(' ', '_').lower()
        map_filename = f"{filename_prefix}_{result.total_cost:.0f}.html"
        
        generated_map_path = map_visualizer.plot_path_on_map(
            waypoints_for_map,
            result.optimized_path_indices,
            result.algorithm,
            result.total_cost,
            output_filename=map_filename
        )
        
        # 지도 생성 시 map의 위치를 저장 
        if generated_map_path:
            result.map_url = f"/maps/{os.path.basename(generated_map_path)}"
        
        else:
            result.map_url = None # 지도 생성 실패 시 None
            
        # Gemini AI 통합: 경로에 대한 종합 팁 및 각 장소 소개 추가
        # TravelService 내부에서 이 정보를 OptimizedRouteResult에 추가했다고 가정
        # (travel_service.py에서 이 로직을 구현해야 함)

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Gemini를 통해 해당 장소에 대한 정보 검색 
@app.post("/get-gemini-info", 
          response_model=GeminiInfoResponse, 
          summary="Get AI-generated information about a specific place")
# GeminiInfoRequest 형식으로 입력 받음 
async def get_gemini_info(request_body: GeminiInfoRequest):
    """
    Gemini API를 사용하여 장소에 대한 AI 생성 정보(일반, 활동, 문화, 최적 시기)를 검색합니다.
    """
    try:
        # gemini가 생성한 장소의 정보 저장 
        response = await travel_service.get_gemini_place_info(request_body)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# 개발 및 테스트를 위한 로컬 실행
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000) # 개발 시 리로드 켜기
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)