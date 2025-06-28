from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os

from src.models.schemas import OptimizeRouteRequest, OptimizedRouteResult, Waypoint, GeminiInfoRequest, GeminiInfoResponse
from src.services.travel_service import TravelService
from src.utils.visualizer import MapVisualizer
from config.config import config

load_dotenv() # .env 파일 로드

app = FastAPI(
    title="GMemoryLane Compass AI",
    description="Optimizes travel routes using various algorithms and provides AI-generated insights for global destinations.",
    version="1.0.0"
)

# 서비스 초기화
travel_service = TravelService()
map_visualizer = MapVisualizer(output_dir=config.MAPS_OUTPUT_DIR) # 지도 시각화 도구

# 정적 파일 서빙 (HTML, CSS, JS 등)
# app.mount("/static", StaticFiles(directory="web_ui/static"), name="static") # Streamlit/Gradio 사용 시 불필요


@app.post("/optimize-route", response_model=OptimizedRouteResult, summary="Optimize a travel route for given waypoints")
async def optimize_route(request_body: OptimizeRouteRequest):
    """
    Optimizes a travel route using the specified algorithm and metric.
    Provides an AI-generated summary and tips for the optimized route.
    """
    try:
        # Pydantic 모델에 포함된 파라미터는 자동으로 유효성 검사됨
        result = await travel_service.get_optimized_travel_route(request_body)
        
        # 지도 생성 및 저장 (백엔드에서 처리)
        # Waypoint 객체 리스트를 dict 리스트로 변환하여 visualizer에 전달
        waypoints_for_map = [wp.model_dump() for wp in request_body.waypoints] # Pydantic v2 .model_dump()
        map_visualizer.plot_path_on_map(
            waypoints_for_map,
            result.optimized_path_indices,
            result.algorithm,
            result.total_cost,
            output_filename=f"{result.algorithm.replace(' ', '_').lower()}_{result.total_cost:.0f}.html"
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/get-gemini-info", response_model=GeminiInfoResponse, summary="Get AI-generated information about a specific place")
async def get_gemini_info(request_body: GeminiInfoRequest):
    """
    Retrieves AI-generated information (general, activities, cultural, best time) about a place using Gemini API.
    """
    try:
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