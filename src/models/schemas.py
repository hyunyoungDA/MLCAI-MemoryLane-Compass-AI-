from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# 경유지점 스키마 정의 
class Waypoint(BaseModel):
    name: str = Field(..., description="경유지의 이름 (e.g., 'Eiffel Tower', 'Seoul Station').")
    lat: float = Field(..., description="경유지의 위도")
    lng: float = Field(..., description="경유지의 경도")
    country: Optional[str] = Field(None, description="Optional country name for better Gemini context.")

# 경로 최적화 요청에 필요한 입력값  
class OptimizeRouteRequest(BaseModel):
    # 경유지 리스트(최소 2개)
    waypoints: List[Waypoint] = Field(..., min_items=2, description="List of waypoints to optimize the route for.")
    # 알고리즘
    algorithm: Literal["greedy", "simulated_annealing", "genetic_algorithm"] = Field(
        "greedy", description="사용할 최적화 알고리즘: greedy, simulated_annealing, genetic_algorithm"
    )
    optimization_metric: Literal["distance", "duration"] = Field(
        "duration", description="최적화할 기준: 'distance' (미터) 또는 'duration' (초)."
    )
    # 각 알고리즘별 파라미터는 필요에 따라 추가
    # 현재 config 파일에서 정의되어있음; description 추가  
    sa_initial_temperature: Optional[float] = Field(None, description="시뮬레이티드 어닐링: 초기 온도.")
    sa_cooling_rate: Optional[float] = Field(None, description="시뮬레이티드 어닐링: 냉각 비율.")
    sa_max_iterations: Optional[int] = Field(None, description="시뮬레이티드 어닐링: 최대 반복 횟수.")
    ga_population_size: Optional[int] = Field(None, description="유전 알고리즘: 개체군 크기.")
    ga_generations: Optional[int] = Field(None, description="유전 알고리즘: 세대 수.")
    ga_mutation_rate: Optional[float] = Field(None, description="유전 알고리즘: 돌연변이율.")
    ga_crossover_rate: Optional[float] = Field(None, description="유전 알고리즘: 교차율.")
    # 힐 클라이밍 알고리즘 파라미터 추가
    hc_max_iterations: Optional[int] = Field(None, description="힐 클라이밍: 최대 반복 횟수.")

# 각 장소에 대한 AI 생성 상세 정보를 위한 스키마
class PlaceDetails(BaseModel):
    name: str = Field(..., description="장소의 이름.")
    general_info: Optional[str] = Field(None, description="장소에 대한 일반적인 AI 생성 소개.")
    activities: Optional[str] = Field(None, description="장소에서 할 수 있는 활동에 대한 AI 생성 정보.")
    cultural_insights: Optional[str] = Field(None, description="장소의 문화적 배경 또는 중요성에 대한 AI 생성 정보.")
    best_time_to_visit: Optional[str] = Field(None, description="장소 방문에 가장 좋은 시기에 대한 AI 생성 팁.")

# 최적 경로 결과 스키마
class OptimizedRouteResult(BaseModel):
    algorithm: str = Field(..., description="사용된 알고리즘의 이름.")
    optimized_path_indices: List[int] = Field(..., description="최적화된 경유지 인덱스 순서.")
    optimized_path_names: List[str] = Field(..., description="최적화된 경유지 이름 순서.")
    total_cost: float = Field(..., description="최적화된 경로의 총 비용 (미터 또는 초).")
    cost_unit: str = Field(..., description="총 비용의 단위 ('meters' 또는 'seconds').")
    execution_time_seconds: float = Field(..., description="최적화 알고리즘 실행 시간 (초).")
    # 생성된 지도 파일의 URL 포함 가능하도록 수정 
    map_url: Optional[str] = Field(None, description="생성된 지도 HTML 파일의 URL.")
    
    # Gemini AI 통합 필드
    ai_overall_tips: Optional[str] = Field(
        None, 
        description="최적화된 경로 전체에 대한 AI 생성 요약 및 일반적인 팁."
    )
    # 각 장소에 대한 상세 정보 리스트
    ai_place_insights: Optional[List[PlaceDetails]] = Field(
        None, 
        description="경로에 포함된 각 장소에 대한 AI 생성 상세 정보."
    )
    
# Gemini 정보 요청 스키마 
class GeminiInfoRequest(BaseModel):
    place_name: str = Field(..., description="정보를 요청할 장소의 이름.")
    country: Optional[str] = Field(None, description="장소가 위치한 국가 (Gemini 모델의 컨텍스트를 위해).")
    query_type: Literal["general_info", "activities", "cultural_insights", "best_time_to_visit"] = Field(
        "general_info", 
        description="요청할 정보 유형: 'general_info', 'activities', 'cultural_insights', 'best_time_to_visit'."
    )
    
# Gemini 정보 응답 스키마 
class GeminiInfoResponse(BaseModel):
    place_name: str = Field(..., description="정보가 제공된 장소의 이름.")
    country: Optional[str] = Field(None, description="장소가 위치한 국가.")
    query_type: str = Field(..., description="요청된 정보의 유형.")
    info: str = Field(..., description="Gemini AI가 생성한 정보 내용.")