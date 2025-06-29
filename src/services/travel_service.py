import asyncio
from typing import List

from src.models.schemas import (
    Waypoint,
    OptimizeRouteRequest,
    OptimizedRouteResult,
    GeminiInfoRequest,
    GeminiInfoResponse,
    PlaceDetails, # PlaceDetails 스키마 임포트
)
from src.core.route_optimizer import RouteOptimizer
from src.api.gemini_api import GeminiAPI
from src.utils.common import format_cost

class TravelService:
    def __init__(self):
        self.route_optimizer = RouteOptimizer()
        self.gemini_api = GeminiAPI()

    # request의 경유지 정보인 waypoints를 통해 거리/시간 행렬 추출  
    async def get_optimized_travel_route(self, request: OptimizeRouteRequest) -> OptimizedRouteResult:
        """
        경로 최적화를 수행하고 Gemini API를 통해 전반적인 팁 및 각 장소별 소개를 생성합니다.
        """
        # 각 경유지의 거리/시간 행렬 가져오기
        distances, durations, error_msg = await self.route_optimizer.get_and_parse_distance_matrices(request.waypoints)
        if error_msg:
            # 에러 처리
            raise ValueError(f"Error getting distance matrix: {error_msg}")

        # 경로 최적화 실행
        optimized_path_indices, total_cost, execution_time, cost_unit = self.route_optimizer.optimize_route(
            request, distances, durations
        )
        if not optimized_path_indices:
            # optimize_route에서 실패 시 total_cost에 에러 메시지가 담겨있을 수 있음
            raise ValueError(f"Error optimizing route: {total_cost}") 

        # 3. 최적화된 경로의 장소 이름 리스트 추출
        # `optimized_path_names`와 `optimized_waypoints`는 중복 제거 후 Gemini에 전달하는 것이 효율적
        optimized_waypoints_unique_indices = []
        for idx in optimized_path_indices:
            if idx not in optimized_waypoints_unique_indices:
                optimized_waypoints_unique_indices.append(idx)
        
        # 실제 방문할 장소들의 Waypoint 객체 리스트 (순서대로)
        final_optimized_waypoints: List[Waypoint] = [
            request.waypoints[i] for i in optimized_path_indices
            if i < len(request.waypoints) # 유효한 인덱스만
        ]
        
        # 중복 없는 장소 이름 리스트 (Gemini 요약용)
        optimized_path_names = list(set([wp.name for wp in final_optimized_waypoints]))

        # 4. Gemini API를 이용한 AI 인사이트 생성 (비동기 처리)

        # 4.1. 경로 전체에 대한 AI 요약 및 팁
        formatted_cost_info = format_cost(total_cost, unit=cost_unit)
        # `get_route_summary_and_tips`는 `ai_overall_tips`에 해당하는 정보를 반환한다고 가정
        # `gemini_summary`는 이제 `ai_overall_tips`로 매핑될 것
        ai_overall_tips = await self.gemini_api.get_route_summary_and_tips(
            [wp.name for wp in final_optimized_waypoints if wp.name is not None], # 전체 경로 장소 이름
            formatted_cost_info,
            request.optimization_metric # 최적화 기준 전달하여 더 정확한 팁 생성
        )

        # 4.2. 각 장소별 AI 상세 소개 (비동기적으로 여러 장소 요청)
        # Unique한 장소 이름에 대해서만 정보를 가져오는 것이 중복 호출을 방지
        place_info_tasks = []
        for wp in request.waypoints: # 원래 요청의 모든 웨이포인트에 대해 정보를 가져옴
            # Waypoint 객체를 직접 전달하여 GeminiAPI가 더 많은 컨텍스트 활용 가능하도록 변경
            place_info_tasks.append(
                self.gemini_api.get_place_details_for_route(wp)
            )
        
        # 모든 비동기 작업을 동시에 실행하고 결과를 모음
        ai_place_insights_raw: List[PlaceDetails] = await asyncio.gather(*place_info_tasks)
        
        # 결과 필터링 (Gemini API 호출 실패 등으로 None이 반환될 수 있음)
        ai_place_insights = [
            insight for insight in ai_place_insights_raw if insight is not None
        ]

        # 5. 결과 반환 (OptimizedRouteResult 스키마에 맞춰 모든 필드 채우기)
        return OptimizedRouteResult(
            algorithm=request.algorithm,
            optimized_path_indices=optimized_path_indices,
            optimized_path_names=optimized_path_names,
            total_cost=total_cost,
            cost_unit=cost_unit,
            execution_time_seconds=execution_time,
            map_url=None, # 이 필드는 main.py에서 채워짐 (map_visualizer 호출 후)
            ai_overall_tips=ai_overall_tips,
            ai_place_insights=ai_place_insights
        )

    async def get_gemini_place_info(self, request: GeminiInfoRequest) -> GeminiInfoResponse:
        """
        Gemini API를 통해 특정 장소에 대한 정보를 가져옵니다.
        이것은 /optimize-route와 별개로, 단일 장소에 대한 상세 정보를 요청할 때 사용됩니다.
        """
        # PlaceDetails 스키마와 유사한 정보를 반환하도록 GeminiAPI 메서드를 호출
        # request.place_name, request.country를 Waypoint 객체로 만들어 전달하는 것이 GeminiAPI 활용에 더 유리
        temp_waypoint = Waypoint(name=request.place_name, lat=0.0, lng=0.0, country=request.country) # lat/lng는 더미 값
        
        # gemini_api.get_place_details_for_route는 PlaceDetails 객체를 반환하므로, 이를 GeminiInfoResponse에 맞게 파싱
        place_details_obj: PlaceDetails = await self.gemini_api.get_place_details_for_route(temp_waypoint)
        
        # PlaceDetails 객체에서 요청된 query_type에 맞는 정보를 추출하여 info 필드에 할당
        info_content = ""
        if place_details_obj:
            if request.query_type == "general_info":
                info_content = place_details_obj.general_info or "정보 없음."
            elif request.query_type == "activities":
                info_content = place_details_obj.activities or "활동 정보 없음."
            elif request.query_type == "cultural_insights":
                info_content = place_details_obj.cultural_insights or "문화 정보 없음."
            elif request.query_type == "best_time_to_visit":
                info_content = place_details_obj.best_time_to_visit or "최적 시기 정보 없음."
        
        return GeminiInfoResponse(
            place_name=request.place_name,
            country=request.country,
            query_type=request.query_type,
            info=info_content
        )
