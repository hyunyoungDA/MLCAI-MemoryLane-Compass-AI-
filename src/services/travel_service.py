from typing import List
from src.models.schemas import Waypoint, OptimizeRouteRequest, OptimizedRouteResult, GeminiInfoRequest, GeminiInfoResponse
from src.core.route_optimizer import RouteOptimizer
from src.api.gemini_api import GeminiAPI
from src.utils.common import format_cost

class TravelService:
    def __init__(self):
        self.route_optimizer = RouteOptimizer()
        self.gemini_api = GeminiAPI()

    async def get_optimized_travel_route(self, request: OptimizeRouteRequest) -> OptimizedRouteResult:
        """
        경로 최적화를 수행하고 Gemini API를 통해 요약을 생성합니다.
        """
        # 1. 거리/시간 행렬 가져오기
        distances, durations, error_msg = await self.route_optimizer.get_and_parse_distance_matrices(request.waypoints)
        if error_msg:
            # 에러 처리 (적절한 HTTP 예외를 발생시키거나, 에러 응답 객체 반환)
            raise ValueError(f"Error getting distance matrix: {error_msg}")

        # 2. 경로 최적화 실행
        optimized_path_indices, total_cost, execution_time, cost_unit = self.route_optimizer.optimize_route(
            request, distances, durations
        )
        if not optimized_path_indices:
            raise ValueError(f"Error optimizing route: {total_cost}") # total_cost가 에러 메시지를 포함

        # 3. Gemini API를 이용한 경로 요약 생성
        optimized_path_names = [request.waypoints[i].name for i in optimized_path_indices if i < len(request.waypoints)]
        
        # TSP 경로일 경우 마지막 중복된 시작 지점 제거 후 요약에 전달
        if optimized_path_names and optimized_path_names[0] == optimized_path_names[-1] and len(optimized_path_names) > 1:
            summary_places = optimized_path_names[:-1]
        else:
            summary_places = optimized_path_names

        formatted_cost_info = format_cost(total_cost, unit=cost_unit)
        gemini_summary = self.gemini_api.get_route_summary_and_tips(summary_places, formatted_cost_info)

        return OptimizedRouteResult(
            algorithm=request.algorithm,
            optimized_path_indices=optimized_path_indices,
            optimized_path_names=optimized_path_names,
            total_cost=total_cost,
            cost_unit=cost_unit,
            execution_time_seconds=execution_time,
            gemini_summary=gemini_summary
        )

    async def get_gemini_place_info(self, request: GeminiInfoRequest) -> GeminiInfoResponse:
        """
        Gemini API를 통해 특정 장소에 대한 정보를 가져옵니다.
        """
        info = self.gemini_api.get_travel_info(request.place_name, request.country, request.query_type)
        return GeminiInfoResponse(
            place_name=request.place_name,
            country=request.country,
            query_type=request.query_type,
            info=info
        )