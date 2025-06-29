import time
import sys
from typing import List, Tuple, Optional
import numpy as np
import asyncio # 비동기 처리를 위한 asyncio 임포트

# src.api.maps_api 대신 src.api.google_maps_api로 이름 변경 (통일성)
from src.api.google_maps_api import GoogleMapsAPI 
from src.algorithms.greedy import GreedySolver
from src.algorithms.simulated_annealing import SimulatedAnnealingSolver
from src.algorithms.genetic_algorithm import GeneticAlgorithmSolver
from src.algorithms.hill_climbing import HillClimbingSolver
from src.models.schemas import Waypoint, OptimizeRouteRequest # Waypoint, OptimizeRouteRequest 스키마 임포트
from src.utils.common import format_cost
from config.config import config # 설정 파일 로드

class RouteOptimizer:
    def __init__(self):
        self.gmaps_api = GoogleMapsAPI()
        # 알고리즘 솔버들은 이제 optimize_route 메서드 내에서 동적으로 초기화됩니다.
        # 따라서 __init__에서는 제거합니다.

    async def get_and_parse_distance_matrices(self, waypoints: List[Waypoint]) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]], Optional[str]]:
        """
        Google Maps API를 통해 거리/시간 행렬을 가져와 파싱합니다.
        Waypoint 리스트의 lat/lng를 사용하여 API를 호출합니다.
        
        Args:
            waypoints (List[Waypoint]): 최적화할 경유지 리스트 (lat, lng 포함).
        Returns:
            Tuple[Optional[List[List[float]]], Optional[List[List[float]]], Optional[str]]:
            거리 행렬, 시간 행렬, 오류 메시지 (없으면 None).
        """
        # Waypoint 객체에서 lat/lng 좌표만 추출
        waypoint_coords = [(wp.lat, wp.lng) for wp in waypoints]
        
        # Google Maps API의 get_distance_matrix는 이제 비동기 함수이므로 await 필요
        # 좌표를 통해 GoogleMapsAPI에서 좌표를 불러옴 (origins, destiantions을 입력으로 받음)
        matrix_result = await self.gmaps_api.get_distance_matrix(waypoint_coords, waypoint_coords)
        
        if not matrix_result or matrix_result.get('status') != 'OK':
            # API 호출 실패 또는 결과 상태가 OK가 아닌 경우
            return None, None, f"Failed to retrieve distance/duration matrix from Google Maps API: {matrix_result.get('error_message', 'Unknown error or API status not OK')}"

        # 불러온 행렬을 파싱하기 쉬운 형태로 변환해주는 메서드 활용
        # distance_matrix와 durations_matrix를 반환 
        distances, durations = self.gmaps_api.parse_distance_matrix(matrix_result)
        
        # 파싱된 행렬의 유효성 검사 (inf 값 여부)
        # 만약 모든 행렬 값이 inf 라면 유효한 경로 X 
        if not distances or not durations or \
           any(val == float('inf') for row in distances for val in row) or \
           any(val == float('inf') for row in durations for val in row):
            return None, None, "Some routes between waypoints are not found (Infinite cost) or matrix parsing failed."

        return distances, durations, None

    def optimize_route(
        self, 
        request: OptimizeRouteRequest, 
        distances: List[List[float]], 
        durations: List[List[float]]
    ) -> Tuple[List[int], float, float, str]:
        """
        지정된 알고리즘과 비용 메트릭을 사용하여 최적 경로를 계산합니다.
        
        Args:
            request (OptimizeRouteRequest): 최적화 요청 정보.
            distances (List[List[float]]): 지점 간 거리 행렬.
            durations (List[List[float]]): 지점 간 시간 행렬.
        Returns:
            Tuple[List[int], float, float, str]:
            최적화된 경로 인덱스, 총 비용, 실행 시간 (초), 비용 단위.
            오류 발생 시 적절한 예외를 발생시킵니다.
        """
        optimization_matrix: List[List[float]]
        cost_unit: str

        # 만약 요청의 최적화 방법이 duration인 경우 
        if request.optimization_metric == "duration":
            optimization_matrix = durations
            cost_unit = "seconds"
        elif request.optimization_metric == "distance":
            optimization_matrix = distances
            cost_unit = "meters"
        else:
            # 이 경우는 OptimizeRouteRequest Pydantic 모델에서 Literal로 이미 검사하지만, 안전을 위해 추가
            raise ValueError("Invalid optimization metric specified.")

        optimized_path_indices: List[int] = []
        total_cost: float = float('inf') # 총 비용
        execution_time: float = 0.0 # 총 실행 시간 

        start_time = time.perf_counter()

        # greedy 알고리즘 
        if request.algorithm == "greedy":
            greedy_solver = GreedySolver() # 매 요청마다 새로 초기화
            optimized_path_indices, total_cost = greedy_solver.solve(optimization_matrix)
        elif request.algorithm == "simulated_annealing":
            # 요청 파라미터가 없으면 config의 기본값 사용
            sa_solver_runtime = SimulatedAnnealingSolver(
                initial_temperature=request.sa_initial_temperature or config.SA_INITIAL_TEMPERATURE,
                cooling_rate=request.sa_cooling_rate or config.SA_COOLING_RATE,
                max_iterations=request.sa_max_iterations or config.SA_MAX_ITERATIONS
            )
            optimized_path_indices, total_cost = sa_solver_runtime.solve(optimization_matrix)
        elif request.algorithm == "genetic_algorithm":
            # 요청 파라미터가 없으면 config의 기본값 사용
            ga_solver_runtime = GeneticAlgorithmSolver(
                population_size=request.ga_population_size or config.GA_POPULATION_SIZE,
                generations=request.ga_generations or config.GA_GENERATIONS,
                mutation_rate=request.ga_mutation_rate or config.GA_MUTATION_RATE,
                crossover_rate=request.ga_crossover_rate or config.GA_CROSSOVER_RATE
            )
            optimized_path_indices, total_cost = ga_solver_runtime.solve(optimization_matrix)
        elif request.algorithm == "hill_climbing":
            hc_solver = HillClimbingSolver(
                max_iterations=request.hc_max_iterations or config.HC_MAX_ITERATIONS
            )
            optimized_path_indices, total_cost, _ = hc_solver_runtime.solve(optimization_matrix)
        else:
            # 이 경우도 Pydantic 모델에서 Literal로 이미 검사하지만, 안전을 위해 추가
            raise ValueError(f"Invalid algorithm specified: {request.algorithm}")

        execution_time = time.perf_counter() - start_time

        if total_cost == float('inf') or not optimized_path_indices:
            # 알고리즘이 유효한 경로를 찾지 못했거나 비용이 무한대인 경우
            raise ValueError("Optimization algorithm failed to find a valid path or returned infinite cost.")
        
        # 알고리즘이 TSP (시작점으로 돌아오는) 형식으로 반환하는지 확인
        # 만약 알고리즘 솔버가 스스로 시작점으로 돌아오는 경로를 포함하지 않는다면,
        # 여기에 첫 번째 지점을 마지막에 추가하는 로직을 넣을 수 있습니다.
        # 예: if optimized_path_indices and optimized_path_indices[0] != optimized_path_indices[-1]:
        #         optimized_path_indices.append(optimized_path_indices[0])
        # 현재는 알고리즘 솔버가 이 부분을 처리한다고 가정하거나,
        # TSP가 아닌 다른 경로 (가장 짧은 단일 경로)를 찾도록 의도했다고 가정합니다.
        # (TSP 문제 정의에 따라 달라짐)

        return optimized_path_indices, total_cost, execution_time, cost_unit

