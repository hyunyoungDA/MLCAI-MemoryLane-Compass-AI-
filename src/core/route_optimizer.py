import time
import sys
from typing import List, Tuple, Dict, Any
import numpy as np

from src.api.maps_api import GoogleMapsAPI
from src.algorithms.greedy import GreedySolver
from src.algorithms.simulated_annealing import SimulatedAnnealingSolver
from src.algorithms.genetic_algorithm import GeneticAlgorithmSolver
from src.models.schemas import Waypoint, OptimizeRouteRequest
from src.utils.common import format_cost
from config.config import config # 설정 파일 로드

class RouteOptimizer:
    def __init__(self):
        self.gmaps_api = GoogleMapsAPI()
        self.greedy_solver = GreedySolver()
        # TODO: SA, GA 솔버는 FastAPI 요청 시 파라미터를 받아 초기화하도록 수정
        # 현재는 기본값으로 초기화
        self.sa_solver = SimulatedAnnealingSolver(
            initial_temperature=config.SA_INITIAL_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
            max_iterations=config.SA_MAX_ITERATIONS
        )
        self.ga_solver = GeneticAlgorithmSolver(
            population_size=config.GA_POPULATION_SIZE,
            generations=config.GA_GENERATIONS,
            mutation_rate=config.GA_MUTATION_RATE,
            crossover_rate=config.GA_CROSSOVER_RATE
        )

    async def get_and_parse_distance_matrices(self, waypoints: List[Waypoint]):
        """
        Google Maps API를 통해 거리/시간 행렬을 가져와 파싱합니다.
        """
        waypoint_coords = [(wp.lat, wp.lng) for wp in waypoints]
        
        # 캐싱 로직은 Maps_api.py 안에 이미 구현되어 있습니다.
        matrix_result = self.gmaps_api.get_distance_matrix(waypoint_coords, waypoint_coords)
        
        if not matrix_result:
            return None, None, "Failed to retrieve distance/duration matrix from Google Maps API."

        distances, durations = self.gmaps_api.parse_distance_matrix(matrix_result)
        
        # 유효성 검사 (inf 값 여부)
        if any(val == float('inf') for row in distances for val in row) or \
           any(val == float('inf') for row in durations for val in row):
            return None, None, "Some routes between waypoints are not found (Infinite cost)."

        return distances, durations, None

    def optimize_route(self, request: OptimizeRouteRequest, distances: List[List[float]], durations: List[List[float]]):
        """
        지정된 알고리즘과 비용 메트릭을 사용하여 최적 경로를 계산합니다.
        """
        optimization_matrix: List[List[float]] = []
        cost_unit: str = ""

        if request.optimization_metric == "duration":
            optimization_matrix = durations
            cost_unit = "seconds"
        elif request.optimization_metric == "distance":
            optimization_matrix = distances
            cost_unit = "meters"
        else:
            return None, "Invalid optimization metric specified.", ""

        optimized_path_indices: List[int] = []
        total_cost: float = float('inf')
        execution_time: float = 0.0

        start_time = time.perf_counter()

        if request.algorithm == "greedy":
            optimized_path_indices, total_cost = self.greedy_solver.solve(optimization_matrix)
        elif request.algorithm == "simulated_annealing":
            # 요청에 따라 SA 솔버 파라미터 초기화
            sa_solver_runtime = SimulatedAnnealingSolver(
                initial_temperature=request.sa_initial_temperature or config.SA_INITIAL_TEMPERATURE,
                cooling_rate=request.sa_cooling_rate or config.SA_COOLING_RATE,
                max_iterations=request.sa_max_iterations or config.SA_MAX_ITERATIONS
            )
            optimized_path_indices, total_cost = sa_solver_runtime.solve(optimization_matrix)
        elif request.algorithm == "genetic_algorithm":
            # 요청에 따라 GA 솔버 파라미터 초기화
            ga_solver_runtime = GeneticAlgorithmSolver(
                population_size=request.ga_population_size or config.GA_POPULATION_SIZE,
                generations=request.ga_generations or config.GA_GENERATIONS,
                mutation_rate=request.ga_mutation_rate or config.GA_MUTATION_RATE,
                crossover_rate=request.ga_crossover_rate or config.GA_CROSSOVER_RATE
            )
            optimized_path_indices, total_cost = ga_solver_runtime.solve(optimization_matrix)
        else:
            return None, "Invalid algorithm specified.", ""

        execution_time = time.perf_counter() - start_time

        if total_cost == float('inf') or not optimized_path_indices:
            return None, "Algorithm failed to find a valid path.", ""
        
        # 경로가 TSP (시작점으로 돌아오는) 형식으로 반환되지 않았을 경우, 시작점 추가
        # (알고리즘 내부에서 처리하도록 설계하는 것이 더 좋음)
        if optimized_path_indices and optimized_path_indices[0] != optimized_path_indices[-1]:
            #print("Warning: Path does not loop back to start. Appending start node.")
            pass # 알고리즘에서 이미 처리하고 있다고 가정

        return optimized_path_indices, total_cost, execution_time, cost_unit