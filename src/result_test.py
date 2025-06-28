from src.api.maps_api import GoogleMapsAPI
from src.algorithms.greedy import GreedySolver
from src.algorithms.simulated_annealing import SimulatedAnnealingSolver
from src.algorithms.genetic_algorithm import GeneticAlgorithmSolver
from src.utils.data_loader import load_waypoints_from_json, save_waypoints_to_json, load_matrix_from_json, save_matrix_to_json
from src.utils.visualizer import MapVisualizer, PlottingVisualizer
from src.utils.common import format_cost
from config.config import config # 설정 파일 로드
import time
import sys

def run_optimization_project():
    print("--- Google Maps API 기반 지역 탐색 알고리즘 비교 분석 프로젝트 시작 ---")

    # 1. Google Maps API 초기화
    gmaps_api = GoogleMapsAPI()

    # 2. 지점 데이터 로드 또는 정의
    # 실제 프로젝트에서는 여기에서 사용자로부터 지점을 입력받거나, 특정 목록을 로드할 수 있습니다.
    # 예시를 위해 임의의 지점 사용 (서울, 부산, 제주, 대전, 광주, 강릉, 대구, 울산)
    waypoints_data = [
        {"name": "Seoul", "lat": 37.5665, "lng": 126.9780},     # 서울 시청
        {"name": "Busan", "lat": 35.1796, "lng": 129.0756},     # 부산역
        {"name": "Jeju", "lat": 33.5097, "lng": 126.5113},      # 제주 국제공항
        {"name": "Daejeon", "lat": 36.3504, "lng": 127.3845},   # 대전역
        {"name": "Gwangju", "lat": 35.1595, "lng": 126.8526},   # 광주역
        {"name": "Gangneung", "lat": 37.7519, "lng": 128.8761}, # 강릉역
        {"name": "Daegu", "lat": 35.8714, "lng": 128.6014},     # 대구역
        {"name": "Ulsan", "lat": 35.5384, "lng": 129.3114}      # 울산역
    ]
    num_waypoints = len(waypoints_data)
    print(f"\n총 {num_waypoints}개의 지점 데이터를 준비했습니다.")

    # 지점 좌표만 추출 (API 호출용)
    waypoint_coords = [(wp['lat'], wp['lng']) for wp in waypoints_data]

    # 3. 거리/시간 행렬 가져오기
    # 캐시에서 로드 시도, 없으면 API 호출
    distance_matrix_raw = load_matrix_from_json(config.DISTANCE_MATRIX_FILE)
    if not distance_matrix_raw:
        distance_matrix_raw = gmaps_api.get_distance_matrix(waypoint_coords, waypoint_coords, mode="driving")
        if distance_matrix_raw:
            save_matrix_to_json(distance_matrix_raw, config.DISTANCE_MATRIX_FILE)
        else:
            print("ERROR: 거리 행렬을 가져오는 데 실패했습니다. 프로그램을 종료합니다.")
            sys.exit(1)

    distances, durations = gmaps_api.parse_distance_matrix(distance_matrix_raw)
    
    # 최적화에 사용할 비용 행렬 선택
    optimization_matrix = None
    if config.DEFAULT_OPTIMIZATION_METRIC == "duration":
        optimization_matrix = durations
        cost_unit = "seconds"
        print("\n최적화 기준으로 '시간(Duration)'을 사용합니다.")
    else: # "distance"
        optimization_matrix = distances
        cost_unit = "meters"
        print("\n최적화 기준으로 '거리(Distance)'를 사용합니다.")

    if not optimization_matrix or any(val == float('inf') for row in optimization_matrix for val in row):
        print("ERROR: 유효한 거리/시간 행렬을 생성할 수 없습니다. 연결되지 않은 지점이 있을 수 있습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 4. 각 알고리즘 실행 및 성능 측정
    results = {} # {알고리즘_이름: (경로, 총_비용, 실행_시간, 비용_히스토리)}

    # Greedy Algorithm
    print("\n--- Greedy Algorithm 실행 중 ---")
    greedy_solver = GreedySolver()
    start_time = time.perf_counter()
    greedy_path, greedy_cost = greedy_solver.solve(optimization_matrix)
    end_time = time.perf_counter()
    greedy_exec_time = end_time - start_time
    results["Greedy"] = (greedy_path, greedy_cost, greedy_exec_time, [])
    print(f"Greedy Path: {greedy_path}")
    print(f"Greedy Total Cost: {format_cost(greedy_cost, unit=cost_unit)}")
    print(f"Greedy Execution Time: {greedy_exec_time:.4f} seconds")

    # Simulated Annealing Algorithm
    print("\n--- Simulated Annealing Algorithm 실행 중 ---")
    sa_solver = SimulatedAnnealingSolver(
        initial_temperature=config.SA_INITIAL_TEMPERATURE,
        cooling_rate=config.SA_COOLING_RATE,
        min_temperature=config.SA_MIN_TEMPERATURE,
        max_iterations=config.SA_MAX_ITERATIONS
    )
    # SA는 비용 히스토리를 반환하도록 solve 메서드를 수정해야 함
    # 현재 코드에서는 _calculate_path_cost 만 있고 히스토리 저장은 없음. (추가 구현 필요)
    # 여기서는 간단히 수정 없이 진행.
    # TODO: SA solver에 history tracking 로직 추가
    sa_path, sa_cost = sa_solver.solve(optimization_matrix) # 히스토리 추가 예정
    sa_exec_time = time.perf_counter() - start_time # 재측정
    results["Simulated Annealing"] = (sa_path, sa_cost, sa_exec_time, []) 
    print(f"Simulated Annealing Path: {sa_path}")
    print(f"Simulated Annealing Total Cost: {format_cost(sa_cost, unit=cost_unit)}")
    print(f"Simulated Annealing Execution Time: {sa_exec_time:.4f} seconds")

    # Genetic Algorithm
    print("\n--- Genetic Algorithm 실행 중 ---")
    ga_solver = GeneticAlgorithmSolver(
        population_size=config.GA_POPULATION_SIZE,
        generations=config.GA_GENERATIONS,
        mutation_rate=config.GA_MUTATION_RATE,
        crossover_rate=config.GA_CROSSOVER_RATE
    )
    # GA도 비용 히스토리를 반환하도록 solve 메서드를 수정해야 함
    # TODO: GA solver에 history tracking 로직 추가
    ga_path, ga_cost = ga_solver.solve(optimization_matrix) # 히스토리 추가 예정
    ga_exec_time = time.perf_counter() - start_time # 재측정
    results["Genetic Algorithm"] = (ga_path, ga_cost, ga_exec_time, [])
    print(f"Genetic Algorithm Path: {ga_path}")
    print(f"Genetic Algorithm Total Cost: {format_cost(ga_cost, unit=cost_unit)}")
    print(f"Genetic Algorithm Execution Time: {ga_exec_time:.4f} seconds")


    # 5. 결과 시각화 및 비교
    map_visualizer = MapVisualizer(output_dir=config.MAPS_OUTPUT_DIR)
    plot_visualizer = PlottingVisualizer(output_dir=config.PLOTS_OUTPUT_DIR)

    all_algo_costs = {}
    print("\n--- 결과 시각화 시작 ---")
    for algo_name, (path, cost, exec_time, history) in results.items():
        if path and cost != float('inf'): # 유효한 경로만 시각화
            map_visualizer.plot_path_on_map(waypoints_data, path, algo_name, format_cost(cost, unit=cost_unit))
            all_algo_costs[algo_name] = cost
        else:
            print(f"Warning: {algo_name} 알고리즘은 유효한 경로를 찾지 못했습니다. 지도를 생성하지 않습니다.")
            all_algo_costs[algo_name] = float('inf') # 비교 그래프에 표시될 수 있도록 inf로 설정

        if history:
            plot_visualizer.plot_algorithm_progress(algo_name, history)

    plot_visualizer.plot_costs_comparison(all_algo_costs)
    
    print("\n--- Google Maps API 기반 지역 탐색 알고리즘 비교 분석 프로젝트 완료 ---")

if __name__ == "__main__":
    # 프로젝트 실행
    # 실행 전에 .env 파일에 Maps_API_KEY를 설정해야 합니다.
    # 예: Maps_API_KEY="YOUR_API_KEY_HERE"
    run_optimization_project()