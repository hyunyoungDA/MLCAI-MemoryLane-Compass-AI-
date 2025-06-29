import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class AppConfig:
    # Google Maps API 설정
    Maps_API_KEY = os.getenv("Maps_API_KEY")
    
    # 데이터 경로 설정
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    WAYPOINTS_FILE = os.path.join(DATA_DIR, "waypoints.json")
    DISTANCE_MATRIX_FILE = os.path.join(DATA_DIR, "distance_matrix.json")
    DURATION_MATRIX_FILE = os.path.join(DATA_DIR, "duration_matrix.json") # 시간 행렬도 저장할 경우

    # 결과 경로 설정
    RESULTS_DIR = "results"
    MAPS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "maps")
    PLOTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")

    # 알고리즘 파라미터
    # Greedy는 파라미터 없음

    # Simulated Annealing
    SA_INITIAL_TEMPERATURE = 10000
    SA_COOLING_RATE = 0.99
    SA_MIN_TEMPERATURE = 0.1
    SA_MAX_ITERATIONS = 50000

    # Genetic Algorithm
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 1000
    GA_MUTATION_RATE = 0.05
    GA_CROSSOVER_RATE = 0.8
    
    # hill_climbing
    HC_MAX_ITERATIONS = 10000

    # 기타 설정
    DEFAULT_OPTIMIZATION_METRIC = "duration" # "distance" or "duration"
    
    def __init__(self):
        # 모든 출력 디렉토리 생성
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.MAPS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PLOTS_OUTPUT_DIR, exist_ok=True)

# 설정 인스턴스 생성
config = AppConfig()

# 예시 사용
if __name__ == "__main__":
    print(f"Google Maps API Key: {'*' * len(config.Maps_API_KEY) if config.Maps_API_KEY else 'Not Set'}")
    print(f"Waypoints File: {config.WAYPOINTS_FILE}")
    print(f"SA Max Iterations: {config.SA_MAX_ITERATIONS}")
    print(f"GA Population Size: {config.GA_POPULATION_SIZE}")
    print(f"Default Metric: {config.DEFAULT_OPTIMIZATION_METRIC}")