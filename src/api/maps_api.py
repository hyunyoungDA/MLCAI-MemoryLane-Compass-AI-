import googlemaps
import os
import json
from datetime import datetime
import time

class GoogleMapsAPI:
    def __init__(self):
        # 환경 변수에서 API 키 로드
        self.api_key = os.getenv("Maps_API_KEY")
        if not self.api_key:
            raise ValueError("Maps_API_KEY 환경 변수가 설정되지 않았습니다.")
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.cache_dir = "data/cache" # 캐시 디렉토리
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_filepath(self, query_hash):
        """쿼리 해시를 기반으로 캐시 파일 경로를 생성합니다."""
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    def _load_from_cache(self, query_hash):
        """캐시에서 데이터를 로드합니다."""
        filepath = self._get_cache_filepath(query_hash)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, query_hash, data):
        """데이터를 캐시에 저장합니다."""
        filepath = self._get_cache_filepath(query_hash)
        with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def get_distance_matrix(self, origins, destinations, mode="driving"):
        """
        주어진 출발지 목록과 목적지 목록 간의 거리 및 시간을 반환합니다.
        캐싱 기능을 포함합니다.
        
        :param origins: 출발지 리스트 (예: ["Seoul, Korea", "Busan, Korea"] 또는 [(lat, lng)])
        :param destinations: 목적지 리스트
        :param mode: 교통 수단 (driving, walking, bicycling, transit)
        :return: 거리 및 시간 행렬 딕셔너리
        """
        # 쿼리 해시 생성 (캐싱을 위해)
        query_hash = hash(frozenset(origins) ^ frozenset(destinations) ^ hash(mode))
        cached_data = self._load_from_cache(query_hash)

        if cached_data:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시에서 거리 행렬 로드 완료.")
            return cached_data

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Google Maps Distance Matrix API 호출 중...")
        try:
            # API 호출
            # traffic_model='best_guess'는 현재 교통 상황을 반영합니다.
            # departure_time='now'는 현재 시간을 기준으로 최적의 경로를 찾습니다.
            result = self.gmaps.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                units="metric", # 미터/킬로미터 사용
                traffic_model="best_guess",
                departure_time="now" # 현재 시간 기준으로 교통량 반영
            )
            self._save_to_cache(query_hash, result)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API 호출 완료 및 캐시 저장.")
            return result
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Google Maps API 호출 중 오류 발생: {e}")
            return None

    def parse_distance_matrix(self, matrix_result):
        """
        Google Maps API의 거리 행렬 결과를 파싱하여 사용하기 쉬운 형태로 변환합니다.
        반환 값은 거리 (meters) 및 시간 (seconds) 행렬입니다.
        
        :param matrix_result: get_distance_matrix에서 반환된 원본 결과
        :return: (distance_matrix, duration_matrix) 튜플
        """
        if not matrix_result or "rows" not in matrix_result:
            return None, None

        num_origins = len(matrix_result["origin_addresses"])
        num_destinations = len(matrix_result["destination_addresses"])

        distance_matrix = [[0 for _ in range(num_destinations)] for _ in range(num_origins)]
        duration_matrix = [[0 for _ in range(num_destinations)] for _ in range(num_origins)]

        for i, row in enumerate(matrix_result["rows"]):
            for j, element in enumerate(row["elements"]):
                if element["status"] == "OK":
                    distance_matrix[i][j] = element["distance"]["value"]  # meters
                    duration_matrix[i][j] = element["duration"]["value"]  # seconds
                else:
                    # 경로를 찾을 수 없는 경우 무한대 또는 큰 값으로 처리
                    distance_matrix[i][j] = float('inf')
                    duration_matrix[i][j] = float('inf')
                    print(f"경로를 찾을 수 없음: {matrix_result['origin_addresses'][i]} -> {matrix_result['destination_addresses'][j]} ({element['status']})")
        
        return distance_matrix, duration_matrix

# 예시 사용 (main.py에서 호출)
if __name__ == "__main__":
    # .env 파일에서 Maps_API_KEY 로드 (실제 사용 시 dotenv 필요)
    from dotenv import load_dotenv
    load_dotenv()

    gmaps_api = GoogleMapsAPI()

    # 테스트를 위한 임의의 지점
    test_waypoints_coords = [
        (37.5665, 126.9780),  # 서울 시청
        (35.1796, 129.0756),  # 부산역
        (33.5097, 126.5113),  # 제주공항
        (36.3504, 127.3845)   # 대전역
    ]
    test_waypoints_names = [
        "Seoul City Hall",
        "Busan Station",
        "Jeju International Airport",
        "Daejeon Station"
    ]

    # 거리 행렬 가져오기
    # origins와 destinations가 같을 경우 TSP 문제에 활용 가능
    distance_matrix_raw = gmaps_api.get_distance_matrix(test_waypoints_coords, test_waypoints_coords)

    if distance_matrix_raw:
        distances, durations = gmaps_api.parse_distance_matrix(distance_matrix_raw)
        if distances and durations:
            print("\n거리 행렬 (미터):")
            for row in distances:
                print(row)
            print("\n소요 시간 행렬 (초):")
            for row in durations:
                print(row)
        else:
            print("거리/시간 행렬 파싱 실패.")
    else:
        print("거리 행렬 데이터를 가져오지 못했습니다.")