import googlemaps
import os
import json
from datetime import datetime
import hashlib # 캐시 키 생성을 위한 hashlib 추가
import asyncio # 비동기 처리를 위한 asyncio 추가
from typing import List, Tuple, Dict, Any, Optional
import time

class GoogleMapsAPI:
    def __init__(self):
        # 환경 변수에서 API 키 로드 (GOOGLE_MAPS_API_KEY로 통일 권장)
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY") # Maps_API_KEY 대신 GOOGLE_MAPS_API_KEY 사용
        if not self.api_key:
            # API 키가 없으면 초기화 시 오류 발생 (서비스 시작 자체를 막음)
            # 또는 경고를 출력하고 API 호출 시점에 None을 반환하도록 처리할 수도 있습니다.
            # 사용자 요구에 따라, 일단은 raise ValueError로 두겠습니다.
            raise ValueError("GOOGLE_MAPS_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.cache_dir = "data/cache/google_maps" # 캐시 디렉토리 세분화
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_filepath(self, query_hash: str) -> str:
        """쿼리 해시를 기반으로 캐시 파일 경로를 생성합니다."""
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    def _load_from_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """캐시에서 데이터를 로드합니다."""
        filepath = self._get_cache_filepath(query_hash)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 파일 '{filepath}' 손상: {e}. 캐시를 무시합니다.")
                os.remove(filepath) # 손상된 캐시 파일 삭제
        return None

    def _save_to_cache(self, query_hash: str, data: Dict[str, Any]):
        """데이터를 캐시에 저장합니다."""
        filepath = self._get_cache_filepath(query_hash)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 파일 '{filepath}' 저장 중 오류 발생: {e}")

    def _generate_query_hash(self, *args) -> str:
        """주어진 인자들로부터 일관된 해시를 생성합니다."""
        # 인자들을 JSON 문자열로 직렬화하여 해시 생성 (더 견고함)
        data_to_hash = json.dumps(args, sort_keys=True, default=str) # 튜플 등을 문자열로 변환
        return hashlib.md5(data_to_hash.encode('utf-8')).hexdigest()

    async def get_distance_matrix(
        self, 
        origins: List[Tuple[float, float]], 
        destinations: List[Tuple[float, float]], 
        mode: str = "driving"
    ) -> Optional[Dict[str, Any]]:
        """
        주어진 출발지 목록과 목적지 목록 간의 거리 및 시간을 비동기적으로 반환합니다.
        캐싱 기능을 포함합니다.

        Args:
            origins (List[Tuple[float, float]]): 출발지 위경도 리스트.
            destinations (List[Tuple[float, float]]): 목적지 위경도 리스트.
            mode (str): 교통 수단 (driving, walking, bicycling, transit).
        Returns:
            Optional[Dict[str, Any]]: 거리 및 시간 행렬 원본 딕셔너리. 실패 시 None.
        """
        # 쿼리 해시 생성
        query_hash = self._generate_query_hash(origins, destinations, mode)
        cached_data = self._load_from_cache(query_hash)

        if cached_data:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시에서 거리 행렬 로드 완료.")
            return cached_data

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Google Maps Distance Matrix API 호출 중...")
        try:
            # googlemaps 클라이언트가 동기적이므로 asyncio.to_thread 사용
            result = await asyncio.to_thread(
                self.gmaps.distance_matrix,
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

    async def get_coordinates_from_placename(self, placename: str, country: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        장소 이름으로부터 위도, 경도 좌표를 비동기적으로 가져옵니다 (Geocoding API).
        
        Args:
            placename (str): 검색할 장소 이름.
            country (Optional[str]): 검색을 특정 국가로 제한 (ISO 2글자 코드). 예: "KR", "US".
        Returns:
            Optional[Tuple[float, float]]: (위도, 경도) 튜플. 찾지 못하면 None.
        """
        query_hash = self._generate_query_hash(placename, country)
        cached_data = self._load_from_cache(query_hash)

        if cached_data:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시에서 지오코딩 결과 로드 완료.")
            return tuple(cached_data.get('lat_lng')) # 튜플로 다시 변환
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Google Maps Geocoding API 호출 중...")
        try:
            # geocode 호출
            # components를 사용하여 검색 결과를 특정 국가로 제한할 수 있습니다.
            components = {'country': country} if country else {}
            
            result = await asyncio.to_thread(
                self.gmaps.geocode,
                address=placename,
                components=components
            )

            if result and len(result) > 0:
                location = result[0]['geometry']['location']
                lat_lng = (location['lat'], location['lng'])
                self._save_to_cache(query_hash, {'lat_lng': lat_lng}) # 캐시에 저장
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 지오코딩 API 호출 완료 및 캐시 저장.")
                return lat_lng
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] '{placename}'에 대한 좌표를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Google Geocoding API 호출 중 오류 발생: {e}")
            return None

    def parse_distance_matrix(self, matrix_result: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Google Maps API의 거리 행렬 결과를 파싱하여 사용하기 쉬운 형태로 변환합니다.
        반환 값은 거리 (meters) 및 시간 (seconds) 행렬입니다.
        
        Args:
            matrix_result (Dict[str, Any]): get_distance_matrix에서 반환된 원본 결과.
        Returns:
            Tuple[List[List[float]], List[List[float]]]: (distance_matrix, duration_matrix) 튜플.
        """
        if not matrix_result or "rows" not in matrix_result or matrix_result.get('status') != 'OK':
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 유효하지 않거나 오류가 있는 거리 행렬 결과: {matrix_result.get('status')}")
            return [], [] # 유효하지 않은 경우 빈 리스트 반환

        # 출발지와 목적지의 개수 저장해둠 
        num_origins = len(matrix_result.get("origin_addresses", []))
        num_destinations = len(matrix_result.get("destination_addresses", []))

        # 출발지, 목적지 개수만큼 거리, 시간 행렬을 0.0의 값으로 채워서 만들어둠 
        distance_matrix = [[0.0 for _ in range(num_destinations)] for _ in range(num_origins)]
        duration_matrix = [[0.0 for _ in range(num_destinations)] for _ in range(num_origins)]

        for i, row in enumerate(matrix_result["rows"]):
            for j, element in enumerate(row["elements"]):
                if element.get("status") == "OK":
                    distance_matrix[i][j] = element["distance"]["value"]  # meters
                    duration_matrix[i][j] = element["duration"]["value"]  # seconds
                else:
                    # 경로를 찾을 수 없는 경우 무한대 또는 큰 값으로 처리
                    distance_matrix[i][j] = float('inf')
                    duration_matrix[i][j] = float('inf')
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 경로를 찾을 수 없음: {matrix_result['origin_addresses'][i]} -> {matrix_result['destination_addresses'][j]} ({element.get('status')})")
        
        return distance_matrix, duration_matrix

# 예시 사용
if __name__ == "__main__":
    # .env 파일에서 GOOGLE_MAPS_API_KEY 로드
    from dotenv import load_dotenv
    load_dotenv()

    async def run_examples():
        try:
            gmaps_api = GoogleMapsAPI()
        except ValueError as e:
            print(f"API 키 오류: {e}")
            return

        # 1. 지오코딩 예시
        print("\n--- 지오코딩 API 테스트 ---")
        seoul_coords = await gmaps_api.get_coordinates_from_placename("Seoul City Hall", "KR")
        if seoul_coords:
            print(f"서울 시청 좌표: {seoul_coords}")
        
        eiffel_coords = await gmaps_api.get_coordinates_from_placename("Eiffel Tower", "FR")
        if eiffel_coords:
            print(f"에펠탑 좌표: {eiffel_coords}")

        unknown_place = await gmaps_api.get_coordinates_from_placename("NonExistentPlaceXYZ")
        if not unknown_place:
            print("존재하지 않는 장소에 대한 지오코딩 테스트 완료 (None 반환 예상).")

        # 2. 거리 행렬 가져오기 예시
        print("\n--- 거리 행렬 API 테스트 ---")
        test_waypoints_coords = [
            seoul_coords,         # 서울 시청
            eiffel_coords,        # 에펠탑
            (35.1796, 129.0756),  # 부산역 (하드코딩)
            (36.3504, 127.3845)   # 대전역 (하드코딩)
        ]
        
        # 유효한 좌표만 필터링
        test_waypoints_coords = [c for c in test_waypoints_coords if c is not None]

        if len(test_waypoints_coords) >= 2:
            distance_matrix_raw = await gmaps_api.get_distance_matrix(
                test_waypoints_coords, 
                test_waypoints_coords, 
                mode="driving"
            )

            if distance_matrix_raw:
                distances, durations = gmaps_api.parse_distance_matrix(distance_matrix_raw)
                if distances and durations:
                    print("\n거리 행렬 (미터):")
                    for row in distances:
                        print([f"{d:.0f}" for d in row]) # 소수점 없이 출력
                    print("\n소요 시간 행렬 (초):")
                    for row in durations:
                        print([f"{d:.0f}" for d in row]) # 소수점 없이 출력
                else:
                    print("거리/시간 행렬 파싱 실패.")
            else:
                print("거리 행렬 데이터를 가져오지 못했습니다.")
        else:
            print("거리 행렬 테스트를 위한 유효한 좌표가 부족합니다.")

    # asyncio.run 호출
    asyncio.run(run_examples())