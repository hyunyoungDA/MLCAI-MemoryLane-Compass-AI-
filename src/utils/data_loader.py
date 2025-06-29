import json
import os
from typing import List, Dict, Any, Optional

def load_waypoints_from_json(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """
    JSON 파일에서 지점(waypoints) 데이터를 로드합니다.
    
    Args:
        filepath (str): JSON 파일 경로.
    Returns:
        Optional[List[Dict[str, Any]]]: 지점 리스트 (예: [{"name": "Seoul", "lat": 37.xxx, "lng": 126.xxx}, ...]), 
                                       로드 실패 시 None.
    """
    if not os.path.exists(filepath):
        print(f"[DataLoader Error] Waypoints file not found at: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            waypoints_data = json.load(f)
            # 기본적인 유효성 검사: 리스트 형태이고, 각 항목이 딕셔너리이며 'lat', 'lng', 'name' 키를 포함하는지
            if not isinstance(waypoints_data, list) or \
               not all(isinstance(wp, dict) and 
                       "lat" in wp and "lng" in wp and "name" in wp for wp in waypoints_data):
                raise ValueError("Invalid waypoints data format: Must be a list of dictionaries with 'name', 'lat', 'lng'.")
            return waypoints_data
    except json.JSONDecodeError as e:
        print(f"[DataLoader Error] Invalid JSON format in {filepath}: {e}")
        return None
    except ValueError as e:
        print(f"[DataLoader Error] Failed to load waypoints due to data validation: {e}")
        return None
    except Exception as e:
        print(f"[DataLoader Error] Unexpected error loading waypoints from {filepath}: {e}")
        return None

def save_waypoints_to_json(waypoints: List[Dict[str, Any]], filepath: str):
    """
    지점(waypoints) 데이터를 JSON 파일로 저장합니다.
    
    Args:
        waypoints (List[Dict[str, Any]]): 저장할 지점 리스트.
        filepath (str): JSON 파일을 저장할 경로.
    """
    try:
        # 디렉토리가 없으면 생성
        dir_name = os.path.dirname(filepath)
        if dir_name: # 파일 경로에 디렉토리 정보가 있는 경우에만 생성 시도
            os.makedirs(dir_name, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(waypoints, f, ensure_ascii=False, indent=4)
        print(f"[DataLoader] Waypoints saved to {filepath}")
    except Exception as e:
        print(f"[DataLoader Error] Failed to save waypoints to {filepath}: {e}")

def load_matrix_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    JSON 파일에서 거리/시간 행렬 데이터를 로드합니다.
    
    Args:
        filepath (str): JSON 파일 경로.
    Returns:
        Optional[Dict[str, Any]]: 행렬 데이터 딕셔너리, 로드 실패 시 None.
    """
    if not os.path.exists(filepath):
        print(f"[DataLoader Error] Matrix file not found at: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            matrix_data = json.load(f)
            # 기본적인 유효성 검사 (Google Maps Distance Matrix 응답 구조)
            if not isinstance(matrix_data, dict) or "rows" not in matrix_data or "status" not in matrix_data:
                raise ValueError("Invalid matrix data format: Missing 'rows' or 'status' key.")
            return matrix_data
    except json.JSONDecodeError as e:
        print(f"[DataLoader Error] Invalid JSON format in {filepath}: {e}")
        return None
    except ValueError as e:
        print(f"[DataLoader Error] Failed to load matrix due to data validation: {e}")
        return None
    except Exception as e:
        print(f"[DataLoader Error] Unexpected error loading matrix from {filepath}: {e}")
        return None

def save_matrix_to_json(matrix_data: Dict[str, Any], filepath: str):
    """
    거리/시간 행렬 데이터를 JSON 파일로 저장합니다.
    
    Args:
        matrix_data (Dict[str, Any]): 저장할 행렬 데이터.
        filepath (str): JSON 파일을 저장할 경로.
    """
    try:
        # 디렉토리가 없으면 생성
        dir_name = os.path.dirname(filepath)
        if dir_name: # 파일 경로에 디렉토리 정보가 있는 경우에만 생성 시도
            os.makedirs(dir_name, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, ensure_ascii=False, indent=4)
        print(f"[DataLoader] Matrix saved to {filepath}")
    except Exception as e:
        print(f"[DataLoader Error] Failed to save matrix to {filepath}: {e}")

# 예시 사용
if __name__ == "__main__":
    # 테스트 데이터
    test_waypoints = [
        {"name": "Location A", "lat": 37.0, "lng": 127.0, "country": "KR"},
        {"name": "Location B", "lat": 37.1, "lng": 127.1, "country": "KR"}
    ]
    test_matrix_data = {
        "destination_addresses": ["Location A", "Location B"],
        "origin_addresses": ["Location A", "Location B"],
        "rows": [
            {"elements": [{"distance": {"text": "0 m", "value": 0}, "duration": {"text": "1 min", "value": 60}, "status": "OK"}, {"distance": {"text": "10 km", "value": 10000}, "duration": {"text": "10 min", "value": 600}, "status": "OK"}]},
            {"elements": [{"distance": {"text": "10 km", "value": 10000}, "duration": {"text": "10 min", "value": 600}, "status": "OK"}, {"distance": {"text": "0 m", "value": 0}, "duration": {"text": "1 min", "value": 60}, "status": "OK"}]}
        ],
        "status": "OK"
    }

    # 파일 경로
    # config.py의 캐시 디렉토리와 일관성 유지
    os.makedirs("data/cache/google_maps", exist_ok=True) 
    waypoints_file = "data/cache/waypoints_test.json"
    matrix_file = "data/cache/google_maps/matrix_test.json"

    print("--- Waypoints 저장 및 로드 테스트 ---")
    save_waypoints_to_json(test_waypoints, waypoints_file)
    loaded_waypoints = load_waypoints_from_json(waypoints_file)
    print("로드된 Waypoints:", loaded_waypoints)
    assert loaded_waypoints == test_waypoints, "Waypoints 로드/저장 테스트 실패!"
    print("Waypoints 로드/저장 테스트 성공!")

    print("\n--- Matrix 저장 및 로드 테스트 ---")
    save_matrix_to_json(test_matrix_data, matrix_file)
    loaded_matrix = load_matrix_from_json(matrix_file)
    print("로드된 Matrix:", loaded_matrix)
    assert loaded_matrix == test_matrix_data, "Matrix 로드/저장 테스트 실패!"
    print("Matrix 로드/저장 테스트 성공!")

    print("\n--- 존재하지 않는 파일 로드 테스트 ---")
    non_existent_waypoints = load_waypoints_from_json("data/non_existent.json")
    print("존재하지 않는 Waypoints 파일 로드 결과:", non_existent_waypoints)
    assert non_existent_waypoints is None, "존재하지 않는 파일 로드 테스트 실패!"

    print("\n--- 잘못된 JSON 형식 로드 테스트 ---")
    # 잘못된 JSON 파일 생성
    with open("data/bad_format.json", "w", encoding='utf-8') as f:
        f.write("{invalid json}")
    bad_waypoints = load_waypoints_from_json("data/bad_format.json")
    print("잘못된 형식 Waypoints 파일 로드 결과:", bad_waypoints)
    assert bad_waypoints is None, "잘못된 JSON 형식 로드 테스트 실패!"
    os.remove("data/bad_format.json") # 테스트 후 파일 삭제

    print("\n--- 잘못된 데이터 형식 로드 테스트 ---")
    with open("data/bad_data.json", "w", encoding='utf-8') as f:
        json.dump([{"lat": 1.0}], f) # name 필드 누락
    bad_data_waypoints = load_waypoints_from_json("data/bad_data.json")
    print("잘못된 데이터 형식 Waypoints 파일 로드 결과:", bad_data_waypoints)
    assert bad_data_waypoints is None, "잘못된 데이터 형식 로드 테스트 실패!"
    os.remove("data/bad_data.json") # 테스트 후 파일 삭제

    print("\n모든 data_loader 테스트 완료!")

