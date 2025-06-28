import json
import os

def load_waypoints_from_json(filepath):
    """
    JSON 파일에서 지점(waypoints) 데이터를 로드합니다.
    :param filepath: JSON 파일 경로
    :return: 지점 리스트 (예: [{"name": "Seoul", "lat": 37.xxx, "lng": 126.xxx}, ...])
    """
    if not os.path.exists(filepath):
        print(f"Error: Waypoints file not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            waypoints = json.load(f)
            # 기본적인 유효성 검사
            if not isinstance(waypoints, list) or not all(isinstance(wp, dict) and "lat" in wp and "lng" in wp for wp in waypoints):
                raise ValueError("Invalid waypoints data format.")
            return waypoints
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except ValueError as e:
        print(f"Error loading waypoints: {e}")
        return None

def save_waypoints_to_json(waypoints, filepath):
    """
    지점(waypoints) 데이터를 JSON 파일로 저장합니다.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(waypoints, f, ensure_ascii=False, indent=4)
        print(f"Waypoints saved to {filepath}")
    except Exception as e:
        print(f"Error saving waypoints to {filepath}: {e}")

def load_matrix_from_json(filepath):
    """
    JSON 파일에서 거리/시간 행렬 데이터를 로드합니다.
    :param filepath: JSON 파일 경로
    :return: 행렬 딕셔너리 또는 None
    """
    if not os.path.exists(filepath):
        print(f"Error: Matrix file not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            matrix_data = json.load(f)
            return matrix_data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None

def save_matrix_to_json(matrix_data, filepath):
    """
    거리/시간 행렬 데이터를 JSON 파일로 저장합니다.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, ensure_ascii=False, indent=4)
        print(f"Matrix saved to {filepath}")
    except Exception as e:
        print(f"Error saving matrix to {filepath}: {e}")

# 예시 사용
if __name__ == "__main__":
    # 테스트 데이터
    test_waypoints = [
        {"name": "Location A", "lat": 37.0, "lng": 127.0},
        {"name": "Location B", "lat": 37.1, "lng": 127.1}
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
    waypoints_file = "data/test_waypoints.json"
    matrix_file = "data/test_matrix.json"

    # 저장
    save_waypoints_to_json(test_waypoints, waypoints_file)
    save_matrix_to_json(test_matrix_data, matrix_file)

    # 로드
    loaded_waypoints = load_waypoints_from_json(waypoints_file)
    loaded_matrix = load_matrix_from_json(matrix_file)

    print("\n로드된 Waypoints:", loaded_waypoints)
    print("로드된 Matrix:", loaded_matrix)