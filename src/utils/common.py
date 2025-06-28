import sys

def calculate_path_cost(path, distance_matrix):
    """
    주어진 경로의 총 비용(거리 또는 시간)을 계산합니다.
    (알고리즘 파일에도 동일한 함수가 있지만, 공통 유틸리티로 분리하면 재사용성이 높아집니다.)
    
    :param path: 지점 인덱스 리스트 (예: [0, 1, 2, 0] - TSP의 경우 시작점으로 다시 돌아와야 함)
    :param distance_matrix: N x N 형태의 거리/시간 행렬
    :return: 총 비용 (meters 또는 seconds)
    """
    cost = 0
    num_points = len(path)
    if num_points < 2:
        return 0 # 경로 길이가 짧으면 비용 없음

    for i in range(num_points - 1):
        from_node = path[i]
        to_node = path[i+1]
        
        # 인덱스 유효성 검사
        if from_node < 0 or from_node >= len(distance_matrix) or \
           to_node < 0 or to_node >= len(distance_matrix):
            print(f"Warning: calculate_path_cost에서 유효하지 않은 노드 인덱스 발견: {from_node} -> {to_node}")
            return float('inf') # 유효하지 않은 경로

        dist = distance_matrix[from_node][to_node]
        if dist == float('inf'): # 연결되지 않은 경로
            return float('inf')
        cost += dist
    
    return cost

def format_cost(cost, unit="meters"):
    """
    비용 값을 보기 좋게 포맷팅합니다 (예: 미터 -> km).
    
    :param cost: 비용 값 (float)
    :param unit: 원본 단위 ("meters" 또는 "seconds")
    :return: 포맷팅된 문자열
    """
    if cost == float('inf'):
        return "N/A (No Valid Path)"
    
    if unit == "meters":
        if cost >= 1000:
            return f"{cost / 1000:,.2f} km"
        return f"{cost:,.0f} meters"
    elif unit == "seconds":
        hours = int(cost // 3600)
        minutes = int((cost % 3600) // 60)
        seconds = int(cost % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    else:
        return str(cost)

# 예시 사용
if __name__ == "__main__":
    # 거리 행렬 예시 (meters)
    test_distance_matrix = [
        [0, 10000, 15000],
        [10000, 0, 5000],
        [15000, 5000, 0]
    ]

    # 테스트 경로 (0 -> 1 -> 2 -> 0)
    test_path = [0, 1, 2, 0]
    
    # TSP 경로의 총 비용은 마지막 지점에서 시작 지점으로 돌아오는 비용도 포함해야 합니다.
    # calculate_path_cost 함수는 이 부분을 직접 처리하지 않으므로, 호출 시 주의해야 합니다.
    # 일반적으로 알고리즘 내부에서 TSP 조건을 만족시키도록 경로를 생성하고,
    # 이 함수는 생성된 경로의 '직선' 비용만 계산합니다.
    # GA, SA에서 사용되는 _calculate_path_cost 함수를 참고하세요.

    # 여기서는 "간선" 비용만 계산하는 예시로 활용.
    cost_0_1 = calculate_path_cost([0, 1], test_distance_matrix) # 10000
    cost_1_2 = calculate_path_cost([1, 2], test_distance_matrix) # 5000
    cost_2_0 = calculate_path_cost([2, 0], test_distance_matrix) # 15000

    print(f"Cost 0->1: {format_cost(cost_0_1)}")
    print(f"Cost 1->2: {format_cost(cost_1_2)}")
    print(f"Cost 2->0: {format_cost(cost_2_0)}")

    # TSP 전체 경로 계산 (manual)
    tsp_total_cost = calculate_path_cost([0, 1, 2], test_distance_matrix) + test_distance_matrix[2][0]
    print(f"TSP Total Cost (0->1->2->0): {format_cost(tsp_total_cost)}")

    # 시간 포맷팅 예시
    print(f"Duration: {format_cost(3665, unit='seconds')}") # 1h 1m 5s
    print(f"Duration: {format_cost(125, unit='seconds')}")  # 2m 5s