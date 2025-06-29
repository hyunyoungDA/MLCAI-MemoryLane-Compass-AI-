import random
from typing import List, Tuple

class HillClimbingSolver:
    def __init__(self, max_iterations: int = 10000):
        """
        힐 클라이밍(Hill Climbing) 알고리즘 솔버를 초기화합니다.

        Args:
            max_iterations (int): 최대 반복 횟수.
        """
        self.max_iterations = max_iterations

    def _calculate_path_cost(self, path: List[int], distance_matrix: List[List[float]]) -> float:
        """
        경로의 총 비용을 계산합니다. TSP 문제이므로 마지막 지점에서 시작 지점으로 돌아오는 비용을 포함합니다.

        Args:
            path (List[int]): 지점 인덱스 리스트.
            distance_matrix (List[List[float]]): N x N 형태의 거리/시간 행렬.
        Returns:
            float: 경로의 총 비용.
        """
        cost = 0.0
        num_points = len(path)
        
        if num_points < 2:
            return 0.0 # 0개 또는 1개 지점 경로의 비용은 0

        # 모든 지점 간의 비용 합산
        for i in range(num_points):
            start_node = path[i]
            end_node = path[(i + 1) % num_points] # 마지막에서 시작으로 돌아오기 위해 모듈러 연산

            dist = distance_matrix[start_node][end_node]
            if dist == float('inf'): # 연결되지 않은 경로
                return float('inf') # 무한대 비용 반환
            cost += dist
        
        return cost

    def _generate_neighbor(self, current_path: List[int]) -> List[int]:
        """
        현재 경로의 이웃 경로를 생성합니다 (2-opt swap).
        
        Args:
            current_path (List[int]): 현재 경로.
        Returns:
            List[int]: 이웃 경로.
        """
        num_points = len(current_path)
        if num_points < 2:
            return list(current_path) # 변경할 수 없는 경우 원본 반환

        # 무작위로 두 지점 선택
        idx1, idx2 = random.sample(range(num_points), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1 # 항상 idx1 < idx2 유지

        # 2-opt swap: 선택된 두 지점 사이의 경로를 역순으로 뒤집기
        new_path = current_path[:idx1] + current_path[idx1:idx2+1][::-1] + current_path[idx2+1:]
        return new_path

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float, List[float]]:
        """
        힐 클라이밍 알고리즘을 사용하여 TSP 문제를 해결합니다.

        Args:
            distance_matrix (List[List[float]]): N x N 형태의 거리/시간 행렬 (인덱스 기준).
        Returns:
            Tuple[List[int], float, List[float]]:
            (최적 경로 인덱스, 총 최적 비용, 반복별 최적 비용 이력) 튜플.
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0.0, []
        if num_points == 1:
            return [0], 0.0, [0.0]

        # 초기 해 생성 (무작위 순열)
        current_path: List[int] = list(range(num_points))
        random.shuffle(current_path)
        current_cost: float = self._calculate_path_cost(current_path, distance_matrix)

        # 초기 해가 유효하지 않은 경우 처리
        if current_cost == float('inf'):
            print("HillClimbingSolver Warning: 초기 경로가 유효하지 않습니다 (연결되지 않은 지점).")
            # 이 경우 최적 경로를 찾을 수 없으므로, 초기 히스토리만 반환
            return [], float('inf'), [float('inf')] * self.max_iterations

        best_path: List[int] = list(current_path)
        best_cost: float = current_cost
        cost_history: List[float] = [] # 반복별 최적 비용을 저장할 리스트

        for iteration in range(self.max_iterations):
            cost_history.append(best_cost) # 현재까지의 최적 비용을 히스토리에 기록

            # 이웃 해 생성
            neighbor_path: List[int] = self._generate_neighbor(current_path)
            neighbor_cost: float = self._calculate_path_cost(neighbor_path, distance_matrix)

            # 이웃 해가 더 좋으면 수용
            if neighbor_cost < current_cost:
                current_path = list(neighbor_path)
                current_cost = neighbor_cost
                # 전체 최적 해 업데이트
                if current_cost < best_cost: # 사실상 항상 참이겠지만 명시적 체크
                    best_path = list(current_path)
                    best_cost = current_cost
            else:
                # 더 이상 개선할 이웃이 없으면 (지역 최적해에 도달) 반복 종료
                # 힐 클라이밍의 기본 원칙은 '오르막'만 이동하는 것이므로,
                # 나쁜 해는 수용하지 않고, 더 이상 좋은 이웃이 없으면 멈춥니다.
                break 

        return best_path, best_cost, cost_history

# 예시 사용
if __name__ == "__main__":
    # 간단한 거리 행렬 예시 (0 -> 1 -> 2 -> 0)
    test_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    # 최적 경로 (예상): [0, 1, 3, 2, 0] 또는 [0, 2, 3, 1, 0] 등

    print("--- Hill Climbing Algorithm 테스트 ---")
    solver = HillClimbingSolver(max_iterations=5000) # 반복 횟수 설정
    path, dist, history = solver.solve(test_matrix)
    print(f"Hill Climbing Path: {path}, Total Distance: {dist}")
    print(f"Cost History (first 10 values): {history[:10]}") # 초반 변화 확인
    print(f"Cost History (last 10 values): {history[-10:]}") # 수렴 확인
    
    # 연결되지 않은 경로 테스트
    print("\n--- 연결되지 않은 경로 테스트 (inf 값 포함) ---")
    disconnected_matrix = [
        [0, 10, float('inf')],
        [10, 0, 5],
        [float('inf'), 5, 0]
    ]
    path_disc, dist_disc, history_disc = solver.solve(disconnected_matrix)
    print(f"Disconnected Path: {path_disc}, Total Distance: {dist_disc}")
    print(f"Cost History: {history_disc}") # 빈 리스트 또는 inf로 채워질 예상

