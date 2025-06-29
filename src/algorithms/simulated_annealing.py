import random
import math
from typing import List, Tuple

class SimulatedAnnealingSolver:
    def __init__(self, initial_temperature: float = 10000, cooling_rate: float = 0.99, min_temperature: float = 0.1, max_iterations: int = 10000):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations

    def _calculate_path_cost(self, path: List[int], distance_matrix: List[List[float]]) -> float:
        """
        경로의 총 비용을 계산합니다. TSP 문제이므로 마지막 지점에서 시작 지점으로 돌아오는 비용을 포함합니다.
        """
        cost = 0.0
        num_points = len(path)
        
        # 0개 또는 1개 지점 경로 처리
        if num_points == 0:
            return 0.0 # 비용 없음
        if num_points == 1:
            return 0.0 # 1개 지점은 자기 자신으로 돌아오는 비용 0

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
        """현재 경로의 이웃 경로를 생성합니다 (2-opt swap)."""
        num_points = len(current_path)
        if num_points < 2:
            return list(current_path) # 변경할 수 없는 경우 원본 반환 (길이가 0 또는 1)

        # 무작위로 두 지점 선택
        idx1, idx2 = random.sample(range(num_points), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1 # 항상 idx1 < idx2 유지

        # 2-opt swap: 선택된 두 지점 사이의 경로를 역순으로 뒤집기
        new_path = current_path[:idx1] + current_path[idx1:idx2+1][::-1] + current_path[idx2+1:]
        return new_path

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float, List[float]]:
        """
        Simulated Annealing 알고리즘을 사용하여 TSP 문제를 해결합니다.
        
        Args:
            distance_matrix (List[List[float]]): N x N 형태의 거리/시간 행렬 (인덱스 기준).
        Returns:
            Tuple[List[int], float, List[float]]: 
            (최적 경로 인덱스, 총 최적 비용, 반복별 최적 비용 이력) 튜플.
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0.0, [] # 빈 경로, 비용 0, 빈 히스토리
        if num_points == 1:
            return [0], 0.0, [0.0] # 1개 지점, 비용 0, 히스토리 [0.0]

        # 초기 해 생성 (무작위 순열)
        current_path: List[int] = list(range(num_points))
        random.shuffle(current_path)
        current_cost: float = self._calculate_path_cost(current_path, distance_matrix)

        best_path: List[int] = list(current_path)
        best_cost: float = current_cost
        
        # 비용이 무한대인 경우, 초기 해가 유효하지 않으므로 초기화
        if best_cost == float('inf'):
            # 모든 경로가 연결되지 않은 경우일 수 있음. 이 경우 유효한 해를 찾을 수 없음.
            return [], float('inf'), [float('inf')] * self.max_iterations # 모든 이력을 inf로 채움

        temperature: float = self.initial_temperature
        cost_history: List[float] = [] # 반복별 최적 비용을 저장할 리스트

        for iteration in range(self.max_iterations):
            # 현재 온도의 최적 비용을 히스토리에 추가
            cost_history.append(best_cost)

            if temperature < self.min_temperature:
                break # 온도가 충분히 낮아지면 종료

            # 이웃 해 생성
            new_path: List[int] = self._generate_neighbor(current_path)
            new_cost: float = self._calculate_path_cost(new_path, distance_matrix)

            # 비용 개선 또는 확률적 수용
            if new_cost < current_cost:
                current_path = list(new_path)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_path = list(new_path)
                    best_cost = new_cost
            elif new_cost != float('inf'): # 새로운 비용이 무한대가 아닐 경우에만 수용 확률 계산
                # 나쁜 해라도 일정 확률로 수용
                acceptance_probability: float = math.exp((current_cost - new_cost) / temperature)
                if random.random() < acceptance_probability:
                    current_path = list(new_path)
                    current_cost = new_cost
            
            # 온도 냉각
            temperature *= self.cooling_rate
        
        # 최종적으로 찾은 최적 경로가 유효한지 확인
        if best_cost == float('inf'):
            print("SimulatedAnnealingSolver Warning: 유효한 경로를 찾지 못했습니다.")
            return [], float('inf'), cost_history # 유효하지 않은 경로로 반환

        return best_path, best_cost, cost_history

# 예시 사용
if __name__ == "__main__":
    # 간단한 거리 행렬 예시 (0 -> 1 -> 2 -> 0)
    # distance_matrix[i][j]는 i에서 j로 가는 거리
    test_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    # 최적 경로 (예상): [0, 1, 3, 2, 0] 또는 [0, 2, 3, 1, 0] 등
    # 총 비용: 80 (0->1->3->2->0) or 80 (0->2->3->1->0)

    print("--- Simulated Annealing Algorithm 테스트 ---")
    solver = SimulatedAnnealingSolver(
        initial_temperature=1000, 
        cooling_rate=0.99, 
        min_temperature=0.1, 
        max_iterations=10000
    )
    path, dist, history = solver.solve(test_matrix) # history를 받도록 수정
    print(f"Simulated Annealing Path: {path}, Total Distance: {dist}")
    print(f"Cost History (last 10 values): {history[-10:]}")
    
    # 연결되지 않은 경로 테스트
    print("\n--- 연결되지 않은 경로 테스트 (inf 값 포함) ---")
    disconnected_matrix = [
        [0, 10, float('inf')],
        [10, 0, 5],
        [float('inf'), 5, 0]
    ]
    path_disc, dist_disc, history_disc = solver.solve(disconnected_matrix)
    print(f"Disconnected Path: {path_disc}, Total Distance: {dist_disc}")
    print(f"Cost History (last 10 values): {history_disc[-10:] if history_disc else 'N/A'}")

