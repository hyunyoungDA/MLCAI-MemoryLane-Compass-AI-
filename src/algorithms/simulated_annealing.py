import random
import math
import sys

class SimulatedAnnealingSolver:
    def __init__(self, initial_temperature=10000, cooling_rate=0.99, min_temperature=0.1, max_iterations=10000):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations

    def _calculate_path_cost(self, path, distance_matrix):
        """경로의 총 비용을 계산합니다."""
        cost = 0
        num_points = len(path)
        if num_points < 2:
            return 0
        
        for i in range(num_points - 1):
            dist = distance_matrix[path[i]][path[i+1]]
            if dist == float('inf'): # 연결되지 않은 경로
                return float('inf')
            cost += dist
        
        # TSP: 마지막 지점에서 시작 지점으로 돌아오는 비용 추가
        dist_to_start = distance_matrix[path[-1]][path[0]]
        if dist_to_start == float('inf'):
            return float('inf')
        cost += dist_to_start
        return cost

    def _generate_neighbor(self, current_path):
        """현재 경로의 이웃 경로를 생성합니다 (2-opt swap)."""
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

    def solve(self, distance_matrix):
        """
        Simulated Annealing 알고리즘을 사용하여 TSP 문제를 해결합니다.
        
        :param distance_matrix: N x N 형태의 거리 행렬 (인덱스 기준)
        :return: (최적 경로, 총 거리) 튜플
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0
        if num_points == 1:
            return [0], 0

        # 초기 해 생성 (무작위 순열)
        current_path = list(range(num_points))
        random.shuffle(current_path)
        current_cost = self._calculate_path_cost(current_path, distance_matrix)

        best_path = list(current_path)
        best_cost = current_cost

        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            if temperature < self.min_temperature:
                break

            # 이웃 해 생성
            new_path = self._generate_neighbor(current_path)
            new_cost = self._calculate_path_cost(new_path, distance_matrix)

            # 비용 개선 또는 확률적 수용
            if new_cost < current_cost:
                current_path = list(new_path)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_path = list(new_path)
                    best_cost = new_cost
            else:
                # 나쁜 해라도 일정 확률로 수용
                acceptance_probability = math.exp((current_cost - new_cost) / temperature)
                if random.random() < acceptance_probability:
                    current_path = list(new_path)
                    current_cost = new_cost
            
            # 온도 냉각
            temperature *= self.cooling_rate
        
        # 마지막으로 찾은 최적 경로가 유효한지 확인
        if best_cost == float('inf'):
            print("SA Warning: 유효한 경로를 찾지 못했습니다.")
            return [], float('inf')

        return best_path, best_cost

# 예시 사용
if __name__ == "__main__":
    # 위 Greedy 예시와 동일한 거리 행렬
    test_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    solver = SimulatedAnnealingSolver(
        initial_temperature=100, 
        cooling_rate=0.95, 
        max_iterations=1000
    )
    path, dist = solver.solve(test_matrix)
    print(f"Simulated Annealing Path: {path}, Total Distance: {dist}")
    # SA는 무작위성이 있어 결과가 실행마다 다를 수 있습니다.
    # 이 예시에서는 [0, 1, 3, 2, 0] 또는 [0, 2, 3, 1, 0]과 같은 경로가 나올 수 있습니다.