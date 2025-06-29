from typing import List, Tuple

class GreedySolver:
    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float, List[float]]:
        """
        Greedy 알고리즘을 사용하여 TSP 문제를 해결합니다.
        
        Args:
            distance_matrix (List[List[float]]): N x N 형태의 거리/시간 행렬 (인덱스 기준).
        Returns:
            Tuple[List[int], float, List[float]]: 
            (최적 경로 인덱스, 총 비용, 비용 이력(빈 리스트)) 튜플.
        """
        num_points = len(distance_matrix)
        
        # 0개 지점 또는 1개 지점 예외 처리
        if num_points == 0:
            return [], 0.0, []
        if num_points == 1:
            return [0], 0.0, [0.0] # 1개 지점 경로는 자신만 포함, 비용 0

        # 시작점을 0번 인덱스로 고정
        current_path: List[int] = [0]
        visited: List[bool] = [False] * num_points
        visited[0] = True
        total_distance: float = 0.0
        current_node: int = 0

        # 모든 지점을 방문할 때까지 루프
        while len(current_path) < num_points:
            next_node: int = -1
            min_dist: float = float('inf') # 무한대로 초기화

            # 방문하지 않은 지점 중 가장 가까운 지점 찾기
            for i in range(num_points):
                if not visited[i]:
                    dist = distance_matrix[current_node][i]
                    if dist < min_dist:
                        min_dist = dist
                        next_node = i
            
            if next_node != -1 and min_dist != float('inf'): # 유효한 다음 노드를 찾았고 연결되어 있다면
                current_path.append(next_node)
                visited[next_node] = True
                total_distance += min_dist
                current_node = next_node
            else:
                # 더 이상 방문할 노드가 없거나, 연결되지 않은 그래프인 경우
                # 이 시점에서 모든 지점을 방문하지 못했다는 의미
                print("GreedySolver Warning: 경로가 연결되지 않았거나 모든 지점을 방문할 수 없습니다.")
                return [], float('inf'), [] # 유효하지 않은 경로로 반환
        
        # 마지막 노드에서 시작 노드로 돌아오는 거리 추가 (TSP 조건)
        # `current_path[0]` (시작 노드)와 `current_node` (마지막으로 방문한 노드)가 다를 경우
        if num_points > 1: # 1개 지점은 이미 처리됨
            dist_to_start = distance_matrix[current_node][current_path[0]]
            if dist_to_start == float('inf'): # 시작점으로 돌아갈 수 없는 경우
                print("GreedySolver Warning: 최종 지점에서 시작 지점으로 돌아갈 수 없습니다.")
                return [], float('inf'), [] # 유효하지 않은 경로로 반환
            total_distance += dist_to_start
            current_path.append(current_path[0]) # 경로에 시작점 다시 추가
        
        return current_path, total_distance, [] # 탐욕 알고리즘은 히스토리가 없음

# 예시 사용
if __name__ == "__main__":
    # 간단한 거리 행렬 예시 (0 -> 1 -> 2 -> 0)
    # distance_matrix[i][j]는 i에서 j로 가는 거리
    test_matrix = [
        [0, 10, 15, 20],  # 0->x
        [10, 0, 35, 25],  # 1->x
        [15, 35, 0, 30],  # 2->x
        [20, 25, 30, 0]   # 3->x
    ]
    # 최적 경로 (탐욕): 0 -> 1 (10) -> 3 (25) -> 2 (30) -> 0 (15)
    # 총 비용: 10 + 25 + 30 + 15 = 80
    
    print("--- Greedy Algorithm 테스트 ---")
    solver = GreedySolver()
    path, dist, history = solver.solve(test_matrix) # history를 받도록 수정
    print(f"Greedy Path: {path}, Total Distance: {dist}")
    print(f"Cost History: {history}") # 빈 리스트 출력 예상

    # 연결되지 않은 경로 테스트
    print("\n--- 연결되지 않은 경로 테스트 (inf 값 포함) ---")
    disconnected_matrix = [
        [0, 10, float('inf')],
        [10, 0, 5],
        [float('inf'), 5, 0]
    ]
    path_disc, dist_disc, history_disc = solver.solve(disconnected_matrix)
    print(f"Disconnected Path: {path_disc}, Total Distance: {dist_disc}")
    print(f"Cost History: {history_disc}") # 빈 리스트 출력 예상

