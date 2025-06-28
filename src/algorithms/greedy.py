import sys

class GreedySolver:
    def solve(self, distance_matrix):
        """
        Greedy 알고리즘을 사용하여 TSP 문제를 해결합니다.
        
        :param distance_matrix: N x N 형태의 거리 행렬 (인덱스 기준)
        :return: (최적 경로, 총 거리) 튜플
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0
        if num_points == 1:
            return [0], 0

        # 시작점을 0번 인덱스로 고정 (TSP는 시작점에 따라 결과가 달라질 수 있지만, Greedy는 보통 특정 시작점에서 시작)
        current_path = [0]
        visited = [False] * num_points
        visited[0] = True
        total_distance = 0
        current_node = 0

        while len(current_path) < num_points:
            next_node = -1
            min_dist = sys.float_info.max

            for i in range(num_points):
                if not visited[i]:
                    dist = distance_matrix[current_node][i]
                    if dist < min_dist:
                        min_dist = dist
                        next_node = i
            
            if next_node != -1:
                current_path.append(next_node)
                visited[next_node] = True
                total_distance += min_dist
                current_node = next_node
            else:
                # 더 이상 방문할 노드가 없거나, 경로를 찾을 수 없는 경우 (연결되지 않은 그래프)
                break
        
        # 마지막 노드에서 시작 노드로 돌아오는 거리 추가 (TSP 조건)
        if num_points > 1 and current_path[0] != current_node:
            dist_to_start = distance_matrix[current_node][current_path[0]]
            if dist_to_start == float('inf'): # 돌아갈 수 없는 경우
                return [], float('inf')
            total_distance += dist_to_start
            current_path.append(current_path[0]) # 경로에 시작점 다시 추가
        
        # 모든 지점을 방문했는지 확인
        if len(set(current_path[:-1])) != num_points: # 마지막 시작점 제외
            print("Greedy Warning: 모든 지점을 방문하지 못했습니다.")
            return [], float('inf') # 유효하지 않은 경로로 간주

        return current_path, total_distance

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

    solver = GreedySolver()
    path, dist = solver.solve(test_matrix)
    print(f"Greedy Path: {path}, Total Distance: {dist}") # 예상: [0, 1, 3, 2, 0], 10+25+30+15 = 80