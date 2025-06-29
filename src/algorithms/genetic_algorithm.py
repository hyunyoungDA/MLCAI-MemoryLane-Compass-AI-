import random
from typing import List, Tuple

class GeneticAlgorithmSolver:
    def __init__(self, population_size: int = 100, generations: int = 500, mutation_rate: float = 0.02, crossover_rate: float = 0.8):
        # 개체 수, 돌연변이율 등 파라미터 초기화 
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _calculate_fitness(self, path: List[int], distance_matrix: List[List[float]]) -> float:
        """
        경로의 적합도(fitness)를 계산합니다. 비용이 낮을수록 적합도가 높습니다.
        TSP 문제이므로 마지막 지점에서 시작 지점으로 돌아오는 비용을 포함합니다.
        """
        cost = 0.0
        num_points = len(path)
        
        if num_points < 2:
            return 0.0 # 유효하지 않거나 단일 지점 경로, 적합도 0 (선택되지 않도록)

        for i in range(num_points):
            start_node = path[i]
            end_node = path[(i + 1) % num_points] # 마지막에서 시작으로 돌아오기 위해 모듈러 연산

            dist = distance_matrix[start_node][end_node]
            if dist == float('inf'):
                return 0.0 # 연결되지 않은 경로, 적합도 0 (선택되지 않도록)
            cost += dist
        
        # 적합도는 비용의 역수로 정의 (높을수록 좋음). 
        # 비용이 0이거나 무한대인 경우 적합도 0으로 처리하여 선택되지 않도록 합니다.
        if cost == 0.0 or cost == float('inf'):
            return 0.0
        
        return 1.0 / cost

    def _create_initial_population(self, num_points: int) -> List[List[int]]:
        """초기 개체군을 생성합니다."""
        population = []
        # 개체군 크기만큼 순회 
        for _ in range(self.population_size):
            individual = list(range(num_points))
            random.shuffle(individual)
            population.append(individual)
        return population

    def _select_parents(self, population: List[List[int]], fitnesses: List[float]) -> Tuple[List[int], List[int]]:
        """룰렛 휠 선택 방식으로 부모를 선택합니다."""
        total_fitness = sum(fitnesses)
        
        # 모든 적합도가 0인 경우 (모든 경로가 유효하지 않거나 연결되지 않은 경우 등)
        if total_fitness == 0: 
            # 무작위로 선택하거나, 예외를 발생시키거나, 빈 리스트 반환 고려
            # 여기서는 무작위로 선택하여 탐색을 이어가도록 함 (덜 효율적일 수 있음)
            if len(population) < 2:
                raise ValueError("Not enough individuals in population to select parents.")
            return random.sample(population, 2)
            
        selection_probs = [f / total_fitness for f in fitnesses]
        
        # 두 개의 부모를 선택 (복원 추출)
        parent1 = random.choices(population, weights=selection_probs, k=1)[0]
        parent2 = random.choices(population, weights=selection_probs, k=1)[0]
        return list(parent1), list(parent2) # 리스트 복사하여 반환

    def _crossover_pmx(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """PMX(Partially Mapped Crossover)를 사용하여 자식을 생성합니다."""
        num_points = len(parent1)
        
        child1 = [None] * num_points
        child2 = [None] * num_points

        # 교차 지점 선택
        # random.sample은 중복 없는 값을 반환하므로 len이 1 이하일 경우 에러 발생 방지
        if num_points < 2:
             return list(parent1), list(parent2) # 교차할 수 없는 경우 부모 복사

        start_idx, end_idx = sorted(random.sample(range(num_points), 2))

        # 부분 복사
        child1[start_idx:end_idx+1] = parent1[start_idx:end_idx+1]
        child2[start_idx:end_idx+1] = parent2[start_idx:end_idx+1]

        # 나머지 채우기
        for i in range(num_points):
            if child1[i] is None:
                current_val = parent2[i]
                while current_val in child1[start_idx:end_idx+1]: # 이미 복사된 구간에 있는지 확인
                    # 매핑된 값 찾기
                    try:
                        idx_in_parent1 = parent1.index(current_val)
                        current_val = parent2[idx_in_parent1]
                    except ValueError: # current_val이 parent1에 없는 경우 (이상 케이스)
                        # 이 경우는 발생하지 않아야 함, PMX 로직에 문제가 있다면 발생 가능
                        print(f"Warning: Value {current_val} not found in parent1 during PMX crossover.")
                        break # 무한 루프 방지
                child1[i] = current_val

            if child2[i] is None:
                current_val = parent1[i]
                while current_val in child2[start_idx:end_idx+1]: # 이미 복사된 구간에 있는지 확인
                    try:
                        idx_in_parent2 = parent2.index(current_val)
                        current_val = parent1[idx_in_parent2]
                    except ValueError:
                        print(f"Warning: Value {current_val} not found in parent2 during PMX crossover.")
                        break
                child2[i] = current_val

        return child1, child2


    def _mutate(self, individual: List[int]) -> List[int]:
        """돌연변이를 적용합니다 (SWAP)."""
        if random.random() < self.mutation_rate:
            if len(individual) < 2: # 길이가 1 이하인 경우 돌연변이 불가
                return individual
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float, List[float]]:
        """
        Genetic Algorithm을 사용하여 TSP 문제를 해결합니다.
        
        Args:
            distance_matrix (List[List[float]]): N x N 형태의 거리 행렬 (인덱스 기준).
        Returns:
            Tuple[List[int], float, List[float]]: 
            (최적 경로 인덱스, 총 최적 비용, 세대별 최적 비용 이력) 튜플.
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0.0, []
        if num_points == 1:
            return [0], 0.0, [0.0]

        population = self._create_initial_population(num_points)
        best_overall_path = None
        best_overall_cost = float('inf') # 최적 비용을 무한대로 초기화
        cost_history: List[float] = [] # 세대별 최적 비용을 저장할 리스트

        for generation in range(self.generations):
            fitnesses = [self._calculate_fitness(individual, distance_matrix) for individual in population]
            
            # 현재 세대의 최적 해 업데이트
            # 적합도가 0인 경우 (유효하지 않은 경로)는 제외하고 최대 적합도 찾기
            valid_fitnesses = [(f, i) for i, f in enumerate(fitnesses) if f > 0]
            
            current_best_path = None
            current_best_cost = float('inf')

            if valid_fitnesses:
                current_best_fitness, current_best_idx = max(valid_fitnesses, key=lambda x: x[0])
                current_best_path = population[current_best_idx]
                current_best_cost = 1.0 / current_best_fitness if current_best_fitness > 0 else float('inf')

            # 전역 최적 해 업데이트
            if current_best_path and current_best_cost < best_overall_cost:
                best_overall_cost = current_best_cost
                best_overall_path = list(current_best_path)
            
            # 비용 이력에 현재 세대의 최적 비용 추가
            # 만약 유효한 경로가 없으면 float('inf')를 추가하거나, 이전 best_overall_cost를 유지
            cost_history.append(best_overall_cost) 

            new_population = []
            # 엘리트주의 (가장 좋은 해는 다음 세대로 무조건 전달)
            if best_overall_path and best_overall_cost != float('inf'):
                new_population.append(list(best_overall_path))
            
            # 나머지 개체군 채우기
            # selection_probs 계산을 위해 항상 fitnesses를 사용
            # 모든 fitness가 0인 경우 random.sample로 처리
            
            # `_select_parents`가 `total_fitness == 0`을 처리하므로, 여기서는 그냥 호출
            while len(new_population) < self.population_size:
                try:
                    parent1, parent2 = self._select_parents(population, fitnesses)
                except ValueError as e:
                    # 인구수가 부족하거나 선택할 부모가 없는 경우
                    print(f"Warning: Parent selection failed in GA (Generation {generation}): {e}. Breaking loop.")
                    break # 이 세대에서의 개체군 생성을 중단

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_pmx(parent1, parent2)
                else:
                    child1, child2 = list(parent1), list(parent2) # 교차 없이 부모 복사

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population

        # 최종적으로 찾은 최적 경로의 비용 다시 계산 (불필요한 경우도 있지만, 안전을 위해)
        # `best_overall_cost`가 이미 최종 비용이므로 다시 계산할 필요는 없습니다.
        # 다만 `best_overall_path`가 유효한지 최종 검사
        if best_overall_path:
            # TSP 문제에서 경로가 시작점으로 돌아오는 형태가 필요하다면
            # `_calculate_fitness`에서 이미 이 부분이 반영되었으므로,
            # `best_overall_path`는 시작점으로 돌아오는 노드를 포함하지 않는 순열 형태입니다.
            # `route_optimizer.py`에서 마지막에 시작 지점을 추가하도록 합니다.
            return best_overall_path, best_overall_cost, cost_history
        
        return [], float('inf'), cost_history # 최적 경로를 찾지 못한 경우

# 예시 사용
if __name__ == "__main__":
    # TSP 예시 (4개 지점)
    # 0: 서울, 1: 부산, 2: 제주, 3: 대전
    test_matrix = [
        [0, 300, 500, 150],  # 0->x
        [300, 0, 400, 200],  # 1->x
        [500, 400, 0, 300],  # 2->x
        [150, 200, 300, 0]   # 3->x
    ]
    # 최적 경로: 0 -> 3 -> 1 -> 2 -> 0 (150 + 200 + 400 + 500 = 1250)
    # 또는 0 -> 1 -> 3 -> 2 -> 0 (300 + 200 + 300 + 500 = 1300)

    print("--- Genetic Algorithm 테스트 ---")
    solver = GeneticAlgorithmSolver(
        population_size=100, 
        generations=500, 
        mutation_rate=0.02, 
        crossover_rate=0.8
    )
    path, dist, history = solver.solve(test_matrix)
    print(f"Genetic Algorithm Path: {path}, Total Distance: {dist}")
    print(f"Cost History (last 10): {history[-10:]}")
    
    # 연결되지 않은 경로 테스트
    print("\n--- 연결되지 않은 경로 테스트 (inf 값 포함) ---")
    disconnected_matrix = [
        [0, 10, float('inf')],
        [10, 0, 5],
        [float('inf'), 5, 0]
    ]
    path_disc, dist_disc, history_disc = solver.solve(disconnected_matrix)
    print(f"Disconnected Path: {path_disc}, Total Distance: {dist_disc}")
    print(f"Cost History: {history_disc[-10:] if history_disc else 'N/A'}")

