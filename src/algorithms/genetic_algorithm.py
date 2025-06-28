import random
import sys

class GeneticAlgorithmSolver:
    def __init__(self, population_size=100, generations=500, mutation_rate=0.02, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _calculate_fitness(self, path, distance_matrix):
        """경로의 적합도(fitness)를 계산합니다. 비용이 낮을수록 적합도가 높습니다."""
        cost = 0
        num_points = len(path)
        if num_points < 2:
            return sys.float_info.max # 유효하지 않은 경로

        for i in range(num_points - 1):
            dist = distance_matrix[path[i]][path[i+1]]
            if dist == float('inf'):
                return sys.float_info.max # 연결되지 않은 경로
            cost += dist
        
        # TSP: 마지막 지점에서 시작 지점으로 돌아오는 비용 추가
        dist_to_start = distance_matrix[path[-1]][path[0]]
        if dist_to_start == float('inf'):
            return sys.float_info.max
        cost += dist_to_start
        
        # 적합도는 비용의 역수로 정의 (높을수록 좋음)
        # 비용이 0이 될 수 없으므로 1.0 / cost
        return 1.0 / cost if cost > 0 else sys.float_info.max

    def _create_initial_population(self, num_points):
        """초기 개체군을 생성합니다."""
        population = []
        for _ in range(self.population_size):
            individual = list(range(num_points))
            random.shuffle(individual)
            population.append(individual)
        return population

    def _select_parents(self, population, fitnesses):
        """룰렛 휠 선택 방식으로 부모를 선택합니다."""
        total_fitness = sum(fitnesses)
        if total_fitness == 0: # 모든 적합도가 0인 경우 (모든 경로가 유효하지 않은 경우 등)
            return random.sample(population, 2) # 무작위로 선택
            
        selection_probs = [f / total_fitness for f in fitnesses]
        
        # 두 개의 부모를 선택 (복원 추출)
        parent1 = random.choices(population, weights=selection_probs, k=1)[0]
        parent2 = random.choices(population, weights=selection_probs, k=1)[0]
        return list(parent1), list(parent2) # 리스트 복사하여 반환

    def _crossover_pmx(self, parent1, parent2):
        """PMX(Partially Mapped Crossover)를 사용하여 자식을 생성합니다."""
        num_points = len(parent1)
        
        child1 = [None] * num_points
        child2 = [None] * num_points

        # 교차 지점 선택
        start_idx, end_idx = sorted(random.sample(range(num_points), 2))

        # 부분 복사
        child1[start_idx:end_idx+1] = parent1[start_idx:end_idx+1]
        child2[start_idx:end_idx+1] = parent2[start_idx:end_idx+1]

        # 나머지 채우기
        for i in range(num_points):
            if child1[i] is None:
                current_val = parent2[i]
                while current_val in child1[start_idx:end_idx+1]:
                    # 매핑된 값 찾기
                    idx = parent1.index(current_val)
                    current_val = parent2[idx]
                child1[i] = current_val

            if child2[i] is None:
                current_val = parent1[i]
                while current_val in child2[start_idx:end_idx+1]:
                    idx = parent2.index(current_val)
                    current_val = parent1[idx]
                child2[i] = current_val

        return child1, child2


    def _mutate(self, individual):
        """돌연변이를 적용합니다 (SWAP)."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def solve(self, distance_matrix):
        """
        Genetic Algorithm을 사용하여 TSP 문제를 해결합니다.
        
        :param distance_matrix: N x N 형태의 거리 행렬 (인덱스 기준)
        :return: (최적 경로, 총 거리) 튜플
        """
        num_points = len(distance_matrix)
        if num_points == 0:
            return [], 0
        if num_points == 1:
            return [0], 0

        population = self._create_initial_population(num_points)
        best_overall_path = None
        best_overall_cost = sys.float_info.max

        for generation in range(self.generations):
            fitnesses = [self._calculate_fitness(individual, distance_matrix) for individual in population]
            
            # 현재 세대의 최적 해 업데이트
            current_best_idx = fitnesses.index(max(fitnesses))
            current_best_path = population[current_best_idx]
            current_best_cost = 1.0 / fitnesses[current_best_idx] if fitnesses[current_best_idx] > 0 else sys.float_info.max

            if current_best_cost < best_overall_cost:
                best_overall_cost = current_best_cost
                best_overall_path = list(current_best_path)

            new_population = []
            # 엘리트주의 (가장 좋은 해는 다음 세대로 무조건 전달)
            if best_overall_path and best_overall_cost != sys.float_info.max:
                new_population.append(list(best_overall_path))

            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitnesses)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_pmx(parent1, parent2)
                else:
                    child1, child2 = list(parent1), list(parent2) # 교차 없이 부모 복사

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size: # 짝수로 생성되므로 하나 더 추가
                    new_population.append(child2)
            
            population = new_population

        # 최종적으로 찾은 최적 경로의 비용 다시 계산
        if best_overall_path:
            final_cost = 1.0 / self._calculate_fitness(best_overall_path, distance_matrix)
            if final_cost == sys.float_info.max: # 유효하지 않은 경로
                 return [], float('inf')
            return best_overall_path, final_cost
        
        return [], float('inf') # 최적 경로를 찾지 못한 경우

# 예시 사용
if __name__ == "__main__":
    # 위 Greedy 예시와 동일한 거리 행렬
    test_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    solver = GeneticAlgorithmSolver(
        population_size=50, 
        generations=200, 
        mutation_rate=0.05, 
        crossover_rate=0.9
    )
    path, dist = solver.solve(test_matrix)
    print(f"Genetic Algorithm Path: {path}, Total Distance: {dist}")
    # GA는 무작위성이 있어 결과가 실행마다 다를 수 있습니다.
    # 여러 번 실행하면서 최적해에 가까운 결과를 찾아가는 과정을 볼 수 있습니다.