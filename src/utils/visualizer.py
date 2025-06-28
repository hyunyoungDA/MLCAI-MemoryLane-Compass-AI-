import folium
import matplotlib.pyplot as plt
import os

class MapVisualizer:
    def __init__(self, output_dir="results/maps"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_path_on_map(self, waypoints, path, algorithm_name, total_cost, output_filename=None):
        """
        주어진 경로를 지도 위에 시각화하고 HTML 파일로 저장합니다.
        
        :param waypoints: [{"name": str, "lat": float, "lng": float}, ...]
        :param path: 알고리즘이 반환한 지점 인덱스 순서 (예: [0, 2, 1, 0])
        :param algorithm_name: 알고리즘 이름 (예: "Greedy", "Simulated Annealing")
        :param total_cost: 총 경로 비용
        :param output_filename: 저장할 파일 이름 (없으면 자동 생성)
        """
        if not waypoints or not path or len(path) < 2:
            print("Warning: 유효한 경로를 시각화할 수 없습니다. (waypoints 또는 path 부족)")
            return

        # 경로 중심점 계산
        avg_lat = sum(wp['lat'] for wp in waypoints) / len(waypoints)
        avg_lng = sum(wp['lng'] for wp in waypoints) / len(waypoints)

        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=10)

        # 지점 마커 추가
        for i, wp in enumerate(waypoints):
            folium.Marker(
                location=[wp['lat'], wp['lng']],
                popup=f"<b>{wp.get('name', f'Point {i}')}</b><br>Idx: {i}",
                icon=folium.Icon(color='blue' if i == path[0] else 'green', icon='info-sign')
            ).add_to(m)

        # 경로 그리기
        route_coords = []
        for i in range(len(path)):
            idx = path[i]
            if idx < len(waypoints): # 유효한 인덱스인지 확인
                route_coords.append([waypoints[idx]['lat'], waypoints[idx]['lng']])
            else:
                print(f"Warning: 경로에 유효하지 않은 지점 인덱스({idx})가 포함되어 있습니다.")
                # 유효하지 않은 인덱스 발생 시 경로 중단
                route_coords = [] 
                break

        if route_coords:
            folium.PolyLine(route_coords, color="red", weight=2.5, opacity=1).add_to(m)

        # 제목 추가
        title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{algorithm_name} Path (Cost: {total_cost:,.2f})</b></h3>
             '''
        m.get_root().html.add_child(folium.Element(title_html))

        if output_filename:
            filepath = os.path.join(self.output_dir, output_filename)
        else:
            filepath = os.path.join(self.output_dir, f"{algorithm_name.replace(' ', '_').lower()}_path_{int(total_cost)}.html")
        
        m.save(filepath)
        print(f"Map saved to {filepath}")


class PlottingVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_costs_comparison(self, algorithm_costs, output_filename="algorithm_costs_comparison.png"):
        """
        여러 알고리즘의 총 비용을 막대 그래프로 비교합니다.
        
        :param algorithm_costs: {알고리즘 이름: 총 비용, ...} 딕셔너리
        :param output_filename: 저장할 파일 이름
        """
        names = list(algorithm_costs.keys())
        costs = list(algorithm_costs.values())
        
        # 무한대 값 처리
        finite_costs = [c if c != float('inf') else 0 for c in costs]
        # inf 값은 그래프에서 제외하거나 특별히 표시
        labels = [name if cost != float('inf') else f"{name} (No Valid Path)" for name, cost in zip(names, costs)]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, finite_costs, color=['skyblue', 'lightcoral', 'lightgreen', 'purple'])
        
        plt.xlabel("Algorithm")
        plt.ylabel("Total Cost (meters or seconds)")
        plt.title("Comparison of Total Path Costs by Algorithm")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 각 막대 위에 값 표시
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max(finite_costs) * 0.02), round(yval, 2), ha='center', va='bottom')

        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath)
        print(f"Cost comparison plot saved to {filepath}")
        plt.close() # 플롯 창 닫기

    def plot_algorithm_progress(self, algorithm_name, costs_history, output_filename=None):
        """
        알고리즘(SA, GA 등)의 반복에 따른 비용 변화를 그래프로 시각화합니다.
        
        :param algorithm_name: 알고리즘 이름
        :param costs_history: 반복 횟수에 따른 비용 리스트
        :param output_filename: 저장할 파일 이름
        """
        if not costs_history:
            print(f"No cost history to plot for {algorithm_name}.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(costs_history, label='Best Cost per Iteration', color='blue', alpha=0.8)
        plt.xlabel("Iteration / Generation")
        plt.ylabel("Cost")
        plt.title(f"{algorithm_name} - Cost Convergence Over Iterations")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        if output_filename:
            filepath = os.path.join(self.output_dir, output_filename)
        else:
            filepath = os.path.join(self.output_dir, f"{algorithm_name.replace(' ', '_').lower()}_convergence.png")
        
        plt.savefig(filepath)
        print(f"Convergence plot saved to {filepath}")
        plt.close()

# 예시 사용
if __name__ == "__main__":
    # 테스트용 지점 데이터
    test_waypoints_for_map = [
        {"name": "Seoul", "lat": 37.5665, "lng": 126.9780},
        {"name": "Busan", "lat": 35.1796, "lng": 129.0756},
        {"name": "Jeju", "lat": 33.5097, "lng": 126.5113},
        {"name": "Daejeon", "lat": 36.3504, "lng": 127.3845}
    ]
    
    # 테스트용 경로 (인덱스 순서) 및 비용
    greedy_path = [0, 3, 1, 2, 0] # 서울 -> 대전 -> 부산 -> 제주 -> 서울
    greedy_cost = 1200000 # 가상의 비용 (미터)

    sa_path = [0, 1, 3, 2, 0] # 서울 -> 부산 -> 대전 -> 제주 -> 서울
    sa_cost = 1150000

    ga_path = [0, 2, 3, 1, 0] # 서울 -> 제주 -> 대전 -> 부산 -> 서울
    ga_cost = 1100000

    map_viz = MapVisualizer()
    map_viz.plot_path_on_map(test_waypoints_for_map, greedy_path, "Greedy", greedy_cost)
    map_viz.plot_path_on_map(test_waypoints_for_map, sa_path, "Simulated Annealing", sa_cost)
    map_viz.plot_path_on_map(test_waypoints_for_map, ga_path, "Genetic Algorithm", ga_cost)

    plot_viz = PlottingVisualizer()
    algo_costs = {
        "Greedy": greedy_cost,
        "Simulated Annealing": sa_cost,
        "Genetic Algorithm": ga_cost
    }
    plot_viz.plot_costs_comparison(algo_costs)

    # SA/GA 수렴 그래프 예시
    sa_history = [1500000, 1400000, 1300000, 1250000, 1200000, 1180000, 1160000, 1150000]
    ga_history = [1600000, 1500000, 1400000, 1300000, 1200000, 1150000, 1120000, 1100000]
    
    plot_viz.plot_algorithm_progress("Simulated Annealing", sa_history)
    plot_viz.plot_algorithm_progress("Genetic Algorithm", ga_history)