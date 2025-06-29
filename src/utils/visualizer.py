import folium
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Optional, Tuple

class MapVisualizer:
    def __init__(self, output_dir: str = "results/maps"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_path_on_map(
        self, 
        waypoints: List[Dict[str, Any]], 
        path_indices: List[int], 
        algorithm_name: str, 
        total_cost: float, 
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        주어진 최적화된 경로를 지도 위에 시각화하고 HTML 파일로 저장합니다.
        
        Args:
            waypoints (List[Dict[str, Any]]): [{"name": str, "lat": float, "lng": float}, ...]
            path_indices (List[int]): 알고리즘이 반환한 지점 인덱스 순서 (예: [0, 2, 1, 0])
            algorithm_name (str): 알고리즘 이름 (예: "Greedy", "Simulated Annealing")
            total_cost (float): 총 경로 비용
            output_filename (Optional[str]): 저장할 파일 이름 (없으면 자동 생성)
        Returns:
            Optional[str]: 저장된 지도 파일의 전체 경로 (URL이 아님), 실패 시 None.
        """
        if not waypoints or not path_indices or len(path_indices) < 2:
            print("Warning: 유효한 경로를 시각화할 수 없습니다. (waypoints 또는 path_indices 부족)")
            return None

        # 경로 중심점 계산 (지도를 가운데로)
        avg_lat = sum(wp['lat'] for wp in waypoints) / len(waypoints)
        avg_lng = sum(wp['lng'] for wp in waypoints) / len(waypoints)

        # Folium 지도 초기화 (더 예쁜 타일셋 사용)
        # tiles="OpenStreetMap"
        # tiles="CartoDB positron"
        # tiles="Stamen Terrain"
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=8, tiles="CartoDB positron")

        # 지점 마커 및 숫자 아이콘 추가
        # 시작 지점: 파란색 마커, 중간 지점: 초록색 마커, 숫자 순서 표시
        for i, wp in enumerate(waypoints):
            # 경로 순서에 따른 인덱스 번호 (0부터 시작)
            # path_indices 내에서 해당 waypoint의 인덱스 번호를 찾아서 표시
            display_idx = path_indices.index(i) if i in path_indices else -1
            
            icon_color = 'blue' if i == path_indices[0] else 'green' # 시작점은 파란색
            
            # 숫자 마커 (DivIcon 사용)
            folium.Marker(
                location=[wp['lat'], wp['lng']],
                popup=f"<b>{wp.get('name', f'Point {i}')}</b><br>경유지 ID: {i}<br>경로 순서: {display_idx+1 if display_idx != -1 else 'N/A'}",
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html=f"""
                    <div style="
                        font-size: 10pt;
                        font-weight: bold;
                        color: white;
                        background-color: {icon_color};
                        border-radius: 50%;
                        width: 20px;
                        height: 20px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        border: 1px solid white;
                    ">{i}</div>
                    """
                )
            ).add_to(m)

        # 경로 그리기 (PolyLine)
        route_coords = []
        for idx in path_indices:
            if idx < len(waypoints): # 유효한 인덱스인지 확인
                route_coords.append([waypoints[idx]['lat'], waypoints[idx]['lng']])
            else:
                print(f"Warning: 경로에 유효하지 않은 지점 인덱스({idx})가 포함되어 있습니다. 경로 시각화가 불완전할 수 있습니다.")
                route_coords = [] # 유효하지 않은 인덱스 발생 시 경로 중단
                break

        if route_coords:
            folium.PolyLine(route_coords, color="red", weight=3, opacity=0.8).add_to(m)
            
            # 경로 방향 화살표 추가 (선택 사항, 복잡성 증가)
            # Folium에 직접적인 화살표 기능은 없으므로, CircleMarker 등을 활용하여 직접 구현하거나
            # Arrow plugin을 사용해야 하는데, 이는 Folium 버전이나 환경에 따라 다를 수 있습니다.
            # 여기서는 복잡성을 줄이기 위해 생략합니다.

        # 제목 추가 (총 비용 포맷 개선)
        title_html = f'''
             <h3 align="center" style="font-size:16px; margin-top:10px;"><b>{algorithm_name} Path (Total Cost: {total_cost:,.2f})</b></h3>
             '''
        m.get_root().html.add_child(folium.Element(title_html))

        # 파일 저장
        if output_filename:
            filepath = os.path.join(self.output_dir, output_filename)
        else:
            # 파일명에 알고리즘 이름과 비용을 포함하여 고유하게 만듦
            clean_algo_name = algorithm_name.replace(' ', '_').lower()
            filepath = os.path.join(self.output_dir, f"{clean_algo_name}_path_{int(total_cost)}.html")
        
        m.save(filepath)
        print(f"Map saved to {filepath}")
        return filepath # 저장된 파일 경로 반환


class PlottingVisualizer:
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_costs_comparison(self, algorithm_costs: Dict[str, float], output_filename: str = "algorithm_costs_comparison.png"):
        """
        여러 알고리즘의 총 비용을 막대 그래프로 비교합니다.
        
        Args:
            algorithm_costs (Dict[str, float]): {알고리즘 이름: 총 비용, ...} 딕셔너리.
            output_filename (str): 저장할 파일 이름.
        """
        names = []
        costs = []
        # float('inf') 값을 가진 알고리즘은 그래프에서 제외
        filtered_costs = {name: cost for name, cost in algorithm_costs.items() if cost != float('inf')}

        if not filtered_costs:
            print("No valid costs to plot for comparison.")
            return

        names = list(filtered_costs.keys())
        costs = list(filtered_costs.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, costs, color=['skyblue', 'lightcoral', 'lightgreen', 'purple', 'orange', 'grey'])
        
        plt.xlabel("Algorithm")
        plt.ylabel("Total Cost (meters or seconds)")
        plt.title("Comparison of Total Path Costs by Algorithm")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 각 막대 위에 값 표시
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:,.2f}", ha='center', va='bottom') # 콤마 포맷 추가

        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath)
        print(f"Cost comparison plot saved to {filepath}")
        plt.close() # 플롯 창 닫기

    def plot_algorithm_progress(self, algorithm_name: str, costs_history: List[float], output_filename: Optional[str] = None):
        """
        알고리즘(SA, GA 등)의 반복에 따른 비용 변화를 그래프로 시각화합니다.
        
        Args:
            algorithm_name (str): 알고리즘 이름.
            costs_history (List[float]): 반복 횟수에 따른 최적 비용 리스트.
            output_filename (Optional[str]): 저장할 파일 이름.
        """
        # cost_history에 float('inf')만 있거나 비어 있는 경우 처리
        if not costs_history or all(cost == float('inf') for cost in costs_history):
            print(f"No meaningful cost history to plot for {algorithm_name} (all infinite or empty).")
            return

        # inf 값을 가진 데이터 포인트는 matplotlib에서 자동으로 처리되지 않으므로, 유효한 값만 플로팅
        finite_costs = [c if c != float('inf') else None for c in costs_history]

        plt.figure(figsize=(10, 6))
        plt.plot(finite_costs, label='Best Cost per Iteration', color='blue', alpha=0.8)
        plt.xlabel("Iteration / Generation")
        plt.ylabel("Cost")
        plt.title(f"{algorithm_name} - Cost Convergence Over Iterations")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        if output_filename:
            filepath = os.path.join(self.output_dir, output_filename)
        else:
            clean_algo_name = algorithm_name.replace(' ', '_').lower()
            filepath = os.path.join(self.output_dir, f"{clean_algo_name}_convergence.png")
        
        plt.savefig(filepath)
        print(f"Convergence plot saved to {filepath}")
        plt.close()

# 예시 사용
if __name__ == "__main__":
    # 테스트용 지점 데이터 (Pydantic Waypoint 모델의 dict 형태)
    test_waypoints_for_map = [
        {"name": "서울시청", "lat": 37.5665, "lng": 126.9780},
        {"name": "부산역", "lat": 35.1796, "lng": 129.0756},
        {"name": "제주공항", "lat": 33.5097, "lng": 126.5113},
        {"name": "대전역", "lat": 36.3504, "lng": 127.3845}
    ]
    
    # 테스트용 경로 (인덱스 순서) 및 비용
    greedy_path_indices = [0, 3, 1, 2, 0] # 서울(0) -> 대전(3) -> 부산(1) -> 제주(2) -> 서울(0)
    greedy_cost = 1200000.0 # 가상의 비용 (미터)

    sa_path_indices = [0, 1, 3, 2, 0] # 서울(0) -> 부산(1) -> 대전(3) -> 제주(2) -> 서울(0)
    sa_cost = 1150000.0

    ga_path_indices = [0, 2, 3, 1, 0] # 서울(0) -> 제주(2) -> 대전(3) -> 부산(1) -> 서울(0)
    ga_cost = 1100000.0

    map_viz = MapVisualizer()
    map_viz.plot_path_on_map(test_waypoints_for_map, greedy_path_indices, "Greedy", greedy_cost)
    map_viz.plot_path_on_map(test_waypoints_for_map, sa_path_indices, "Simulated Annealing", sa_cost)
    map_viz.plot_path_on_map(test_waypoints_for_map, ga_path_indices, "Genetic Algorithm", ga_cost)

    plot_viz = PlottingVisualizer()
    algo_costs = {
        "Greedy": greedy_cost,
        "Simulated Annealing": sa_cost,
        "Genetic Algorithm": ga_cost,
        "Disconnected Algo": float('inf') # 테스트용 (유효하지 않은 경로)
    }
    plot_viz.plot_costs_comparison(algo_costs)

    # SA/GA 수렴 그래프 예시
    sa_history = [1500000.0, 1400000.0, 1300000.0, 1250000.0, 1200000.0, 1180000.0, 1160000.0, 1150000.0]
    ga_history = [1600000.0, 1500000.0, 1400000.0, 1300000.0, 1200000.0, 1150000.0, 1120000.0, 1100000.0]
    
    plot_viz.plot_algorithm_progress("Simulated Annealing", sa_history)
    plot_viz.plot_algorithm_progress("Genetic Algorithm", ga_history)
    
    # 히스토리가 비어있거나 모두 inf인 경우 테스트
    plot_viz.plot_algorithm_progress("Empty History", [])
    plot_viz.plot_algorithm_progress("All Infinite History", [float('inf'), float('inf'), float('inf')])
