# web_ui/streamlit_app.py

import streamlit as st
import requests
import json
import folium
from streamlit_folium import folium_static
import os

# FastAPI 백엔드 URL 설정 (실제 배포 시 변경)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Global AI Travel Planner")

st.title("✈️ 글로벌 AI 기반 여행 최적 경로 플래너")
st.markdown("여러분의 여행지를 입력하면 최적의 경로와 AI 요약을 제공합니다!")

# 사이드바: API 키 정보 (FastAPI 서버에 키가 설정되어 있으므로, 여기서는 단순히 설명)
st.sidebar.header("설정")
st.sidebar.info("이 앱은 백엔드에서 Google Maps API 및 Gemini API를 사용합니다. API 키는 서버에 설정되어 있습니다.")

# --- 웨이포인트 입력 ---
st.header("1. 여행 지점 입력")
num_waypoints = st.session_state.get('num_waypoints', 2)

# 웨이포인트 추가/삭제 버튼
col1, col2 = st.columns([1, 10])
with col1:
    if st.button("지점 추가"):
        num_waypoints += 1
        st.session_state.num_waypoints = num_waypoints
with col2:
    if st.button("지점 제거") and num_waypoints > 2:
        num_waypoints -= 1
        st.session_state.num_waypoints = num_waypoints

waypoints_input = []
for i in range(num_waypoints):
    st.subheader(f"지점 {i+1}")
    with st.expander(f"지점 {i+1} 세부 정보"):
        name = st.text_input(f"이름 (예: Eiffel Tower)", key=f"name_{i}")
        lat = st.number_input(f"위도 (예: 48.8584)", format="%.6f", key=f"lat_{i}")
        lng = st.number_input(f"경도 (예: 2.2945)", format="%.6f", key=f"lng_{i}")
        country = st.text_input(f"국가 (선택 사항, 예: France)", key=f"country_{i}")
    waypoints_input.append({"name": name, "lat": lat, "lng": lng, "country": country if country else None})

# --- 최적화 설정 ---
st.header("2. 최적화 설정")
col_algo, col_metric = st.columns(2)
with col_algo:
    algorithm = st.selectbox(
        "최적화 알고리즘",
        ("greedy", "simulated_annealing", "genetic_algorithm"),
        help="경로를 찾을 알고리즘을 선택하세요."
    )
with col_metric:
    optimization_metric = st.selectbox(
        "최적화 기준",
        ("duration", "distance"),
        help="최단 '시간' 또는 최단 '거리' 중 어떤 것을 기준으로 최적화할까요?"
    )

# 알고리즘별 추가 파라미터 (옵션)
st.subheader("알고리즘 파라미터 (선택 사항)")
if algorithm == "simulated_annealing":
    sa_initial_temperature = st.number_input("초기 온도", value=10000.0, help="탐색 시작 시 온도")
    sa_cooling_rate = st.number_input("냉각 속도", value=0.99, help="매 반복마다 온도가 줄어드는 비율")
    sa_max_iterations = st.number_input("최대 반복 횟수", value=50000, step=1000)
    st.session_state.sa_params = {"initial_temperature": sa_initial_temperature, "cooling_rate": sa_cooling_rate, "max_iterations": sa_max_iterations}
elif algorithm == "genetic_algorithm":
    ga_population_size = st.number_input("개체군 크기", value=100, step=10)
    ga_generations = st.number_input("세대 수", value=1000, step=100)
    ga_mutation_rate = st.number_input("돌연변이율", value=0.05, format="%.2f")
    ga_crossover_rate = st.number_input("교차율", value=0.80, format="%.2f")
    st.session_state.ga_params = {"population_size": ga_population_size, "generations": ga_generations, "mutation_rate": ga_mutation_rate, "crossover_rate": ga_crossover_rate}

# --- 경로 최적화 및 결과 표시 ---
st.header("3. 경로 최적화 결과")
if st.button("경로 최적화 시작"):
    valid_waypoints = [wp for wp in waypoints_input if wp['name'] and wp['lat'] and wp['lng']]
    if len(valid_waypoints) < 2:
        st.error("최소 2개 이상의 유효한 지점을 입력해야 합니다.")
    else:
        request_data = {
            "waypoints": valid_waypoints,
            "algorithm": algorithm,
            "optimization_metric": optimization_metric
        }
        if algorithm == "simulated_annealing":
            request_data.update({
                "sa_initial_temperature": st.session_state.sa_params["initial_temperature"],
                "sa_cooling_rate": st.session_state.sa_params["cooling_rate"],
                "sa_max_iterations": st.session_state.sa_params["max_iterations"]
            })
        elif algorithm == "genetic_algorithm":
            request_data.update({
                "ga_population_size": st.session_state.ga_params["population_size"],
                "ga_generations": st.session_state.ga_params["generations"],
                "ga_mutation_rate": st.session_state.ga_params["mutation_rate"],
                "ga_crossover_rate": st.session_state.ga_params["crossover_rate"]
            })
        
        with st.spinner("경로 최적화 및 AI 요약 생성 중..."):
            try:
                response = requests.post(f"{FASTAPI_URL}/optimize-route", json=request_data)
                response.raise_for_status() # HTTP 에러 발생 시 예외 처리
                result = response.json()

                st.subheader("최적화된 경로")
                st.write(f"**알고리즘**: {result['algorithm']}")
                st.write(f"**총 비용**: {result['total_cost']:.2f} {result['cost_unit']} ({result['cost_unit']} 기준)")
                st.write(f"**실행 시간**: {result['execution_time_seconds']:.4f} 초")
                
                optimized_path_names = " -> ".join(result['optimized_path_names'])
                st.write(f"**최적 경로**: {optimized_path_names}")

                st.subheader("AI 여행 요약 및 팁 (Powered by Gemini)")
                st.write(result.get('gemini_summary', 'AI 요약을 가져오지 못했습니다.'))

                # 지도 시각화 (FastAPI 서버에서 생성한 HTML을 임베드)
                # 직접 스트림릿에서 Folium으로 그리는 것이 더 편리할 수 있음.
                # 여기서는 FastAPI에서 생성한 HTML 경로를 스트림릿이 직접 읽도록 (개발 편의상)
                # 실제 배포 환경에서는 S3 같은 스토리지에 저장하고 URL로 접근.
                
                # FastAPI가 저장한 HTML 파일명을 알고 있다고 가정
                # 예: algorithm_cost.html -> Streamlit에서 해당 파일을 직접 읽어 표시
                # 이 부분은 실제 파일 시스템 접근 권한에 따라 달라질 수 있으므로,
                # Streamlit 내에서 Folium을 직접 생성하는 것이 더 안정적입니다.
                
                st.subheader("경로 지도")
                # 스트림릿에서 직접 folium으로 지도 그리기
                if 'optimized_path_indices' in result and result['optimized_path_indices']:
                    # 지도 중심점 계산
                    path_coords = []
                    for idx in result['optimized_path_indices']:
                        if idx < len(valid_waypoints):
                            path_coords.append([valid_waypoints[idx]['lat'], valid_waypoints[idx]['lng']])
                        else:
                            st.warning(f"유효하지 않은 지점 인덱스 {idx}가 경로에 포함되었습니다.")
                            path_coords = [] # 경로 초기화
                            break

                    if path_coords and len(path_coords) > 1:
                        avg_lat = sum(wp['lat'] for wp in valid_waypoints) / len(valid_waypoints)
                        avg_lng = sum(wp['lng'] for wp in valid_waypoints) / len(valid_waypoints)
                        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=4) # 세계지도 스케일에 맞춰 zoom 조정

                        # 마커 추가
                        for i, wp in enumerate(valid_waypoints):
                            icon_color = 'blue' if i == result['optimized_path_indices'][0] else 'green'
                            folium.Marker(
                                location=[wp['lat'], wp['lng']],
                                popup=f"<b>{wp.get('name', f'Point {i}')}</b><br>Idx: {i}",
                                icon=folium.Icon(color=icon_color, icon='info-sign')
                            ).add_to(m)
                        
                        # 경로 그리기
                        folium.PolyLine(path_coords, color="red", weight=2.5, opacity=1).add_to(m)

                        folium_static(m, width=900, height=600)
                    else:
                        st.warning("경로 시각화를 위한 유효한 좌표 데이터가 부족합니다.")
                else:
                    st.warning("최적화된 경로가 유효하지 않아 지도를 표시할 수 없습니다.")

            except requests.exceptions.ConnectionError:
                st.error(f"FastAPI 서버에 연결할 수 없습니다. 서버가 {FASTAPI_URL}에서 실행 중인지 확인하세요.")
            except requests.exceptions.RequestException as e:
                st.error(f"API 요청 중 오류가 발생했습니다: {e}")
                if response:
                    st.error(f"서버 응답: {response.text}")
            except Exception as e:
                st.error(f"예상치 못한 오류가 발생했습니다: {e}")

# --- 개별 장소 정보 얻기 ---
st.header("4. 개별 여행 장소 정보 (Gemini API)")
gemini_place_name = st.text_input("정보를 얻고 싶은 장소 이름", "Namsan Tower", key="gem_place_name")
gemini_place_country = st.text_input("장소 국가 (선택 사항)", "South Korea", key="gem_place_country")
gemini_query_type = st.selectbox(
    "정보 유형",
    ("general_info", "activities", "cultural_insights", "best_time_to_visit"),
    key="gem_query_type"
)

if st.button("정보 가져오기"):
    if gemini_place_name:
        gemini_request_data = {
            "place_name": gemini_place_name,
            "country": gemini_place_country if gemini_place_country else None,
            "query_type": gemini_query_type
        }
        with st.spinner(f"{gemini_place_name}에 대한 정보 가져오는 중..."):
            try:
                response = requests.post(f"{FASTAPI_URL}/get-gemini-info", json=gemini_request_data)
                response.raise_for_status()
                gemini_result = response.json()
                st.subheader(f"✨ {gemini_result['place_name']} 정보")
                st.write(gemini_result['info'])
            except requests.exceptions.ConnectionError:
                st.error(f"FastAPI 서버에 연결할 수 없습니다. 서버가 {FASTAPI_URL}에서 실행 중인지 확인하세요.")
            except requests.exceptions.RequestException as e:
                st.error(f"API 요청 중 오류가 발생했습니다: {e}")
                if response:
                    st.error(f"서버 응답: {response.text}")
            except Exception as e:
                st.error(f"예상치 못한 오류가 발생했습니다: {e}")
    else:
        st.warning("장소 이름을 입력해주세요.")