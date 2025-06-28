import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수 로드

class GeminiAPI:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=self.api_key)
        # 모델 설정: 'gemini-pro' 또는 다른 적합한 모델 선택
        self.model = genai.GenerativeModel('gemini-pro') 

    def get_travel_info(self, place_name, country=None, query_type="general_info"):
        """
        Gemini API를 사용하여 특정 장소에 대한 여행 정보를 가져옵니다.
        
        :param place_name: 장소 이름 (예: "Eiffel Tower", "Namsan Tower")
        :param country: 국가 (정보의 정확도를 높이기 위함)
        :param query_type: 요청 유형 (예: "general_info", "activities", "cultural_insights", "best_time_to_visit")
        :return: Gemini API 응답 텍스트
        """
        prompt_map = {
            "general_info": f"Provide general travel information about {place_name} in {country if country else 'the world'}.",
            "activities": f"What are the top 3 activities to do at {place_name} in {country if country else 'the world'}?",
            "cultural_insights": f"Share 3 interesting cultural insights or historical facts about {place_name} in {country if country else 'the world'}.",
            "best_time_to_visit": f"What is the best time to visit {place_name} in {country if country else 'the world'} and why?",
            "route_summary": f"Given a travel route visiting these places: {place_name}. Provide a brief, engaging summary of what a traveler can expect.",
            # 필요에 따라 더 많은 쿼리 유형 추가
        }

        prompt = prompt_map.get(query_type, f"Tell me about {place_name} in {country if country else 'the world'}.")

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API 호출 중 오류 발생: {e}")
            return f"정보를 가져올 수 없습니다. ({e})"

    def get_route_summary_and_tips(self, places_in_order, total_cost_info, trip_type="sightseeing"):
        """
        최적화된 경로에 대한 종합적인 요약 및 팁을 Gemini API를 통해 생성합니다.
        
        :param places_in_order: 경로에 포함된 장소 이름 리스트 (순서대로)
        :param total_cost_info: 총 비용 정보 (예: "500 km, 10 hours")
        :param trip_type: 여행 유형 (예: "sightseeing", "adventure", "food_tour")
        :return: Gemini API 응답 텍스트
        """
        place_list_str = " -> ".join(places_in_order)
        prompt = f"""
        You are a helpful travel assistant.
        A user has planned a {trip_type} trip with the following optimized route:
        {place_list_str}

        The total estimated travel cost/duration is {total_cost_info}.

        Please provide:
        1. An engaging summary of this travel route, highlighting what makes it interesting.
        2. 3-5 practical tips for this specific route or the general travel in these regions (e.g., local transport, best time to visit for this route, unique experiences).
        3. A suggested local delicacy or experience to try at one of the key stops.

        Keep the response concise and encouraging, around 200-300 words.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API 호출 중 오류 발생: {e}")
            return "경로 요약을 생성할 수 없습니다."

# 예시 사용
if __name__ == "__main__":
    gemini_api = GeminiAPI()
    
    # .env 파일에 GEMINI_API_KEY 설정 필요
    # Maps_API_KEY도 필요 (Maps_api.py 때문)
    
    # 장소 정보 요청 예시
    info = gemini_api.get_travel_info("Eiffel Tower", "France", "general_info")
    print(f"\n에펠탑 일반 정보:\n{info}\n")

    activities = gemini_api.get_travel_info("Gyeongbokgung Palace", "South Korea", "activities")
    print(f"\n경복궁 추천 활동:\n{activities}\n")

    # 경로 요약 요청 예시
    route_places = ["Seoul", "Busan", "Jeju Island"]
    total_cost_duration = "1200 km, 20 hours"
    route_summary = gemini_api.get_route_summary_and_tips(route_places, total_cost_duration, "sightseeing")
    print(f"\n경로 요약 및 팁:\n{route_summary}\n")