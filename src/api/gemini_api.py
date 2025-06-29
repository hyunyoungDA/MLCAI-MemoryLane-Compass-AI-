import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Optional
import asyncio

# Pydantic 스키마 임포트
from src.models.schemas import Waypoint, PlaceDetails # Waypoint와 PlaceDetails 스키마 임포트
load_dotenv() # .env 파일에서 환경 변수 로드

class GeminiAPI:
    def __init__(self):
        """
        Gemini API 클라이언트를 초기화합니다.
        GEMINI_API_KEY 환경 변수가 설정되어 있어야 합니다.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # configure 설정하여 모델 초기화
        genai.configure(api_key=self.api_key)
        # 모델 확인 필요 
        self.model = genai.GenerativeModel('gemini-2.0-flash') 

    async def _generate_content_with_retry(self, prompt: str, schema: Optional[Dict] = None) -> Optional[str]:
        """
        Gemini API 호출을 위한 헬퍼 함수 (재시도 및 에러 처리 포함).
        구조화된 응답을 위한 스키마를 전달할 수 있습니다.
        """
        for i in range(3): # 최대 3번 재시도
            try:
                # payload 구성
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                }
                if schema:
                    payload["generationConfig"] = {
                        "responseMimeType": "application/json",
                        "responseSchema": schema
                    }

                # API 키는 빈 문자열로 두면 Canvas 런타임에서 자동으로 제공됩니다.
                # 직접 API 키를 삽입하는 대신 이 방식을 사용합니다.
                api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

                # fetch 호출
                response = await asyncio.to_thread(
                    lambda: requests.post(
                        api_url,
                        headers={'Content-Type': 'application/json'},
                        json=payload
                    )
                )
                response.raise_for_status() # HTTP 오류가 발생하면 예외 발생

                result = response.json()
                
                # 결과 구조 확인 및 파싱
                if result.get("candidates") and result["candidates"][0].get("content") and \
                   result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print(f"Gemini API 응답 구조가 예상과 다릅니다: {result}")
                    return None
            except requests.exceptions.RequestException as req_err:
                print(f"Gemini API 호출 중 네트워크/HTTP 오류 발생 (재시도 {i+1}/3): {req_err}")
                await asyncio.sleep(2 ** i) # 지수 백오프
            except json.JSONDecodeError as json_err:
                print(f"Gemini API 응답 JSON 디코딩 오류 (재시도 {i+1}/3): {json_err}")
                await asyncio.sleep(2 ** i)
            except Exception as e:
                print(f"Gemini API 호출 중 알 수 없는 오류 발생 (재시도 {i+1}/3): {e}")
                await asyncio.sleep(2 ** i)
        print("Gemini API 호출 최대 재시도 횟수 초과.")
        return None

    async def get_route_summary_and_tips(
        self, 
        place_names: List[str], 
        total_cost_info: str, 
        optimization_metric: str
    ) -> Optional[str]:
        """
        최적화된 경로에 대한 종합적인 요약 및 팁을 Gemini API를 통해 생성합니다.
        
        Args:
            place_names (List[str]): 경로에 포함된 장소 이름 리스트 (순서대로).
            total_cost_info (str): 총 비용 정보 (예: "500 km, 10 hours").
            optimization_metric (str): 최적화 기준 (예: "distance", "duration").
        Returns:
            Optional[str]: Gemini API 응답 텍스트. 실패 시 None.
        """
        place_list_str = " -> ".join(place_names)
        
        prompt = f"""
        You are an enthusiastic and helpful AI travel assistant specializing in creating memorable journeys.
        A user has an optimized travel route visiting these places: {place_list_str}.
        The total estimated travel cost/duration for this route is {total_cost_info}, optimized for {optimization_metric}.

        Please provide:
        1.  An engaging and exciting summary of this travel route, highlighting its unique aspects and what makes it special. Focus on the journey's potential for new memories.
        2.  3-5 concise, practical, and inspiring tips for travelers embarking on this specific route or visiting these regions (e.g., local transport advice, unique experiences to seek out, cultural etiquette, photography spots, hidden gems).
        3.  Suggest one unique local delicacy or a must-try experience at one of the key stops along this route.

        Ensure the response is inviting, encouraging, and around 200-350 words.
        """
        
        return await self._generate_content_with_retry(prompt)

    async def get_place_details_for_route(self, waypoint: Waypoint) -> Optional[PlaceDetails]:
        """
        Gemini API를 사용하여 특정 장소에 대한 상세 정보를 PlaceDetails 스키마에 맞춰 생성합니다.
        
        Args:
            waypoint (Waypoint): 정보를 요청할 장소의 Waypoint 객체.
        Returns:
            Optional[PlaceDetails]: AI 생성 장소 상세 정보. 실패 시 None.
        """
        place_name_with_country = f"{waypoint.name}"
        if waypoint.country:
            place_name_with_country = f"{waypoint.name}, {waypoint.country}"

        prompt = f"""
        You are an expert travel guide. Provide detailed and engaging information about the following place.
        Return the response in a JSON format matching the provided schema.
        
        Place: {place_name_with_country}

        Please include:
        1.  A general overview/introduction of the place.
        2.  Top activities or things to do there.
        3.  Interesting cultural insights or historical facts.
        4.  The best time to visit and why.

        Ensure all fields are populated, even if briefly. If a specific detail is not applicable, state it concisely.
        """
        
        # Pydantic 모델의 JSON 스키마를 직접 사용
        place_details_schema = PlaceDetails.model_json_schema()
        
        json_string_response = await self._generate_content_with_retry(prompt, schema=place_details_schema)
        
        if json_string_response:
            try:
                # Gemini가 반환한 JSON 문자열을 파싱
                parsed_json = json.loads(json_string_response)
                # Waypoint 이름을 PlaceDetails에 추가 (Gemini가 제공하지 않을 수 있으므로)
                parsed_json['name'] = waypoint.name 
                return PlaceDetails(**parsed_json)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from Gemini for {waypoint.name}: {e}")
                print(f"Raw response: {json_string_response}")
                return None
            except Exception as e:
                print(f"Error parsing PlaceDetails for {waypoint.name}: {e}")
                return None
        return None

# 예시 사용 (main.py에서 직접 호출하지 않고, TravelService를 통해 호출될 것임)
if __name__ == "__main__":
    # 비동기 함수 실행을 위한 래퍼
    async def run_examples():
        gemini_api = GeminiAPI()
        
        # .env 파일에 GEMINI_API_KEY 설정 필요
        
        # 각 장소 정보 요청 예시
        print("\n--- 장소별 상세 정보 요청 예시 ---")
        waypoint1 = Waypoint(name="Eiffel Tower", lat=48.8584, lng=2.2945, country="France")
        waypoint2 = Waypoint(name="Gyeongbokgung Palace", lat=37.5796, lng=126.9770, country="South Korea")
        
        eiffel_details = await gemini_api.get_place_details_for_route(waypoint1)
        if eiffel_details:
            print(f"\n{eiffel_details.name} 상세 정보:")
            print(f"  General: {eiffel_details.general_info[:100]}...")
            print(f"  Activities: {eiffel_details.activities[:100]}...")
        else:
            print(f"\n{waypoint1.name} 정보를 가져올 수 없습니다.")

        gyeongbokgung_details = await gemini_api.get_place_details_for_route(waypoint2)
        if gyeongbokgung_details:
            print(f"\n{gyeongbokgung_details.name} 상세 정보:")
            print(f"  General: {gyeongbokgung_details.general_info[:100]}...")
            print(f"  Cultural: {gyeongbokgung_details.cultural_insights[:100]}...")
        else:
            print(f"\n{waypoint2.name} 정보를 가져올 수 없습니다.")


        # 경로 요약 요청 예시
        print("\n--- 경로 요약 및 팁 요청 예시 ---")
        route_places = ["Eiffel Tower", "Gyeongbokgung Palace", "Colosseum"]
        total_cost_duration = "5000 km, 72 hours"
        route_summary = await gemini_api.get_route_summary_and_tips(route_places, total_cost_duration, "sightseeing")
        if route_summary:
            print(f"\n경로 요약 및 팁:\n{route_summary}\n")
        else:
            print("\n경로 요약을 가져올 수 없습니다.")

    import requests # 비동기 http 요청을 위해 requests 임포트
    # asyncio.run 호출
    asyncio.run(run_examples())
