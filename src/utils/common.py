from typing import Union

# calculate_path_cost 함수는 각 알고리즘 솔버 내부에서 TSP 로직과 함께 이미 구현되어 있으므로,
# 여기서는 제거하여 역할의 중복과 혼란을 방지합니다.

def format_cost(cost: Union[float, int], unit: str = "meters") -> str:
    """
    비용 값을 보기 좋게 포맷팅합니다 (예: 미터 -> km, 초 -> 시간/분/초).
    
    Args:
        cost (Union[float, int]): 비용 값.
        unit (str): 원본 단위 ("meters" 또는 "seconds").
    Returns:
        str: 포맷팅된 문자열.
    """
    if cost == float('inf'):
        return "N/A (No Valid Path)"
    
    # 숫자 포맷팅을 위한 로케일 설정 (선택 사항, 필요 시 추가)
    # import locale
    # locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # 또는 'ko_KR.UTF-8'

    if unit == "meters":
        if cost >= 1000:
            return f"{cost / 1000:,.2f} km" # 콤마와 소수점 2자리
        return f"{cost:,.0f} meters" # 콤마와 정수
    elif unit == "seconds":
        hours = int(cost // 3600)
        minutes = int((cost % 3600) // 60)
        seconds = int(cost % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or (hours == 0 and minutes == 0): # 0초일 때도 0s 표시
            parts.append(f"{seconds}s")
        
        return " ".join(parts) if parts else "0s"
    else:
        # 알 수 없는 단위인 경우 원본 값을 문자열로 반환 (콤마 포맷팅 적용)
        return f"{cost:,.2f} {unit}" if isinstance(cost, float) else f"{cost:,} {unit}"

# 예시 사용
if __name__ == "__main__":
    print("--- 비용 포맷팅 테스트 ---")
    
    # Meters 테스트
    print(f"Distance 1: {format_cost(100000.5, unit='meters')}") # 100.00 km
    print(f"Distance 2: {format_cost(500, unit='meters')}")    # 500 meters
    print(f"Distance 3: {format_cost(1234567.89, unit='meters')}") # 1,234.57 km
    print(f"Distance 4: {format_cost(0, unit='meters')}") # 0 meters
    print(f"Distance 5: {format_cost(float('inf'), unit='meters')}") # N/A (No Valid Path)

    # Seconds 테스트
    print(f"Duration 1: {format_cost(3665, unit='seconds')}")   # 1h 1m 5s
    print(f"Duration 2: {format_cost(125, unit='seconds')}")    # 2m 5s
    print(f"Duration 3: {format_cost(59, unit='seconds')}")     # 59s
    print(f"Duration 4: {format_cost(0, unit='seconds')}")      # 0s
    print(f"Duration 5: {format_cost(7200, unit='seconds')}")   # 2h 0s (분 단위가 0일 때)
    print(f"Duration 6: {format_cost(3600, unit='seconds')}")   # 1h 0s

    # 알 수 없는 단위 테스트
    print(f"Unknown Unit: {format_cost(123.456, unit='points')}") # 123.46 points
    print(f"Unknown Unit Int: {format_cost(1000, unit='units')}") # 1,000 units
