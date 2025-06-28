from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Waypoint(BaseModel):
    name: str = Field(..., description="The name of the waypoint (e.g., 'Eiffel Tower', 'Seoul Station').")
    lat: float = Field(..., description="Latitude of the waypoint.")
    lng: float = Field(..., description="Longitude of the waypoint.")
    country: Optional[str] = Field(None, description="Optional country name for better Gemini context.")

class OptimizeRouteRequest(BaseModel):
    waypoints: List[Waypoint] = Field(..., min_items=2, description="List of waypoints to optimize the route for.")
    algorithm: Literal["greedy", "simulated_annealing", "genetic_algorithm"] = Field(
        "greedy", description="The optimization algorithm to use."
    )
    optimization_metric: Literal["distance", "duration"] = Field(
        "duration", description="The metric to optimize: 'distance' (meters) or 'duration' (seconds)."
    )
    # 각 알고리즘별 파라미터는 필요에 따라 추가
    sa_initial_temperature: Optional[float] = None
    sa_cooling_rate: Optional[float] = None
    sa_max_iterations: Optional[int] = None
    ga_population_size: Optional[int] = None
    ga_generations: Optional[int] = None
    ga_mutation_rate: Optional[float] = None
    ga_crossover_rate: Optional[float] = None

class OptimizedRouteResult(BaseModel):
    algorithm: str = Field(..., description="The name of the algorithm used.")
    optimized_path_indices: List[int] = Field(..., description="The optimized order of waypoint indices.")
    optimized_path_names: List[str] = Field(..., description="The optimized order of waypoint names.")
    total_cost: float = Field(..., description="The total cost of the optimized route (in meters or seconds).")
    cost_unit: str = Field(..., description="Unit of the total_cost ('meters' or 'seconds').")
    execution_time_seconds: float = Field(..., description="Time taken to run the optimization algorithm.")
    gemini_summary: Optional[str] = Field(None, description="AI-generated summary and tips for the optimized route.")

class GeminiInfoRequest(BaseModel):
    place_name: str
    country: Optional[str] = None
    query_type: Literal["general_info", "activities", "cultural_insights", "best_time_to_visit"] = "general_info"

class GeminiInfoResponse(BaseModel):
    place_name: str
    country: Optional[str] = None
    query_type: str
    info: str