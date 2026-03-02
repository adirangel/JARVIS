"""Weather endpoint - Open-Meteo API (no key required)."""

import httpx
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["weather"])

# Default: Quezon City, PH (from reference image)
DEFAULT_LAT = 14.6760
DEFAULT_LON = 121.0437
DEFAULT_CITY = "Quezon City, PH"


@router.get("/weather")
async def get_weather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
    """Return weather from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {
            "temp_c": 25.2,
            "location": DEFAULT_CITY,
            "condition": "overcast clouds",
            "humidity": 94,
            "wind_m_s": 5.8,
            "feels_like_c": 26.3,
            "error": str(e),
        }

    current = data.get("current", {})
    code = current.get("weather_code", 0)
    condition = _weather_code_to_text(code)

    return {
        "temp_c": current.get("temperature_2m"),
        "location": DEFAULT_CITY,
        "condition": condition,
        "humidity": current.get("relative_humidity_2m"),
        "wind_m_s": current.get("wind_speed_10m"),
        "feels_like_c": current.get("apparent_temperature"),
    }


def _weather_code_to_text(code: int) -> str:
    """WMO weather code to human-readable text."""
    codes = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        61: "slight rain",
        80: "slight rain showers",
        95: "thunderstorm",
    }
    return codes.get(code, "overcast clouds")
