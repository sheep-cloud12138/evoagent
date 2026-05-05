from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WEATHER_CODE_ZH = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴/多云",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "较强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "短时小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float
    timezone: str
    country: str = ""
    admin1: str = ""


def is_weather_query(query: str) -> bool:
    text = query.strip().lower()
    if not text:
        return False
    zh_markers = ("天气", "气温", "降雨", "下雨", "温度")
    en_markers = ("weather", "temperature", "forecast", "rain")
    return any(marker in text for marker in (*zh_markers, *en_markers))


def extract_weather_location(query: str) -> str | None:
    text = query.strip()
    if not text:
        return None

    cleaned = re.sub(r"[?？!！。,.，；;]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    lowered = cleaned.lower()
    en_match = re.search(
        r"(?:weather|temperature|forecast|rain)\s+(?:in|for)\s+([a-zA-Z][a-zA-Z\s'-]{1,60})",
        lowered,
    )
    if en_match:
        return _clean_english_location(en_match.group(1))
    en_match = re.search(
        r"(?:in|for)\s+([a-zA-Z][a-zA-Z\s'-]{1,60})\s+(?:weather|temperature|forecast|rain)",
        lowered,
    )
    if en_match:
        return _clean_english_location(en_match.group(1))

    if any("\u4e00" <= ch <= "\u9fff" for ch in cleaned):
        weather_pos = min(
            [
                idx
                for marker in ("天气", "气温", "降雨", "下雨", "温度")
                if (idx := cleaned.find(marker)) >= 0
            ],
            default=-1,
        )
        candidate = cleaned[:weather_pos] if weather_pos >= 0 else cleaned
        for token in (
            "今天",
            "今日",
            "现在",
            "当前",
            "明天",
            "后天",
            "最近",
            "一下",
            "请问",
            "帮我看看",
            "帮我查",
            "查一下",
            "怎么样",
            "如何",
            "会不会",
            "有没有",
            "的",
            " ",
        ):
            candidate = candidate.replace(token, "")
        candidate = candidate.strip()
        return candidate or None

    return None


def _clean_english_location(text: str) -> str | None:
    words = [
        word
        for word in text.strip().split()
        if word.lower()
        not in {"today", "tomorrow", "now", "current", "forecast", "weather"}
    ]
    cleaned = " ".join(words).strip()
    return cleaned.title() if cleaned else None


def _request_json(url: str, timeout_seconds: int = 12) -> dict[str, Any]:
    req = Request(url=url, headers={"User-Agent": "evoagent-weather/1.0"}, method="GET")
    with urlopen(req, timeout=timeout_seconds) as resp:  # nosec B310
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _location_queries(location: str) -> list[str]:
    q = location.strip()
    if not q:
        return []
    queries = [q]
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    if has_cjk and not q.endswith(("市", "县", "区", "州", "省")):
        queries.insert(0, f"{q}市")
    return list(dict.fromkeys(queries))


def geocode_location(location: str) -> WeatherLocation | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for q in _location_queries(location):
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {
                "name": q,
                "count": 10,
                "language": "zh",
                "format": "json",
            }
        )
        try:
            payload = _request_json(url)
        except Exception:
            continue
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                lat = float(item["latitude"])
                lon = float(item["longitude"])
            except Exception:
                continue
            feature = str(item.get("feature_code", ""))
            feature_rank = {"PPLC": 4, "PPLA": 3, "PPLA2": 2, "PPLA3": 1}.get(
                feature, 0
            )
            population = item.get("population") or 0
            try:
                population_score = min(float(population) / 1_000_000, 20.0)
            except Exception:
                population_score = 0.0
            exact_bonus = (
                2.0
                if str(item.get("name", "")).replace("市", "")
                == location.replace("市", "")
                else 0.0
            )
            country_bonus = 1.0 if str(item.get("country_code", "")) == "CN" else 0.0
            score = feature_rank + population_score + exact_bonus + country_bonus
            if score > best_score:
                best = {**item, "latitude": lat, "longitude": lon}
                best_score = score

    if best is None:
        return None
    return WeatherLocation(
        name=str(best.get("name", location)),
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
        timezone=str(best.get("timezone") or "auto"),
        country=str(best.get("country", "")),
        admin1=str(best.get("admin1", "")),
    )


def fetch_weather_report(location_text: str) -> str:
    location = geocode_location(location_text)
    if location is None:
        return f"error: 未找到地点：{location_text}"

    timezone = (
        location.timezone
        if location.timezone and location.timezone != "auto"
        else "auto"
    )
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": timezone,
            "forecast_days": 1,
        }
    )
    try:
        payload = _request_json(url)
    except Exception as exc:
        return f"error: 天气服务请求失败：{exc}"

    current = (
        payload.get("current", {}) if isinstance(payload.get("current"), dict) else {}
    )
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}
    if not current:
        return "error: 天气服务没有返回当前天气"

    code = int(current.get("weather_code", -1))
    daily_codes = daily.get("weather_code") or []
    daily_code = int(daily_codes[0]) if daily_codes else code
    temp_max = (daily.get("temperature_2m_max") or [None])[0]
    temp_min = (daily.get("temperature_2m_min") or [None])[0]
    rain_probability = (daily.get("precipitation_probability_max") or [None])[0]

    area = "，".join(
        part for part in (location.country, location.admin1, location.name) if part
    )
    parts = [
        f"{area}当前天气：{WEATHER_CODE_ZH.get(code, f'天气代码 {code}')}，气温 {current.get('temperature_2m')}°C",
        f"体感 {current.get('apparent_temperature')}°C，湿度 {current.get('relative_humidity_2m')}%",
        f"风速 {current.get('wind_speed_10m')} km/h，当前降水 {current.get('precipitation')} mm",
    ]
    summary = "；".join(parts) + "。"
    today = (
        f"今天预报：{WEATHER_CODE_ZH.get(daily_code, f'天气代码 {daily_code}')}"
        f"，最高 {temp_max}°C，最低 {temp_min}°C"
    )
    if rain_probability is not None:
        today += f"，最大降水概率 {rain_probability}%"
    today += "。"
    source = "数据源：Open-Meteo（免 API key），结果为自动定位后的实时/今日预报。"
    return f"{summary}\n{today}\n{source}"
