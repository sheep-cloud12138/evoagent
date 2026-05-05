import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.weather import extract_weather_location, is_weather_query


def test_extract_weather_location_from_chinese_query() -> None:
    assert is_weather_query("珠海今天天气怎么样")
    assert extract_weather_location("珠海今天天气怎么样") == "珠海"
    assert extract_weather_location("今天广州天气如何？") == "广州"


def test_extract_weather_location_from_english_query() -> None:
    assert is_weather_query("weather in Zhuhai today")
    assert extract_weather_location("weather in Zhuhai today") == "Zhuhai"


def test_weather_query_uses_builtin_fastpath(monkeypatch) -> None:
    monkeypatch.setattr(
        "evoagent.app.fetch_weather_report",
        lambda location: f"{location}当前天气：晴，气温 28°C。\n数据源：test",
    )
    system = EvoAgentSystem()
    result = asyncio.run(system.run("珠海今天天气怎么样", context={"session_id": "test-weather-fastpath"}))

    assert result.metadata.get("mode") == "builtin_weather_fastpath"
    assert result.metadata.get("weather_location") == "珠海"
    assert result.outputs == []
    assert "珠海当前天气" in result.final_answer
