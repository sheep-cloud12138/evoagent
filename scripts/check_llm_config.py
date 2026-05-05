from __future__ import annotations

import argparse
import time

from evoagent.core.config import settings
from evoagent.core.llm import LLMClient


def _mask_presence(value: str | None) -> str:
    return "set" if value else "empty"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check EvoAgent LLM routing without printing secrets.")
    parser.add_argument("--prompt", default="只回复 OK", help="Prompt used for the smoke test.")
    parser.add_argument("--skip-call", action="store_true", help="Only print sanitized config.")
    args = parser.parse_args()

    client = LLMClient()
    candidates = client._model_candidates(LLMClient.Profile.STANDARD)
    normalized = [client._normalize_model(model) for model in candidates]
    active = [client._normalize_model(model) for model in client._prepare_model_candidates(candidates)]

    print("LLM_PROVIDER=", settings.llm_provider)
    print("MODEL=", settings.llm_model)
    print("STANDARD_MODEL=", settings.llm_standard_model)
    print("REASONING_MODEL=", settings.llm_reasoning_model)
    print("NORMALIZED_STANDARD_CANDIDATES=", ",".join(normalized))
    print("ACTIVE_STANDARD_CANDIDATES=", ",".join(active))
    print("LLM_TIMEOUT_SECONDS=", settings.llm_timeout_seconds)
    print("LLM_HEALTH_CACHE_ENABLED=", settings.llm_health_cache_enabled)
    print("LLM_HEALTH_CACHE_PATH=", settings.llm_health_cache_path)
    print("OPENAI_API_KEY=", _mask_presence(settings.openai_api_key))
    print("DEEPSEEK_API_KEY=", _mask_presence(settings.deepseek_api_key))
    print("QWEN_API_KEY=", _mask_presence(settings.qwen_api_key))
    print("VOLCENGINE_API_KEY=", _mask_presence(settings.volcengine_api_key))
    print("VOLCENGINE_BASE_URL=", settings.volcengine_base_url)
    print("MODEL_HEALTH=")
    for row in client.health_report(candidates):
        last_error = str(row["last_error"]).replace("\n", " ")[:120]
        print(
            "  - "
            f"model={row['model']} "
            f"status={row['status']} "
            f"cooldown_remaining_seconds={row['cooldown_remaining_seconds']} "
            f"failures={row['failures']} "
            f"last_error={last_error}"
        )

    if args.skip_call:
        return

    started = time.perf_counter()
    answer = client.generate(args.prompt, temperature=0.0, profile=LLMClient.Profile.STANDARD)
    elapsed = time.perf_counter() - started
    print("SMOKE_ELAPSED_SECONDS=", round(elapsed, 2))
    print("SMOKE_USED_FALLBACK=", answer.startswith("[Fallback-LLM]"))
    print("SMOKE_PREVIEW=", answer[:240].replace("\n", " "))


if __name__ == "__main__":
    main()
