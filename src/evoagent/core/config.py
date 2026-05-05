from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")
    llm_model: str = Field(
        default="gpt-4.1-mini",
        validation_alias=AliasChoices("MODEL", "LLM_MODEL"),
    )
    llm_fast_model: str = Field(
        default="gpt-4.1-mini",
        validation_alias=AliasChoices("FAST_MODEL", "LLM_FAST_MODEL"),
    )
    llm_standard_model: str = Field(
        default="gpt-4.1",
        validation_alias=AliasChoices("STANDARD_MODEL", "LLM_STANDARD_MODEL"),
    )
    llm_reasoning_model: str = Field(
        default="o3-mini",
        validation_alias=AliasChoices("REASONING_MODEL", "LLM_REASONING_MODEL"),
    )
    llm_backup_models: str = Field(default="", alias="LLM_BACKUP_MODELS")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str | None = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str | None = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    qwen_api_key: str | None = Field(default=None, alias="QWEN_API_KEY")
    qwen_base_url: str | None = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    volcengine_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VOLCENGINE_API_KEY", "ARK_API_KEY"),
    )
    volcengine_base_url: str | None = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        validation_alias=AliasChoices(
            "VOLCENGINE_BASE_URL", "ARK_BASE_URL", "ARK_API_BASE"
        ),
    )
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    github_api_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GITHUB_API_TOKEN", "GITHUB_TOKEN"),
    )

    db_path: Path = Field(default=Path("data/evoagent.db"), alias="EVO_DB_PATH")
    semantic_store_path: Path = Field(
        default=Path("data/semantic_store"), alias="EVO_SEMANTIC_PATH"
    )
    skill_store_path: Path = Field(
        default=Path("data/skills_store"), alias="EVO_SKILL_STORE_PATH"
    )
    workspace_store_path: Path = Field(
        default=Path("data/workspaces"), alias="EVO_WORKSPACE_STORE_PATH"
    )
    use_langgraph: bool = Field(default=False, alias="USE_LANGGRAPH")

    simple_threshold: float = 0.34
    medium_threshold: float = 0.67

    max_parallel_subagents: int = 4
    skill_success_decay_days: int = 30
    skill_ab_test_rate: float = 0.2
    llm_timeout_seconds: int = 30
    llm_temperature_default: float = 0.15
    llm_model_failure_cooldown_seconds: int = Field(
        default=300,
        alias="LLM_MODEL_FAILURE_COOLDOWN_SECONDS",
    )
    llm_max_candidate_attempts: int = Field(
        default=2, alias="LLM_MAX_CANDIDATE_ATTEMPTS"
    )
    llm_health_cache_enabled: bool = Field(
        default=True, alias="LLM_HEALTH_CACHE_ENABLED"
    )
    llm_health_cache_path: Path = Field(
        default=Path("data/observability/llm_health.json"),
        alias="LLM_HEALTH_CACHE_PATH",
    )
    llm_disable_provider_noise: bool = Field(
        default=True, alias="LLM_DISABLE_PROVIDER_NOISE"
    )
    llm_enable_function_call: bool = Field(
        default=True, alias="LLM_ENABLE_FUNCTION_CALL"
    )
    llm_tool_max_rounds: int = Field(default=2, alias="LLM_TOOL_MAX_ROUNDS")
    mcp_server_url: str | None = Field(default=None, alias="MCP_SERVER_URL")
    mcp_auth_token: str | None = Field(default=None, alias="MCP_AUTH_TOKEN")
    mcp_timeout_seconds: int = Field(default=20, alias="MCP_TIMEOUT_SECONDS")
    main_agent_model: str = Field(
        default="deepseek/deepseek-chat", alias="MAIN_AGENT_MODEL"
    )
    subagent_api_first: bool = Field(default=True, alias="SUBAGENT_API_FIRST")
    subagent_retry_attempts: int = Field(default=2, alias="SUBAGENT_RETRY_ATTEMPTS")
    subagent_retry_backoff_seconds: float = Field(
        default=0.8, alias="SUBAGENT_RETRY_BACKOFF_SECONDS"
    )
    subagent_api_timeout_seconds: int = Field(
        default=20, alias="SUBAGENT_API_TIMEOUT_SECONDS"
    )
    subagent_api_retries: int = Field(default=3, alias="SUBAGENT_API_RETRIES")
    router_low_confidence_threshold: float = Field(
        default=0.45, alias="ROUTER_LOW_CONFIDENCE_THRESHOLD"
    )
    tool_success_rate_threshold: float = Field(
        default=0.35, alias="TOOL_SUCCESS_RATE_THRESHOLD"
    )
    memory_distill_score_threshold: float = Field(
        default=0.74, alias="MEMORY_DISTILL_SCORE_THRESHOLD"
    )
    orchestrator_task_timeout_seconds: float = Field(
        default=90.0, alias="ORCHESTRATOR_TASK_TIMEOUT_SECONDS"
    )
    reflection_enabled: bool = Field(default=False, alias="REFLECTION_ENABLED")
    reflection_max_rounds: int = Field(default=1, alias="REFLECTION_MAX_ROUNDS")
    reflection_min_confidence: float = Field(
        default=0.6, alias="REFLECTION_MIN_CONFIDENCE"
    )
    reflection_complex_threshold: int = Field(
        default=4, alias="REFLECTION_COMPLEX_THRESHOLD"
    )
    reflection_min_avg_confidence: float = Field(
        default=0.68, alias="REFLECTION_MIN_AVG_CONFIDENCE"
    )
    reflection_min_router_confidence: float = Field(
        default=0.55, alias="REFLECTION_MIN_ROUTER_CONFIDENCE"
    )
    graph_retry_enabled: bool = Field(default=True, alias="GRAPH_RETRY_ENABLED")
    graph_max_retries: int = Field(default=1, alias="GRAPH_MAX_RETRIES")
    graph_retry_quality_threshold: float = Field(
        default=0.72, alias="GRAPH_RETRY_QUALITY_THRESHOLD"
    )
    observability_enabled: bool = Field(default=True, alias="OBSERVABILITY_ENABLED")
    observability_log_path: Path = Field(
        default=Path("data/observability/events.jsonl"), alias="OBSERVABILITY_LOG_PATH"
    )
    observability_metrics_path: Path = Field(
        default=Path("data/observability/metrics.json"),
        alias="OBSERVABILITY_METRICS_PATH",
    )
    observability_stdout: bool = Field(default=False, alias="OBSERVABILITY_STDOUT")
    skill_candidate_rollout_rate: float = Field(
        default=0.2, alias="SKILL_CANDIDATE_ROLLOUT_RATE"
    )
    skill_candidate_min_calls: int = Field(default=8, alias="SKILL_CANDIDATE_MIN_CALLS")
    skill_candidate_promote_success_rate: float = Field(
        default=0.75,
        alias="SKILL_CANDIDATE_PROMOTE_SUCCESS_RATE",
    )
    skill_candidate_archive_success_rate: float = Field(
        default=0.4,
        alias="SKILL_CANDIDATE_ARCHIVE_SUCCESS_RATE",
    )
    skill_runtime_timeout_seconds: int = Field(
        default=10, alias="SKILL_RUNTIME_TIMEOUT_SECONDS"
    )
    allow_private_network_fetch: bool = Field(
        default=False, alias="EVO_ALLOW_PRIVATE_NETWORK_FETCH"
    )
    semantic_forget_max_age_days: int = Field(
        default=45, alias="SEMANTIC_FORGET_MAX_AGE_DAYS"
    )
    semantic_forget_min_access_count: int = Field(
        default=1, alias="SEMANTIC_FORGET_MIN_ACCESS_COUNT"
    )
    semantic_forget_max_delete: int = Field(
        default=200, alias="SEMANTIC_FORGET_MAX_DELETE"
    )
    semantic_retrieval_mode: str = Field(
        default="hybrid", alias="SEMANTIC_RETRIEVAL_MODE"
    )
    semantic_embedding_model: str = Field(
        default="text-embedding-3-small", alias="SEMANTIC_EMBEDDING_MODEL"
    )
    semantic_hybrid_alpha: float = Field(default=0.6, alias="SEMANTIC_HYBRID_ALPHA")
    skill_low_value_fallback_max_chars: int = Field(
        default=220, alias="SKILL_LOW_VALUE_FALLBACK_MAX_CHARS"
    )
    skill_low_value_fallback_min_quality: float = Field(
        default=0.82, alias="SKILL_LOW_VALUE_FALLBACK_MIN_QUALITY"
    )
    skill_allowed_permissions: str = Field(
        default="sandbox,memory", alias="SKILL_ALLOWED_PERMISSIONS"
    )
    skill_min_successes: int = Field(default=3, alias="SKILL_MIN_SUCCESSES")
    sandbox_backend: str = Field(default="auto", alias="SANDBOX_BACKEND")
    router_cold_start_history_prior: float = Field(
        default=0.35, alias="ROUTER_COLD_START_HISTORY_PRIOR"
    )
    router_cold_start_min_episodes: int = Field(
        default=8, alias="ROUTER_COLD_START_MIN_EPISODES"
    )
    router_history_pattern_threshold: float = Field(
        default=0.55, alias="ROUTER_HISTORY_PATTERN_THRESHOLD"
    )
    conversation_context_enabled: bool = Field(
        default=True, alias="CONVERSATION_CONTEXT_ENABLED"
    )
    conversation_history_window: int = Field(
        default=8, alias="CONVERSATION_HISTORY_WINDOW"
    )
    conversation_max_turn_chars: int = Field(
        default=240, alias="CONVERSATION_MAX_TURN_CHARS"
    )


settings = Settings()
