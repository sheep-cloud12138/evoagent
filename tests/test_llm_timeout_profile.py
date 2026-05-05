from evoagent.core.llm import LLMClient


def test_timeout_profile_ordering() -> None:
    client = LLMClient()
    t_fast = client._timeout_for_profile(LLMClient.Profile.FAST)
    t_std = client._timeout_for_profile(LLMClient.Profile.STANDARD)
    t_reason = client._timeout_for_profile(LLMClient.Profile.REASONING)
    assert t_fast <= t_std <= t_reason
