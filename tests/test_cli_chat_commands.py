from evoagent.cli import _ChatState, _handle_chat_command


class _FakeConversation:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear_session(self, session_id: str) -> None:
        self.cleared.append(session_id)


class _FakeMemory:
    def __init__(self) -> None:
        self.conversation = _FakeConversation()


class _FakeSystem:
    def __init__(self) -> None:
        self.memory = _FakeMemory()


def test_plain_text_is_not_chat_command() -> None:
    state = _ChatState()
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("hello", state, system, transcript)

    assert handled is False
    assert should_exit is False


def test_session_switch_command() -> None:
    state = _ChatState(session_id="default")
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("/session test-session", state, system, transcript)

    assert handled is True
    assert should_exit is False
    assert state.session_id == "test-session"


def test_reset_command_clears_current_session() -> None:
    state = _ChatState(session_id="session-a")
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("/reset", state, system, transcript)

    assert handled is True
    assert should_exit is False
    assert system.memory.conversation.cleared == ["session-a"]


def test_json_switch_command() -> None:
    state = _ChatState(json_output=False)
    system = _FakeSystem()
    transcript = []

    handled_on, should_exit_on = _handle_chat_command("/json on", state, system, transcript)
    handled_off, should_exit_off = _handle_chat_command("/json off", state, system, transcript)

    assert handled_on is True
    assert should_exit_on is False
    assert handled_off is True
    assert should_exit_off is False
    assert state.json_output is False


def test_exit_command_requests_termination() -> None:
    state = _ChatState()
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("/exit", state, system, transcript)

    assert handled is True
    assert should_exit is True


def test_new_command_generates_chat_session_id() -> None:
    state = _ChatState(session_id="default")
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("/new", state, system, transcript)

    assert handled is True
    assert should_exit is False
    assert state.session_id.startswith("chat-")


def test_history_command_is_handled() -> None:
    state = _ChatState()
    system = _FakeSystem()
    transcript = []

    handled, should_exit = _handle_chat_command("/history 3", state, system, transcript)

    assert handled is True
    assert should_exit is False
