from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from evoagent.app import EvoAgentSystem
from evoagent.core.models import ExecutionResult

app = typer.Typer(no_args_is_help=True)
console = Console()


@dataclass
class _ChatState:
    session_id: str = "default"
    json_output: bool = False
    turn_count: int = 0
    last_quality: float | None = None
    last_latency: float | None = None


@dataclass
class _ChatMessage:
    role: str
    content: str
    meta: dict[str, Any]


def _normalize_session_id(value: str) -> str:
    cleaned = value.strip()
    return cleaned or "default"


def _render_result(result: ExecutionResult, json_output: bool = False) -> None:
    if json_output:
        console.print(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2), markup=False
        )
        return

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="bold cyan")
    stats.add_column()
    stats.add_row("task_id", result.task_id)
    stats.add_row("difficulty", result.decision.difficulty.value)
    stats.add_row("score", f"{result.decision.score:.2f}")
    stats.add_row("quality", f"{result.quality_score:.2f}")
    stats.add_row("latency", f"{result.latency_seconds:.2f}s")

    session_id = result.metadata.get("session_id")
    turns_used = result.metadata.get("conversation_turns_used")
    if session_id is not None:
        stats.add_row("session_id", str(session_id))
    if turns_used is not None:
        stats.add_row("history_turns", str(turns_used))
    post_actions = result.metadata.get("post_actions")
    if isinstance(post_actions, dict):
        skill_evolution = post_actions.get("skill_evolution")
        if skill_evolution is not None:
            stats.add_row("skill_evolution", str(skill_evolution))
        registered_tools = post_actions.get("skill_tools_registered")
        if registered_tools:
            stats.add_row(
                "skill_tools",
                ", ".join(str(item) for item in list(registered_tools)[:4]),
            )
        file_output = post_actions.get("file_output")
        if isinstance(file_output, dict) and file_output.get("path"):
            stats.add_row("file_output", str(file_output.get("path")))

    console.print(stats)
    console.print("\n[bold green]final answer[/bold green]")
    console.print(result.final_answer)


def _extract_tool_summary(result: ExecutionResult) -> tuple[int, list[str]]:
    names: list[str] = []
    for output in result.outputs:
        meta = output.metadata or {}
        events = meta.get("tool_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            raw_name = str(event.get("name", "")).strip()
            if raw_name:
                names.append(raw_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return len(names), deduped


def _render_assistant_message(result: ExecutionResult) -> None:
    tool_count, tools = _extract_tool_summary(result)
    badges = [
        f"difficulty={result.decision.difficulty.value}",
        f"quality={result.quality_score:.2f}",
        f"latency={result.latency_seconds:.2f}s",
        f"tools={tool_count}",
    ]
    if tools:
        badges.append(f"tool_names={', '.join(tools[:4])}")
    mode = result.metadata.get("mode")
    if mode:
        badges.append(f"mode={mode}")
    post_actions = result.metadata.get("post_actions")
    if isinstance(post_actions, dict):
        skill_evolution = post_actions.get("skill_evolution")
        if skill_evolution:
            badges.append(f"skill_evolution={skill_evolution}")
        file_output = post_actions.get("file_output")
        if isinstance(file_output, dict) and file_output.get("filename"):
            badges.append(f"file={file_output.get('filename')}")

    console.print(
        Panel(
            result.final_answer,
            title="EvoAgent",
            subtitle=" | ".join(badges),
            border_style="green",
            expand=True,
        )
    )


def _render_user_message(text: str) -> None:
    console.print(
        Panel(
            text,
            title="You",
            border_style="cyan",
            expand=True,
        )
    )


def _render_chat_shell(state: _ChatState) -> None:
    console.print(
        Panel.fit(
            "EvoAgent CLI UI\n"
            "- Enter your task directly to run\n"
            "- Commands start with '/'\n"
            "- Use /help to view all shortcuts",
            title="Agent Console",
            border_style="bright_blue",
        )
    )
    _print_status_bar(state)


def _print_status_bar(state: _ChatState) -> None:
    quality_text = "-" if state.last_quality is None else f"{state.last_quality:.2f}"
    latency_text = "-" if state.last_latency is None else f"{state.last_latency:.2f}s"
    console.print(
        f"[dim]session={state.session_id} | turns={state.turn_count} | "
        f"json={state.json_output} | last_quality={quality_text} | last_latency={latency_text}[/dim]"
    )


def _render_transcript(messages: list[_ChatMessage], limit: int = 6) -> None:
    if not messages:
        console.print("[dim]No local transcript yet.[/dim]")
        return

    tail = messages[-max(1, limit) :]
    for item in tail:
        title = "You" if item.role == "user" else "EvoAgent"
        style = "cyan" if item.role == "user" else "green"
        subtitle = ""
        if item.role != "user":
            quality = item.meta.get("quality")
            latency = item.meta.get("latency")
            tools = item.meta.get("tools")
            subtitle = f"quality={quality} latency={latency}s tools={tools}"

        console.print(
            Panel(
                item.content,
                title=title,
                subtitle=subtitle,
                border_style=style,
                expand=True,
            )
        )


def _print_chat_help() -> None:
    table = Table(
        title="EvoAgent Chat Commands", show_header=True, header_style="bold cyan"
    )
    table.add_column("command", style="bold")
    table.add_column("description")
    table.add_row("/help", "Show available commands")
    table.add_row("/status", "Show current chat settings")
    table.add_row("/new", "Switch to a new auto-generated session id")
    table.add_row("/session <id>", "Switch active session id")
    table.add_row("/reset", "Clear current session memory")
    table.add_row("/history [n]", "Show local transcript (default 6)")
    table.add_row("/clear", "Clear terminal screen")
    table.add_row("/json on|off", "Toggle JSON response output")
    table.add_row("/exit", "Exit chat mode")
    console.print(table)


def _handle_chat_command(
    raw_text: str,
    state: _ChatState,
    system: EvoAgentSystem,
    transcript: list[_ChatMessage],
) -> tuple[bool, bool]:
    text = raw_text.strip()
    if not text.startswith("/"):
        return False, False

    parts = text[1:].split()
    if not parts:
        return True, False

    command = parts[0].lower()
    args = parts[1:]

    if command in {"exit", "quit", "q"}:
        return True, True

    if command == "help":
        _print_chat_help()
        return True, False

    if command == "status":
        _print_status_bar(state)
        return True, False

    if command == "new":
        state.session_id = datetime.now().strftime("chat-%Y%m%d-%H%M%S")
        console.print(f"Started new session: {state.session_id}")
        return True, False

    if command == "session":
        if not args:
            console.print("Usage: /session <session_id>")
            return True, False
        state.session_id = _normalize_session_id(" ".join(args))
        console.print(f"Switched session_id to: {state.session_id}")
        return True, False

    if command == "reset":
        deleted = system.memory.conversation.clear_session(state.session_id)
        transcript.clear()
        console.print(f"Cleared conversation for session_id={state.session_id}")
        console.print(f"Deleted turns: {deleted}")
        state.turn_count = 0
        state.last_quality = None
        state.last_latency = None
        _print_status_bar(state)
        return True, False

    if command == "history":
        limit = 6
        if args:
            try:
                limit = max(1, int(args[0]))
            except ValueError:
                console.print("Usage: /history [n]")
                return True, False
        _render_transcript(transcript, limit=limit)
        return True, False

    if command == "clear":
        console.clear()
        _render_chat_shell(state)
        return True, False

    if command == "json":
        if not args:
            console.print("Usage: /json on|off")
            return True, False
        flag = args[0].lower()
        if flag in {"on", "true", "1"}:
            state.json_output = True
        elif flag in {"off", "false", "0"}:
            state.json_output = False
        else:
            console.print("Usage: /json on|off")
            return True, False
        console.print(f"json_output={state.json_output}")
        return True, False

    console.print(f"Unknown command: /{command}. Use /help.")
    return True, False


@app.command()
def run(
    query: str,
    json_output: bool = False,
    session_id: str = "default",
    reset_session: bool = False,
    api: bool = False,
    api_url: str = "http://127.0.0.1:8000",
) -> None:
    """Run EvoAgent with a user query."""
    if api:
        response = httpx.post(
            f"{api_url.rstrip('/')}/runs",
            json={"query": query, "metadata": {"session_id": session_id}},
            timeout=20,
        )
        response.raise_for_status()
        console.print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return

    system = EvoAgentSystem()
    if reset_session:
        system.memory.conversation.clear_session(session_id)
    result = system.gateway.run_sync(query, context={"session_id": session_id})
    _render_result(result, json_output=json_output)


@app.command()
def chat(
    session_id: str = "default",
    json_output: bool = False,
    reset_session: bool = False,
) -> None:
    """Interactive terminal chat UI for EvoAgent."""
    system = EvoAgentSystem()
    state = _ChatState(
        session_id=_normalize_session_id(session_id),
        json_output=json_output,
    )
    transcript: list[_ChatMessage] = []
    if reset_session:
        system.memory.conversation.clear_session(state.session_id)

    _render_chat_shell(state)

    while True:
        try:
            user_input = Prompt.ask(
                f"[bold cyan]{state.session_id}[/bold cyan] [dim]▸[/dim]"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting chat.")
            break

        if not user_input:
            continue

        handled, should_exit = _handle_chat_command(
            user_input, state, system, transcript
        )
        if handled:
            if should_exit:
                console.print("Exiting chat.")
                break
            continue

        _render_user_message(user_input)
        transcript.append(_ChatMessage(role="user", content=user_input, meta={}))
        with console.status("[bold cyan]Running task...[/bold cyan]"):
            result = system.gateway.run_sync(
                user_input, context={"session_id": state.session_id}
            )

        state.turn_count += 1
        state.last_quality = result.quality_score
        state.last_latency = result.latency_seconds

        if state.json_output:
            _render_result(result, json_output=True)
        else:
            _render_assistant_message(result)

        tool_count, _ = _extract_tool_summary(result)
        transcript.append(
            _ChatMessage(
                role="assistant",
                content=result.final_answer,
                meta={
                    "quality": f"{result.quality_score:.2f}",
                    "latency": f"{result.latency_seconds:.2f}",
                    "tools": str(tool_count),
                },
            )
        )
        _print_status_bar(state)


@app.command("console")
def chat_alias(
    session_id: str = "default",
    json_output: bool = False,
    reset_session: bool = False,
) -> None:
    """Alias for chat command."""
    chat(session_id=session_id, json_output=json_output, reset_session=reset_session)


@app.command()
def recent(limit: int = 5) -> None:
    """Show recent episodic memory records."""
    system = EvoAgentSystem()
    records = system.memory.episodic.search_recent(limit=limit)
    if not records:
        console.print("No history found.")
        return

    for rec in records:
        console.print(
            f"- id={rec.id} success={rec.success} score={rec.score:.2f} summary={rec.summary[:120]}"
        )


@app.command()
def semantic(query: str, top_k: int = 3) -> None:
    """Query distilled semantic memory."""
    system = EvoAgentSystem()
    items = system.memory.recall_semantic(query, top_k=top_k)
    if not items:
        console.print("No semantic memory hit.")
        return

    for item in items:
        console.print(
            f"- id={item['id']} fact={item['fact'][:150]} metadata={item['metadata']}"
        )


@app.command()
def skills(status: str = "active", limit: int = 20) -> None:
    """List skills in registry."""
    system = EvoAgentSystem()
    records = system.skill_registry.list_skills(status=status, limit=limit)
    if not records:
        console.print("No skills found.")
        return

    for rec in records:
        console.print(
            f"- id={rec.id} name={rec.name} v={rec.version} calls={rec.calls} "
            f"rate={rec.success_rate:.2f} status={rec.status}"
        )


@app.command("skill-versions")
def skill_versions(name: str) -> None:
    """Show all persisted versions for a skill."""
    system = EvoAgentSystem()
    items = system.skill_artifacts.list_versions(name)
    if not items:
        console.print("No versions found.")
        return
    for item in items:
        console.print(
            f"- name={item.get('name')} v={item.get('version')} status={item.get('status')}"
        )


@app.command("skill-rollback")
def skill_rollback(name: str, version: str) -> None:
    """Activate a historical skill version and archive others."""
    system = EvoAgentSystem()
    fs_ok = system.skill_artifacts.activate_version(name, version)
    db_ok = system.skill_registry.activate_version(name, version)
    payload = {
        "name": name,
        "version": version,
        "artifact_ok": fs_ok,
        "registry_ok": db_ok,
    }
    console.print(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("skill-ab-report")
def skill_ab_report() -> None:
    """Show A/B testing report for auto-applied evolved skills."""
    system = EvoAgentSystem()
    report = system.skill_artifacts.ab_report()
    console.print(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def health() -> None:
    """Check current runtime health and key components."""
    system = EvoAgentSystem()
    recent_records = system.memory.episodic.search_recent(limit=1)
    active_skills = system.skill_registry.list_skills(status="active", limit=1)
    payload = {
        "status": "ok",
        "has_recent_episode": bool(recent_records),
        "has_active_skill": bool(active_skills),
    }
    console.print(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def ui(host: str = "127.0.0.1", port: int = 7788) -> None:
    """Deprecated: use terminal chat UI instead."""
    _ = host
    _ = port
    console.print("Web UI has been removed from this build.")
    console.print("Please use: evoagent chat")
    console.print("Or alias: evoagent console")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve the FastAPI gateway."""
    import uvicorn

    uvicorn.run("evoagent.api.main:app", host=host, port=port)


if __name__ == "__main__":
    app()
