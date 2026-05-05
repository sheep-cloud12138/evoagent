from evoagent.app import EvoAgentSystem


def test_requested_markdown_document_is_written(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    answer = (
        "以下为文档内容，将保存至 `五一北京出行指南.md`。\n"
        "```markdown\n"
        "# 五一北京出行指南\n\n"
        "## 天气\n"
        "多云。\n"
        "```\n"
    )

    meta = EvoAgentSystem._maybe_write_requested_file("写个文档在当前文件夹", answer)

    assert meta is not None
    assert meta["filename"] == "五一北京出行指南.md"
    target = tmp_path / "五一北京出行指南.md"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "# 五一北京出行指南\n\n## 天气\n多云。\n"


def test_file_output_is_not_written_without_explicit_request(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    meta = EvoAgentSystem._maybe_write_requested_file("解释一下 TCP 三次握手", "# 文档")

    assert meta is None
    assert list(tmp_path.iterdir()) == []


def test_existing_requested_file_gets_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "report.md").write_text("old", encoding="utf-8")
    answer = "```markdown\nnew\n``` 文件名为 report.md"

    meta = EvoAgentSystem._maybe_write_requested_file("保存文档到当前文件夹", answer)

    assert meta is not None
    assert meta["filename"] == "report_2.md"
    assert (tmp_path / "report_2.md").read_text(encoding="utf-8") == "new\n"
