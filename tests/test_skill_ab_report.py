from evoagent.skills.runtime import SkillArtifactManager


def test_ab_report_record(tmp_path) -> None:
    mgr = SkillArtifactManager(tmp_path / "store")
    mgr.record_ab_result("demo_skill", "0.1.0", "skill")
    mgr.record_ab_result("demo_skill", "0.1.0", "baseline")
    report = mgr.ab_report()
    key = "demo_skill:0.1.0"
    assert key in report
    assert report[key]["total"] == 2
