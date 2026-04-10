from __future__ import annotations

from aura.runtime.skills import SkillStore


def test_skills_loading(make_runtime):
    rt = make_runtime(tools_enabled=False)

    skill_dir = rt.project_root / ".aura" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test skill.\n---\n\nUse this skill in tests.\n",
        encoding="utf-8",
    )
    (skill_dir / "ref.txt").write_text("hello", encoding="utf-8")

    store = SkillStore(project_root=rt.project_root)
    skills = store.list()
    names = [s.name for s in skills]
    assert "test-skill" in names

    loaded = store.load("test-skill")
    assert loaded.meta.name == "test-skill"
    assert "Test skill." in loaded.meta.description
    assert "Use this skill" in loaded.instructions
    assert "ref.txt" in loaded.resources

