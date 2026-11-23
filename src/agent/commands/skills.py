"""Handler for the /skills slash command."""


def handle_skills_command(user_input, loaded_skills, selected_skill_key, session_ui):
    """Handle /skills and return (selected_skill_key, loaded_skills)."""
    import re

    requested_skill = user_input[len("/skills") :].strip()
    if not requested_skill:
        if not loaded_skills:
            session_ui.print(
                "[warning][Skills][/warning] No skills are loaded. "
                "Configure a skills folder containing Markdown "
                "skill files to use this command."
            )
            return selected_skill_key, loaded_skills
        from rich.table import Table

        skill_table = Table(
            title="Available Skills",
            show_header=True,
            header_style="bold cyan",
        )
        skill_table.add_column("Name")
        skill_table.add_column("Command key")
        skill_table.add_column("Description")
        for key, skill in sorted(
            loaded_skills.items(),
            key=lambda item: item[1].name.casefold(),
        ):
            skill_table.add_row(skill.name, key, skill.description)
        session_ui.print(skill_table)
        session_ui.print(
            "Select one for your next request with "
            "`/skills <command key>`; use Tab to complete available keys."
        )
        return selected_skill_key, loaded_skills

    normalized_skill = re.sub(r"[\s-]+", "_", requested_skill).casefold()
    selected_match = next(
        (
            key
            for key, skill in loaded_skills.items()
            if re.sub(r"[\s-]+", "_", key).casefold() == normalized_skill
            or re.sub(r"[\s-]+", "_", skill.name).casefold() == normalized_skill
        ),
        None,
    )
    if selected_match is None:
        session_ui.print(
            f"[warning][Skills][/warning] Skill "
            f"{requested_skill!r} is not available. Use `/skills` "
            "to list loaded skills."
        )
        return selected_skill_key, loaded_skills
    selected_skill_key = selected_match
    selected_name = loaded_skills[selected_match].name
    session_ui.print(
        f"[info][Skills][/info] {selected_name} selected for your next request."
    )
    return selected_skill_key, loaded_skills
