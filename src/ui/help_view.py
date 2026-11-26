"""Rich-native rendering for the grouped command help.

The help used to be one pre-joined markup string, so the groups were only
indentation and the line could not re-wrap when the terminal narrowed.
``help_renderable`` builds a ``rich.tree.Tree`` instead: one branch per group,
one leaf per command with a dim description.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Group
from rich.text import Text
from rich.tree import Tree

from .console import resolve_style


def _command_leaf(name: str, description: str) -> Text:
    label = Text()
    label.append(name, style=resolve_style("accent"))
    label.append(" — ", style=resolve_style("muted"))
    label.append(description)
    return label


def help_tree(
    groups: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
    *,
    title: str = "Commands",
    flat: bool = False,
) -> Tree:
    """Build the grouped command tree.

    ``flat`` puts every command directly under the root in mapping order, which
    mirrors the ungrouped ``/help`` mode. Commands missing from ``descriptions``
    are skipped, and a command listed in two groups appears once.
    """
    # Styled in two passes: rich cannot parse a compound style containing a
    # theme name ("bold accent"), only a whole-token lookup.
    title_text = Text(title, style="bold")
    title_text.stylize(resolve_style("accent"))
    root = Tree(title_text, guide_style=resolve_style("muted"))
    seen: set[str] = set()
    if flat:
        for name, description in descriptions.items():
            root.add(_command_leaf(name, description))
        return root
    for group_name, command_names in groups.items():
        commands = [
            name for name in command_names if name in descriptions and name not in seen
        ]
        if not commands:
            continue
        branch = root.add(
            Text(group_name, style="bold"), guide_style=resolve_style("muted")
        )
        for name in commands:
            seen.add(name)
            branch.add(_command_leaf(name, descriptions[name]))
    leftover = [name for name in descriptions if name not in seen]
    if leftover:
        branch = root.add(
            Text("Other", style="bold"), guide_style=resolve_style("muted")
        )
        for name in leftover:
            branch.add(_command_leaf(name, descriptions[name]))
    return root


def help_renderable(
    groups: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, str],
    *,
    title: str = "Commands",
    footer: Sequence[str] = (),
    flat: bool = False,
) -> Group:
    """Return the tree plus the footer lines as one printable renderable.

    ``footer`` entries are markup strings, matching the string help so both
    modes describe file context in the same words.
    """
    renderables: list = [help_tree(groups, descriptions, title=title, flat=flat)]
    for line in footer:
        plain = Text.from_markup(line).plain
        renderables.append(Text(plain, style=resolve_style("muted")))
    return Group(*renderables)


__all__ = ["help_renderable", "help_tree"]
