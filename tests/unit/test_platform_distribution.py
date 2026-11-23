import platform

from platform_support import (
    detect_distribution,
    format_install_command,
    system_dependency_status,
)


def test_detect_distribution_from_os_release(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'ID=ubuntu\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID_LIKE="debian"\n',
        encoding="utf-8",
    )

    info = detect_distribution(os_release_path=os_release)

    assert info.distro_id == "ubuntu"
    assert info.name == "Ubuntu"
    assert info.version == "24.04"
    assert info.package_manager == "apt-get"


def test_detect_distribution_fedora_uses_dnf(tmp_path, monkeypatch):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'ID=fedora\nNAME="Fedora"\nVERSION_ID="40"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/dnf" if name == "dnf" else None
    )

    info = detect_distribution(os_release_path=os_release)

    assert info.distro_id == "fedora"
    assert info.package_manager == "dnf"


def test_format_install_command_for_common_managers():
    assert format_install_command(["bubblewrap"], "apt-get").startswith(
        "sudo apt-get update && sudo apt-get install -y bubblewrap"
    )
    assert format_install_command(["bubblewrap"], "dnf") == (
        "sudo dnf install -y bubblewrap"
    )
    assert format_install_command(["bubblewrap"], "pacman") == (
        "sudo pacman -S --noconfirm bubblewrap"
    )
    assert format_install_command(["bubblewrap"], "brew") == "brew install bubblewrap"
    assert format_install_command(["bubblewrap"], "pkg") == "pkg install bubblewrap"


def test_system_dependency_status_reports_tool_availability(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/prlimit" if name == "prlimit" else None
    )

    statuses = system_dependency_status()

    by_tool = {item["tool"]: item for item in statuses}
    assert by_tool["bwrap"]["available"] is False
    assert by_tool["prlimit"]["available"] is True
