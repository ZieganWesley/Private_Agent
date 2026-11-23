import platform

from platform_support import (
    PlatformInfo,
    detect_platform,
    is_termux,
    is_wsl,
    platform_name,
)


def test_linux_without_wsl_or_termux(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    proc = tmp_path / "version"
    proc.write_text("Linux version 6.8.0-generic\n", encoding="utf-8")

    info = detect_platform(proc_version_path=proc, environ={})

    assert info.is_linux is True
    assert info.is_macos is False
    assert info.is_windows is False
    assert info.is_wsl is False
    assert info.is_termux is False
    assert info.name == "Linux"
    assert "command-isolation" in info.available_features
    assert isinstance(info, PlatformInfo)


def test_wsl_detected_from_proc_version(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    proc = tmp_path / "version"
    proc.write_text(
        "Linux version 5.15.167.1-microsoft-standard-WSL2\n", encoding="utf-8"
    )

    assert is_wsl(proc_version_path=proc, environ={}) is True
    info = detect_platform(proc_version_path=proc, environ={})
    assert info.is_wsl is True
    assert info.name == "WSL"


def test_wsl_detected_from_environment(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    assert (
        is_wsl(proc_version_path="/no/such/file", environ={"WSL_DISTRO_NAME": "Ubuntu"})
        is True
    )


def test_termux_detected_from_environment(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")

    info = detect_platform(
        proc_version_path="/no/such/file",
        environ={
            "TERMUX_VERSION": "0.118.0",
            "PREFIX": "/data/data/com.termux/files/usr",
        },
    )

    assert info.is_termux is True
    assert info.name == "Termux"
    assert info.available_features == ("termux",)


def test_termux_detected_from_path(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")

    assert (
        is_termux(
            environ={},
            path_exists=lambda path: path == "/data/data/com.termux",
        )
        is True
    )


def test_macos_detection(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    info = detect_platform(proc_version_path="/no/such/file", environ={})

    assert info.is_macos is True
    assert info.is_linux is False
    assert info.name == "macOS"
    assert "seatbelt" in info.available_features


def test_windows_detection(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr("sys.platform", "win32", raising=False)

    info = detect_platform(proc_version_path="/no/such/file", environ={})

    assert info.is_windows is True
    assert info.is_linux is False
    assert info.name == "Windows"
    assert info.available_features == ()


def test_windows_mount_path_detection():
    from platform_support import is_windows_mount_path

    assert is_windows_mount_path("/mnt/c/projects/app") is True
    assert is_windows_mount_path("/mnt/d") is True
    assert is_windows_mount_path("/home/user/project") is False
    assert is_windows_mount_path("/mnt/wsl/foo") is False
    assert is_windows_mount_path("/mnt") is False


def test_platform_name_helper(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    assert (
        platform_name(detect_platform(proc_version_path="/no/such/file", environ={}))
        == "Linux"
    )
