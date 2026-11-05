import pytest

from private_agent.sandbox import (
    evaluate_shell_command,
    requires_shell_interpreter,
    shell_command_programs,
)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "python -m pytest",
        "echo /usr",
        "echo sudo",
        "ls /usr/bin",
        "cat /etc/os-release",
        "grep -r TODO src",
        "rm build/output.txt",
        "rm -f build/output.txt",
        "chmod +x script.sh",
        "chmod 755 script.sh",
        "cp /usr/share/dict/words words.txt",
        "find . -name '*.pyc'",
        "pip install --user requests",
        "ls > /dev/null 2>&1",
        "echo done >/dev/null",
        "git status && git diff",
        "resume_parser --input data.csv",
        'echo "rm -rf /"',
        "git commit -m 'sudo make me a sandwich'",
        "",
        "   ",
    ],
)
def test_safe_commands_are_not_flagged(command):
    assert evaluate_shell_command(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -r build",
        "rm -fr build",
        "rm -Rf build",
        "rm -f -r build",
        "rm --recursive build",
        "/bin/rm -rf build",
    ],
)
def test_recursive_rm_is_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "sudo apt update",
        "su",
        "su - root",
        "doas reboot",
        "pkexec bash",
        "/usr/bin/sudo ls",
    ],
)
def test_privilege_escalation_is_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "chmod 777 script.py",
        "chmod 0777 script.py",
        "chmod -R 755 src",
        "chmod a+rwx script.py",
        "chown user file.txt",
        "chgrp staff file.txt",
    ],
)
def test_permission_changes_are_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "mkfs.ext4 /dev/sda1",
        "mkfs -t ext4 /dev/sda1",
        "dd if=/dev/zero of=disk.img",
        "shred secrets.txt",
        "wipefs -a /dev/sdb",
        ":(){ :|:& };:",
        ":(){:|:&};:",
    ],
)
def test_destructive_programs_are_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "echo data > /dev/sda",
        "echo data >/dev/sda",
        "echo 'nameserver 1.1.1.1' >> /etc/resolv.conf",
        "echo 1 | tee /proc/sys/vm/drop_caches",
        "touch /etc/motd",
        "mv config /etc/app.conf",
        "cp app.conf /etc/",
        "rm //etc/hosts",
    ],
)
def test_writes_to_system_paths_are_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "find . -delete",
        "find . -name '*.tmp' -exec rm {} \\;",
        "find . -type f -execdir /bin/rm {} +",
    ],
)
def test_find_deletions_are_flagged(command):
    assert evaluate_shell_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "ls && rm -rf build",
        "echo hi; sudo reboot",
        "false || chown user file",
        "cat files.txt | xargs rm -rf",
        "(cd build && rm -rf dist)",
        "env DEBUG=1 sudo ls",
        "DEBUG=1 rm -rf build",
        "nohup rm -rf build &",
        "sh -c 'rm -rf build'",
        'bash -c "sudo apt update"',
    ],
)
def test_risky_commands_hidden_in_compound_commands_are_flagged(command):
    assert evaluate_shell_command(command) is True


def test_wrapped_shell_with_safe_script_is_not_flagged():
    assert evaluate_shell_command("bash -c 'echo /usr'") is False


def test_unbalanced_quotes_still_flag_risky_programs():
    assert evaluate_shell_command("sudo echo 'unterminated") is True
    assert evaluate_shell_command("echo 'unterminated") is False


@pytest.mark.parametrize(
    "command",
    [
        "cd src && python --version",
        "cd /home/user/project && source venv/bin/activate && python --version",
        "pytest -q | tail -5",
        "echo one; echo two",
        "false || echo recovered",
        "make > build.log 2>&1",
        "wc -l < out.txt",
        "cat <<< hello",
        "cd src",
        "source venv/bin/activate",
        "export DEBUG=1",
        "echo $HOME",
        "echo `date`",
        "ls *.py",
        "ls ~",
        "(cd build && make)",
    ],
)
def test_shell_syntax_requires_interpreter(command):
    assert requires_shell_interpreter(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "python -m pytest -q",
        "git status",
        'grep -n "TODO" src/app.py',
        'echo "a && b"',
        "echo 'pipe | inside quotes'",
    ],
)
def test_plain_commands_do_not_require_interpreter(command):
    assert requires_shell_interpreter(command) is False


def test_shell_command_programs_lists_every_simple_command():
    assert shell_command_programs(
        "cd src && DEBUG=1 python app.py | tee log.txt; /usr/bin/git status"
    ) == ["cd", "python", "tee", "git"]
