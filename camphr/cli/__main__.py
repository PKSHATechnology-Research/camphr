import importlib
import sys

from wasabi import msg  # type: ignore

commands = ["train"]
MSG_AVAILABLE_COMMANDS = f"Available commands: {', '.join(commands)}"


def main():
    if len(sys.argv) == 1:
        msg.info("usage: camphr <command>", MSG_AVAILABLE_COMMANDS, exits=1)
    cmd = sys.argv.pop(1)
    if cmd not in commands:
        msg.fail(f"unknown command {cmd}.", MSG_AVAILABLE_COMMANDS, exits=1)
    else:
        run(cmd)


def run(cmd: str):
    # Lazy load `cmd` because Hydra's instantiation takes few seconds.
    # All `cmd` must exist in the form of `camphr.cli.cmd.main`.
    cmd = cmd.replace("-", "_")
    m = importlib.import_module(f"camphr.cli.{cmd}")
    getattr(m, "main")()


if __name__ == "__main__":
    main()
