import subprocess
import sys

import pytest


@pytest.mark.parametrize("cmd,err", [(["train", "--help"], False), ([], False)])
def test_cmd(cmd, err):
    res = subprocess.run(
        [sys.executable, "-m", "camphr.cli.__main__"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert (res.returncode != 0) == err, res.stderr.decode()
