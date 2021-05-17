import subprocess
import sys

import pytest


@pytest.mark.parametrize("cmd,err", [(["train", "--help"], False), ([], True)])
def test_cmd(cmd, err):
    res = subprocess.run(
        [sys.executable, "-m", "camphr_cli.__main__"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert (res.returncode != 0) == err, res.stderr.decode()
