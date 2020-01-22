import subprocess
import sys

import pytest


@pytest.mark.parametrize("cmd,err", [(["train", "--help"], False), ([], True)])
def test_cmd(cmd, err):
    res = subprocess.run([sys.executable, "-m", "camphr.cli.__main__"] + cmd)
    assert (res.returncode != 0) == err
