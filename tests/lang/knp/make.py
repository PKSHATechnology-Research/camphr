from pyknp import Juman
from test_tokenizer import TOKENIZER_TESTS
import subprocess

j = Juman()
for t in TOKENIZER_TESTS:
    ml = j.analysis(t[0])
    surfl, posl = [], []
    for m in ml:
        surf, pos = m.midasi, m.hinsi + "/" + m.bunrui
        surfl.append(surf)
        posl.append(pos)
    print((t[0], posl), ",")

