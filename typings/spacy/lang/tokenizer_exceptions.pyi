"""
This type stub file was generated by pyright.
"""

import re

URL_PATTERN = (
    r"^"
    r"(?:(?:[\w\+\-\.]{2,})://)?"
    r"(?:\S+(?::\S*)?@)?"
    r"(?:"
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    r"(?:(?:[a-z0-9\-]*)?[a-z0-9]+)"
    r"(?:\.(?:[a-z0-9])(?:[a-z0-9\-])*[a-z0-9])?"
    r"(?:\.(?:[a-z]{2,}))"
    r")"
    r"(?::\d{2,5})?"
    r"(?:[/?#]\S*)?"
    r"$".strip()
)
TOKEN_MATCH = re.compile(URL_PATTERN, re.UNICODE).match
BASE_EXCEPTIONS = {}
emoticons = set(
    r"""
:)
:-)
:))
:-))
:)))
:-)))
(:
(-:
=)
(=
")
:]
:-]
[:
[-:
:o)
(o:
:}
:-}
8)
8-)
(-8
;)
;-)
(;
(-;
:(
:-(
:((
:-((
:(((
:-(((
):
)-:
=(
>:(
:')
:'-)
:'(
:'-(
:/
:-/
=/
=|
:|
:-|
:1
:P
:-P
:p
:-p
:O
:-O
:o
:-o
:0
:-0
:()
>:o
:*
:-*
:3
:-3
=3
:>
:->
:X
:-X
:x
:-x
:D
:-D
;D
;-D
=D
xD
XD
xDD
XDD
8D
8-D

^_^
^__^
^___^
>.<
>.>
<.<
._.
;_;
-_-
-__-
v.v
V.V
v_v
V_V
o_o
o_O
O_o
O_O
0_o
o_0
0_0
o.O
O.o
O.O
o.o
0.0
o.0
0.o
@_@
<3
<33
<333
</3
(^_^)
(-_-)
(._.)
(>_<)
(*_*)
(¬_¬)
ಠ_ಠ
ಠ︵ಠ
(ಠ_ಠ)
¯\(ツ)/¯
(╯°□°）╯︵┻━┻
><(((*>
""".split()
)