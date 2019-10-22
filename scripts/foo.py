import json

with open("ene/foo.a") as f:
    for line in f:
        d = json.loads(line)
        ents = d[1]["entities"]
        if len(ents):
            print(line, end="")
