"""Script to convert bccwj NER dataset to jsonl

Usage:

$ python bccwj2jsonl xml/ output/

# convert to irex

$ pythonn bccwj2jsonl xml/ output/ irex
"""
import json
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path
from typing import *
from typing import IO

import fire
import regex as re
from tqdm import tqdm

__dir__ = Path(__file__).parent
with open(__dir__/"ene2irexmap.json") as f:
    IREXMAP=json.load(f)

r = re.compile("<(?P<tag>[a-zA-Z-_]+)>(?P<body>.*?)</[a-zA-Z-_]+>")
rtag = re.compile("</?[a-zA-Z-_]+>")

Entry = namedtuple("Entry", ["text", "label"])


def convert(xml_string: str, mapping: Optional[Dict[str, str]] = None) -> Entry:
    offset = 0
    spans = []
    for t in r.finditer(xml_string):
        i = t.start()
        tag, body = t.groups()
        start = i - offset
        end = start + len(body)
        offset += 2 * len(tag) + 5
        if mapping:
            tag = IREXMAP.get(tag, "")
        if tag:
            spans.append((start, end, tag))
    notag = rtag.sub("", xml_string)
    return Entry(notag, {"entities": spans})


def check_conversion(item: Entry, xml_text, is_tag_removed=False) -> bool:
    text, label = item
    entities: List[Tuple[int, int, str]] = label["entities"]
    if not is_tag_removed:
        for (i, j, _), item in zip(entities, r.finditer(xml_text)):
            if text[i:j] != item.groups()[1]:
                return False

    try:
        a = ET.fromstring(f"<a>{xml_text}</a>")
    except:
        return False
    expected = ET.tostring(a, method="text", encoding="utf-8").decode()
    return expected == text


def proc(
    xml: IO[str], output: IO[str], tag_mapping=""
) -> Tuple[int, List[Any]]:
    """Convert xml to jsonl."""
    count = 0
    flag = False
    failed=[]
    for i, line in enumerate(xml):
        line = line.strip()
        if not line:
            continue
        if not flag:
            if line == "<TEXT>":
                flag = True
            continue
        if line == "</TEXT>":
            break
        for sent in line.split("。"):
            sent += "。"
            if tag_mapping == "irex":
                ent = convert(sent, mapping=IREXMAP)
            else:
                ent = convert(sent)
            if not check_conversion(ent, sent, is_tag_removed=tag_mapping != ""):
                failed.append(i)
                continue
            output.write(json.dumps(ent, ensure_ascii=False) + "\n")
            count += 1
    return count, failed


def main(
    xml_dir: Union[str, Path],
    jsonl_dir: Union[str, Path],
    tag_mapping="",
    failed_log="log.txt",
):
    """Convert all xml files in xml_dir to jsonl, and save them in jsonl_dir."""
    xml_dir = Path(xml_dir).absolute()
    jsonl_dir = Path(jsonl_dir).absolute()
    assert xml_dir.exists()
    fcount = 0
    itemcount = 0
    with open(failed_log, "w") as fw:
        for xml in tqdm(xml_dir.glob("**/*.xml")):
            outputpath = jsonl_dir / (str(xml).lstrip(str(xml_dir)) + ".jsonl")
            outputpath.parent.mkdir(exist_ok=True, parents=True)
            failed = []

            with open(xml) as f, outputpath.open("w") as fj:
                c, failed = proc(f, fj, tag_mapping=tag_mapping)
            fw.write("\n".join(map(lambda x: str(xml)+f": {x}", failed)))
            itemcount += c
            fcount += 1
    print(f"{fcount} files, {itemcount} items parsed.")


if __name__ == "__main__":
    fire.Fire(main)
