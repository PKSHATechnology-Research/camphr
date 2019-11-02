"""
Parses Universal Dependencies data
"""

from __future__ import unicode_literals

import re
from collections import OrderedDict

DEFAULT_FIELDS = (
    "id",
    "form",
    "lemma",
    "upostag",
    "xpostag",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)

deps_pattern = r"\d+:[a-z][a-z_-]*(:[a-z][a-z_-]*)?"
MULTI_DEPS_PATTERN = re.compile(r"^{}(\|{})*$".format(deps_pattern, deps_pattern))


class ParseException(Exception):
    pass


def parse_token_and_metadata(data, fields=None):
    if not data:
        raise ParseException("Can't create TokenList, no data sent to constructor.")

    fields = fields or DEFAULT_FIELDS

    tokens = []
    metadata = OrderedDict()

    for line in data.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            var_name, var_value = parse_comment_line(line)
            if var_name:
                metadata[var_name] = var_value
        else:
            tokens.append(parse_line(line, fields=fields))

    return tokens, metadata


def parse_line(line, fields=DEFAULT_FIELDS, parse_feats=True):
    line = re.split(r"\t| {2,}", line)

    if len(line) == 1 and " " in line[0]:
        raise ParseException(
            "Invalid line format, line must contain either tabs or two spaces."
        )

    data = OrderedDict()

    for i, field in enumerate(fields):
        # Allow parsing CoNNL-U files with fewer columns
        if i >= len(line):
            break

        if field == "id":
            value = parse_id_value(line[i])
            data["multi_id"] = parse_multi_id_value(line[i])

        elif field == "xpostag":
            value = parse_nullable_value(line[i])

        elif field == "feats":
            if parse_feats:
                value = parse_dict_value(line[i])
            else:
                value = line[i]

        elif field == "head":
            value = parse_int_value(line[i])

        elif field == "deps":
            value = parse_paired_list_value(line[i])

        elif field == "misc":
            value = parse_dict_value(line[i])

        else:
            value = line[i]

        data[field] = value

    return data


def parse_comment_line(line):
    line = line.strip()
    if line[0] != "#":
        raise ParseException("Invalid comment format, comment must start with '#'")
    if "=" not in line:
        return None, None
    var_name, var_value = line[1:].split("=", 1)
    var_name = var_name.strip()
    var_value = var_value.strip()
    return var_name, var_value


def parse_int_value(value):
    if value == "_":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_id_value(value):
    # return value if "-" not in value else None
    return value if "-" not in value and "." not in value else None
    # TODO: handle special ids with "."


def parse_multi_id_value(value):
    if len(value.split("-")) == 2:
        return value
    return None


def parse_paired_list_value(value):
    if re.match(MULTI_DEPS_PATTERN, value):
        return [
            (part.split(":", 1)[1], parse_int_value(part.split(":", 1)[0]))
            for part in value.split("|")
        ]

    return parse_nullable_value(value)


def parse_dict_value(value):
    if "=" in value:
        return OrderedDict(
            [
                (part.split("=")[0], parse_nullable_value(part.split("=")[1]))
                for part in value.split("|")
                if len(part.split("=")) == 2
            ]
        )

    return parse_nullable_value(value)


def parse_nullable_value(value):
    if not value or value == "_":
        return None

    return value
