import json
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path
from typing import *

import fire
import regex as re
from tqdm import tqdm

IREXMAP = {
    #     'NAME':  # PERSON, LOCATION, ORGANIZATION, ARTIFACT
    #     {
    #         'Name_Other': {},
    "Person": "PERSON",
    #         'God': {},
    #         'Organization':  # 'ORGANIZATION'
    #         {
    "Organization_Other": "ORGANIZATION",
    "International_Organization": "ORGANIZATION",
    "Show_Organization": "ORGANIZATION",
    "Family": "ORGANIZATION",
    #             'Ethnic_Group':
    #             {
    "Ethnic_Group_Other": "ORGANIZATION",
    "Nationality": "ORGANIZATION",
    #             },
    #             'Sports_Organization':
    #             {
    "Sports_Organization_Other": "ORGANIZATION",
    "Pro_Sports_Organization": "ORGANIZATION",
    "Sports_League": "ORGANIZATION",
    #             },
    #             'Corporation':
    #             {
    "Corporation_Other": "ORGANIZATION",
    "Company": "ORGANIZATION",
    "Company_Group": "ORGANIZATION",
    #             },
    #             'Political_Organization':
    #             {
    "Political_Organization_Other": "ORGANIZATION",
    "Government": "ORGANIZATION",
    "Political_Party": "ORGANIZATION",
    "Cabinet": "ORGANIZATION",
    "Military": "ORGANIZATION",
    #             }
    #         },
    #         'Location':  # LOCATION
    #         {
    "Location_Other": "LOCATION",
    "Spa": "LOCATION",
    #             'GPE':
    #             {
    "GPE_Other": "LOCATION",
    "City": "LOCATION",
    "County": "LOCATION",
    "Province": "LOCATION",
    "Country": "LOCATION",
    #             },
    #             'Region':
    #             {
    "Region_Other": "LOCATION",
    "Continental_Region": "LOCATION",
    "Domestic_Region": "LOCATION",
    #             },
    #             'Geological_Region':
    #             {
    "Geological_Region_Other": "LOCATION",
    "Mountain": "LOCATION",
    "Island": "LOCATION",
    "River": "LOCATION",
    "Lake": "LOCATION",
    "Sea": "LOCATION",
    "Bay": "LOCATION",
    #             },
    #             'Astral_Body':  # - LOCATION
    #             {
    #                 'Astral_Body_Other': {},
    #                 'Star': {},
    #                 'Planet': {},
    #                 'Constellation': {},
    #             },
    #             'Address':
    #             {
    "Address_Other": "LOCATION",
    "Postal_Address": "LOCATION",
    #                 'Phone_Number': {},  # - LOCATION
    #                 'Email': {},  # - LOCATION
    #                 'URL': {}  # - LOCATION
    #             },
    #         },
    #         'Facility':  # 'LOCATION'
    #         {
    "Facility_Other": "LOCATION",
    "Facility_Part": "LOCATION",
    #             'Archaeological_Place':
    #             {
    "Archaeological_Place_Other": "LOCATION",
    "Tumulus": "LOCATION",
    #             },
    #             'GOE':
    #             {
    "GOE_Other": "LOCATION",
    "Public_Institution": "LOCATION",
    "School": "LOCATION",
    "Research_Institute": "LOCATION",
    "Market": "LOCATION",
    "Park": "LOCATION",
    "Sports_Facility": "LOCATION",
    "Museum": "LOCATION",
    "Zoo": "LOCATION",
    "Amusement_Park": "LOCATION",
    "Theater": "LOCATION",
    "Worship_Place": "LOCATION",
    "Car_Stop": "LOCATION",
    "Station": "LOCATION",
    "Airport": "LOCATION",
    "Port": "LOCATION",
    #             },
    #             'Line':  # - LOCATION
    #             {
    #                 'Line_Other': {},
    #                 'Railroad': {},
    #                 'Road': {},
    #                 'Canal': {},
    #                 'Water_Route': {},
    #                 'Tunnel': {},
    #                 'Bridge': {}
    #             }
    #         },
    #         'Product':   # 'ARTIFACT'
    #         {
    "Product_Other": "ARTIFACT",
    "Material": "ARTIFACT",
    "Clothing": "ARTIFACT",
    "Money_Form": "ARTIFACT",
    "Drug": "ARTIFACT",
    "Weapon": "ARTIFACT",
    "Stock": "ARTIFACT",
    "Award": "ARTIFACT",
    "Decoration": "ARTIFACT",
    "Offence": "ARTIFACT",
    "Service": "ARTIFACT",
    "Class": "ARTIFACT",
    "Character": "ARTIFACT",
    "ID_Number": "ARTIFACT",
    #             'Vehicle':
    #             {
    "Vehicle_Other": "ARTIFACT",
    "Car": "ARTIFACT",
    "Train": "ARTIFACT",
    "Aircraft": "ARTIFACT",
    "Spaceship": "ARTIFACT",
    "Ship": "ARTIFACT",
    #             },
    #             'Food':
    #             {
    "Food_Other": "ARTIFACT",
    "Dish": "ARTIFACT",
    #             },
    #             'Art':
    #             {
    "Art_Other": "ARTIFACT",
    "Picture": "ARTIFACT",
    "Broadcast_Program": "ARTIFACT",
    "Movie": "ARTIFACT",
    "Show": "ARTIFACT",
    "Music": "ARTIFACT",
    "Book": "ARTIFACT",
    #             },
    #             'Printing':
    #             {
    "Printing_Other": "ARTIFACT",
    "Newspaper": "ARTIFACT",
    "Magazine": "ARTIFACT",
    #             },
    #             'Doctrine_Method':
    #             {
    "Doctrine_Method_Other": "ARTIFACT",
    "Culture": "ARTIFACT",
    "Religion": "ARTIFACT",
    "Academic": "ARTIFACT",
    "Sport": "ARTIFACT",
    "Style": "ARTIFACT",
    "Movement": "ARTIFACT",
    "Theory": "ARTIFACT",
    "Plan": "ARTIFACT",
    #             },
    #             'Rule':
    #             {
    "Rule_Other": "ARTIFACT",
    "Treaty": "ARTIFACT",
    "Law": "ARTIFACT",
    #             },
    #             'Title':
    #             {
    "Title_Other": "ARTIFACT",
    "Position_Vocation": "ARTIFACT",
    #             },
    #             'Language':
    #             {
    "Language_Other": "ARTIFACT",
    "National_Language": "ARTIFACT",
    #             },
    #             'Unit':
    #             {
    "Unit_Other": "ARTIFACT",
    "Currency": "ARTIFACT",
    #             }
    #         },
    #         'Event':
    #         {
    #             'Event_Other': {},
    #             'Occasion':
    #             {
    #                 'Occasion_Other': {},
    #                 'Religious_Festival': {},
    #                 'Game': {},
    #                 'Conference': {}
    #             },
    #             'Incident':
    #             {
    #                 'Incident_Other': {},
    #                 'War': {}
    #             },
    #             'Natural_Phenomenon':
    #             {
    #                 'Natural_Phenomenon_Other': {},
    #                 'Natural_Disaster': {},
    #                 'Earthquake': {}
    #             }
    #         },
    #         'Natural_Object':
    #         {
    #             'Natural_Object_Other': {},
    #             'Element': {},
    #             'Compound': {},
    #             'Mineral': {},
    #             'Living_Thing':
    #             {
    #                 'Living_Thing_Other': {},
    #                 'Fungus': {},
    #                 'Mollusc_Arthropod': {},
    #                 'Insect': {},
    #                 'Fish': {},
    #                 'Amphibia': {},
    #                 'Reptile': {},
    #                 'Bird': {},
    #                 'Mammal': {},
    #                 'Flora': {}
    #             },
    #             'Living_Thing_Part':
    #             {
    #                 'Living_Thing_Part_Other': {},
    #                 'Animal_Part': {},
    #                 'Flora_Part': {}
    #             }
    #         },
    #         'Disease':
    #         {
    #             'Disease_Other': {},
    #             'Animal_Disease': {}
    #         },
    #         'Color':
    #         {
    #             'Color_Other': {},
    #             'Nature_Color': {}
    #         }
    #     },
    #     'Time_Top':
    #     {
    #         'Time_Top_Other': {},
    #         'Timex':
    #         {
    "Timex_Other": "TIME",  # OK?
    "Time": "TIME",
    "Date": "DATE",
    "Day_Of_Week": "DATE",
    "Era": "DATE",
    #         },
    #         'Periodx':
    #         {
    #             'Periodx_Other': {},
    #             'Period_Time': {},
    #             'Period_Day': {},
    #             'Period_Week': {},
    #             'Period_Month': {},
    #             'Period_Year': {}
    #         }
    #     },
    #     'Numex':  # MONEY, PERCENT
    #     {
    "Money": "MONEY",
    #         'Stock_Index': {},
    #         'Point': {},
    "Percent": "PERCENT",
    #         'Multiplication': {},
    #         'Frequency': {},
    #         'Age': {},
    #         'School_Age': {},
    #         'Ordinal_Number': {},
    #         'Rank': {},
    #         'Latitude_Longtitude': {},
    #         'Measurement':
    #         {
    #             'Measurement_Other': {},
    #             'Physical_Extent': {},
    #             'Space': {},
    #             'Volume': {},
    #             'Weight': {},
    #             'Speed': {},
    #             'Intensity': {},
    #             'Temperature': {},
    #             'Calorie': {},
    #             'Seismic_Intensity': {},
    #             'Seismic_Magnitude': {}
    #         },
    #         'Countx':
    #         {
    #             'Countx_Other': {},
    #             'N_Person': {},
    #             'N_Organization': {},
    #             'N_Location':
    #             {
    #                 'N_Location_Other': {},
    #                 'N_Country': {}
    #             },
    #             'N_Facility': {},
    #             'N_Product': {},
    #             'N_Event': {},
    #             'N_Natural_Object':
    #             {
    #                 'N_Natural_Object_Other': {},
    #                 'N_Animal': {},
    #                 'N_Flora': {}
    #             }
    #         }
    #     }
}

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


def preprocess(text: str) -> str:
    return text.replace("\u3000", "-")


def proc(
    xml_file: Union[Path, str], output_jsonl: Union[Path, str] = "", tag_mapping=""
) -> Tuple[int, List[Any]]:
    xml_file = Path(xml_file)
    failed = []
    if not output_jsonl:
        output_jsonl = xml_file.parent / (xml_file.stem + ".jsonl")
    else:
        output_jsonl = Path(output_jsonl)
    count = 0
    with xml_file.open() as f, output_jsonl.open("w") as fw:
        flag = False
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if not flag:
                if line == "<TEXT>":
                    flag = True
                continue
            if line == "</TEXT>":
                break
            line = preprocess(line)
            for sent in line.split("。"):
                sent += "。"
                if tag_mapping == "irex":
                    ent = convert(sent, mapping=IREXMAP)
                else:
                    ent = convert(sent)
                if not check_conversion(ent, sent, is_tag_removed=tag_mapping != ""):
                    failed.append(f"{xml_file} {i} failed")
                    continue
                fw.write(json.dumps(ent, ensure_ascii=False) + "\n")
                count += 1
    return count, failed


def main(
    xml_dir: Union[str, Path],
    jsonl_dir: Union[str, Path],
    tag_mapping="",
    failed_log="log.txt",
):
    xml_dir = Path(xml_dir)
    jsonl_dir = Path(jsonl_dir)
    assert xml_dir.exists()
    fcount = 0
    itemcount = 0
    with open(failed_log, "w") as fw:
        for xml in tqdm(xml_dir.glob("**/*.xml")):
            outputpath = jsonl_dir / (str(xml).lstrip(str(xml_dir)) + ".jsonl")
            outputpath.parent.mkdir(exist_ok=True, parents=True)
            c, failed = proc(xml, outputpath, tag_mapping=tag_mapping)
            fw.write("\n".join(failed))
            itemcount += c
            fcount += 1
    print(f"{fcount} files, {itemcount} items parsed.")


if __name__ == "__main__":
    fire.Fire()
