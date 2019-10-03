import pandas as pd
import fire
import jaconv

requires = [
    "表層形",
    "左文脈ID",
    "右文脈ID",
    "コスト",
    "品詞",
    "品詞細分類1",
    "品詞細分類2",
    "品詞細分類3",
    "活用型",
    "活用形",
    "原形",
    "読み",
    "発音",
]


def main(input_file: str = "", output_file: str = "", check: bool = False):
    if check:
        return
    if not (input_file or output_file or check):
        raise ValueError

    df = pd.read_csv(
        input_file, sep="\t", comment=";", header=None, names=["読み", "表層形", "品詞細分類3"]
    )
    df = df.append([{"読み": "なすか", "表層形": "為耶", "品詞細分類3": "名"}])
    df["読み"] = df["読み"].apply(jaconv.hira2kata)
    df["発音"] = df["読み"]
    df["原形"] = df["表層形"]
    df["左文脈ID"] = None
    df["右文脈ID"] = None
    df["コスト"] = None
    df["品詞"] = "名詞"
    df["品詞細分類1"] = "固有名詞"
    df["品詞細分類2"] = "人名"
    df["活用型"] = "*"
    df["活用形"] = "*"
    NAMES_TO_BALANCE = {"高松", "山形", "宮城", "愛"}
    df["コスト"] = df.apply(
        lambda x: -200 if x["表層形"] in NAMES_TO_BALANCE else None, axis=1
    )
    df[requires].to_csv(output_file, header=False, index=False)


if __name__ == "__main__":
    fire.Fire(main)
