# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import re

import fire
import jaconv
import pandas as pd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HIRAGANA_REGEX = re.compile(r"^[あ-ん]+$")
KATAKANA_REGEX = re.compile(r"[\u30A1-\u30F4]+")
NAMES_TO_BALANCE = ["高松", "山形", "宮城"]


def filter_names(name):
    """氏名フィルタリングロジック。短すぎる氏名等は辞書に載せない"""
    if (
        HIRAGANA_REGEX.fullmatch(name)
        or KATAKANA_REGEX.fullmatch(name)
        or len(name) <= 1
    ):
        return False
    return True


def _create_mecab_user_dic_df_from_jinmei_data(data: pd.DataFrame) -> pd.DataFrame:
    dic = pd.DataFrame()
    dic["表層形"] = data["表層形"]
    dic["左文脈ID"] = None
    dic["右文脈ID"] = None
    dic["コスト"] = None
    dic["品詞"] = "名詞"
    dic["品詞細分類1"] = "固有名詞"
    dic["品詞細分類2"] = "人名"
    dic["品詞細分類3"] = data["品詞細分類3"]
    dic["活用型"] = "*"
    dic["活用形"] = "*"
    dic["原形"] = data["表層形"]
    dic["読み"] = data[0].apply(lambda x: jaconv.hira2kata(x))
    dic["発音"] = dic["読み"]
    return dic


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


def load_jinmei_list():
    jinmei = pd.read_csv(
        f"{BASE_PATH}/data/JINMEI30.csv", encoding="utf-8", header=None
    )
    jinmei = jinmei.append(pd.DataFrame([{0: "なすか", 1: "為耶:名"}]))
    jinmei.reset_index(drop=True, inplace=True)
    jinmei["表層形"] = jinmei[1].apply(lambda x: x.split(":")[0])
    jinmei["品詞細分類3"] = jinmei[1].apply(lambda x: x.split(":")[1])
    return _create_mecab_user_dic_df_from_jinmei_data(jinmei)


def load_false_recognition_result():
    """通常のipadicで認識できなかった指名リストの読み込み"""

    def _create_mecab_user_dic_df_from_false_result(data: pd.DataFrame) -> pd.DataFrame:
        dic = pd.DataFrame()
        dic["表層形"] = data["表層形"]
        dic["左文脈ID"] = None
        dic["右文脈ID"] = None
        dic["コスト"] = None
        dic["品詞"] = "名詞"
        dic["品詞細分類1"] = "固有名詞"
        dic["品詞細分類2"] = "人名"
        dic["品詞細分類3"] = data["品詞細分類3"]
        dic["活用型"] = "*"
        dic["活用形"] = "*"
        dic["原形"] = data["表層形"]
        dic["読み"] = data["読み"]
        dic["発音"] = dic["読み"]
        return dic

    false_result = pd.read_csv(f"{BASE_PATH}/data/false_recognition_result.csv")
    # 姓・名を分割してカラム作成
    false_result["surname"] = false_result["名前"].apply(lambda x: x.split(" ")[0])
    false_result["firstname"] = false_result["名前"].apply(lambda x: x.split(" ")[1])
    false_result["surname_yomi"] = false_result["ふりがな"].apply(lambda x: x.split(" ")[0])
    false_result["firstname_yomi"] = false_result["ふりがな"].apply(
        lambda x: x.split(" ")[1]
    )

    # 姓ごと・名ごとにDFを作成して縦に結合
    result_df = false_result[["surname", "surname_yomi"]]
    result_df.columns = ["表層形", "読み"]
    result_df["品詞細分類3"] = "姓"
    firstname_df = false_result[["firstname", "firstname_yomi"]]
    firstname_df.columns = ["表層形", "読み"]
    firstname_df["品詞細分類3"] = "名"
    result_df = result_df.append(firstname_df).reset_index(drop=True)

    return _create_mecab_user_dic_df_from_false_result(result_df)


def balance_cost(dic: pd.DataFrame) -> pd.DataFrame:
    """コスト調整が必要な単語について調整する"""
    dic["コスト"] = dic.apply(
        lambda x: -200 if x["表層形"] in NAMES_TO_BALANCE else None, axis=1
    )
    return dic


def main():
    dictionary_df_list = [load_jinmei_list(), load_false_recognition_result()]
    mecab_dic = pd.concat(dictionary_df_list)
    # 辞書間で同じ表層形が入ることがあるので重複排除
    mecab_dic.drop_duplicates(subset=["表層形"], inplace=True)
    mecab_dic.reset_index(drop=True, inplace=True)
    # 高松は確実に人名として認識させたい、等のニーズに答えるために特定の単語のコストを調整
    mecab_dic = balance_cost(mecab_dic)

    file_path = f"{BASE_PATH}/user.csv"
    mecab_dic.to_csv(file_path, header=None, index=False)


if __name__ == "__main__":
    fire.Fire(main)
