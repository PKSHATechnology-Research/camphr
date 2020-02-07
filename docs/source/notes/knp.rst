.. include:: ../replaces.txt

KNP
===

`KNP <http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP>`_ は日本語の格解析モデルです．
Camphrは，knpの解析結果をspaCyで扱うための機能を提供しています．

:note: KNP is a Japanese language analysis model, and can be used only in Japanese. Therefore, this page is written in Japanese.

Installation
------------

`このページに従い <http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP>`_, 事前にjumanppとknpをインストールしてください．

そのあと，以下のようにcamphrをインストールします．

.. parsed-literal::

    |install-camphr-with-knp|

Usage
-----

.. code:: python3

    >>> import camphr
    >>> nlp = camphr.load("knp")

以下のようにすると，KNPの解析結果がdocに保存されます.

.. code:: python3

    >>> doc = nlp("太郎はリンゴとみかんを食べながら富士山へ行った。")

形態素オブジェクトは以下のように取得することができます.

`pyknp <https://github.com/ku-nlp/pyknp>`_ の :code:`Morpheme` が入っています.

.. code:: python3

    >>> token = doc[0]
    >>> token._.knp_morph_element
    <pyknp.juman.morpheme.Morpheme at 0x10eeed810>


pyknpの :code:`BList` を取得するには以下のようにします．

.. code:: python3

    >>> sent = list(doc.sents)[0]
    >>> sent._.knp_bunsetsu_list_
    <pyknp.knp.blist.BList at 0x13e0cae50>

KNPの文節，基本句はspaCyの :code:`Span` として格納されています．

例えば全ての文節を取得したい場合，以下のようにします．

.. code:: python3

    >>> sent._.knp_bunsetsu_spans
    [太郎は, リンゴと, みかんを, 食べながら, 富士山へ, 行った。]

基本句についても同様に取得できます．

.. code:: python3

    >>> sent._.knp_tag_spans
    [太郎は, リンゴと, みかんを, 食べながら, 富士, 山へ, 行った。]

係り受け関係は以下のようにして取得できます．

.. code:: python3

    >>> tag = sent._.knp_tag_spans[3] # 基本句
    >>> print(tag.text)
    食べながら
    >>> print(tag._.knp_tag_children) # 係り受け先のリスト
    [みかんを]
    >>> print(tag._.knp_tag_parent) # 係り受け元
    行った。


knpの解析結果 (:code:`features`) は :code:`._.knp_tag_element.features` に格納されています．

例えば基本句のfeaturesは以下のようにして取得できます．

.. code:: python3

    >>> tag._.knp_tag_element.features
    {'助詞': True,
     '用言': '動',
     '係': '連用',
     'レベル': 'A',
     '区切': '0-5',
     'ID': '〜ながら',
     '連用要素': True,
     '連用節': True,
     '動態述語': True,
     '正規化代表表記': '食べる/たべる',
     '用言代表表記': '食べる/たべる',
     '格関係2': 'ヲ:みかん',
     '格解析結果': '食べる/たべる:動1:ガ/U/-/-/-/-;ヲ/C/リンゴ/1/0/1;ヲ/C/みかん/2/0/1;ニ/U/-/-/-/-;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;ヨリ/U/-/-/-/-;マデ/U/-/-/-/-;ヘ/U/-/-/-/-;時間/U/-/-/-/-;外の関係/U/-/-/-/-;修飾/U/-/-/-/-;ノ/U/-/-/-/-;トスル/U/-/-/-/-;ニヨル/U/-/-/-/-;ニツク/U/-/-/-/-;トイウ/U/-/-/-/-;ニナラブ/U/-/-/-/-;ニツヅク/U/-/-/-/-;ニアワセル/U/-/-/-/-'}


NERの解析結果は，:code:`doc.ents` に格納されています.

.. code:: python3

    >>> doc.ents
    (太郎, 富士山)
