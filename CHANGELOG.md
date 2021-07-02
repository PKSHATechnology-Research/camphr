# Changelog

## 0.7.0 (21/08/2020)
- [**dependencies**] Bump pyknp from 0.4.4 to 0.4.5 [#80](https://github.com/PKSHATechnology-Research/camphr/pull/80)
- [**dependencies**] Bump spacy from 2.2.4 to 2.3.2 [#81](https://github.com/PKSHATechnology-Research/camphr/pull/81)
- [**dependencies**] Bump torch from 1.5.1 to 1.6.0 [#82](https://github.com/PKSHATechnology-Research/camphr/pull/82)
- [**closed**] move allennlp to camphr_allennlp [#79](https://github.com/PKSHATechnology-Research/camphr/pull/79)
- [**dependencies**] Bump hypothesis from 5.23.11 to 5.23.12 [#73](https://github.com/PKSHATechnology-Research/camphr/pull/73)
- [**dependencies**] Bump pytest from 5.4.3 to 6.0.1 [#66](https://github.com/PKSHATechnology-Research/camphr/pull/66)
- [**closed**] fix get_doc_char_span and covering span [#78](https://github.com/PKSHATechnology-Research/camphr/pull/78)
- [**closed**] fix index error [#77](https://github.com/PKSHATechnology-Research/camphr/pull/77)
- [**closed**] add lemma search to PatternSearch [#76](https://github.com/PKSHATechnology-Research/camphr/pull/76)
- [**dependencies**] Bump pytextspan from 0.2.2 to 0.3.0 [#74](https://github.com/PKSHATechnology-Research/camphr/pull/74)
- [**closed**] improve beamsearch performance for k ==1 [#75](https://github.com/PKSHATechnology-Research/camphr/pull/75)
- [**closed**] use pyknp [#71](https://github.com/PKSHATechnology-Research/camphr/pull/71)
- [**closed**] add normalizer to pattern search [#70](https://github.com/PKSHATechnology-Research/camphr/pull/70)
- [**closed**] Pattern searcher becomes able to search with lemma and lower [#65](https://github.com/PKSHATechnology-Research/camphr/pull/65)
- [**closed**] 形容詞接頭辞 into PART [#63](https://github.com/PKSHATechnology-Research/camphr/pull/63)
- [**closed**] fix deps [#62](https://github.com/PKSHATechnology-Research/camphr/pull/62)

---

## 0.6.0 (09/07/2020)
- [**dependencies**] Bump scikit-learn from 0.22.2.post1 to 0.23.1 [#61](https://github.com/PKSHATechnology-Research/camphr/pull/61)
- [**dependencies**] Bump pytest from 5.3.2 to 5.4.3 [#60](https://github.com/PKSHATechnology-Research/camphr/pull/60)
- [**closed**] support allennlp v1 [#59](https://github.com/PKSHATechnology-Research/camphr/pull/59)
- [**closed**] Improvement for サ変 of KNP [#56](https://github.com/PKSHATechnology-Research/camphr/pull/56)
- [**closed**] refactor [#55](https://github.com/PKSHATechnology-Research/camphr/pull/55)

---

## 0.5.22 (24/04/2020)
- [**bug**] fix transformers eval batchsize failure [#50](https://github.com/PKSHATechnology-Research/camphr/pull/50)

---

## 0.5.21 (22/04/2020)
- [**bug**] Proper treatment of PUNCTs for KNP [#48](https://github.com/PKSHATechnology-Research/camphr/pull/48)

---

## 0.5.20 (14/04/2020)
- [**enhancement**] dependency improvement for KNP [#47](https://github.com/PKSHATechnology-Research/camphr/pull/47)

Thanks for contributing, @KoichiYasuoka!
---

## 0.5.19 (13/04/2020)
- [**enhancement**] update transformers dependency [#46](https://github.com/PKSHATechnology-Research/camphr/pull/46)
- [**CI**] Skip slow ci if unnecessary [#45](https://github.com/PKSHATechnology-Research/camphr/pull/45)
- [**enhancement**] Refactor/knp dependency parser [#44](https://github.com/PKSHATechnology-Research/camphr/pull/44)
- [**enhancement**] Tentative dependencies for KNP [#43](https://github.com/PKSHATechnology-Research/camphr/pull/43)

Thanks for contributing, @KoichiYasuoka!
---

## 0.5.18 (10/04/2020)
- [**enhancement**] juman TAG_MAP tentative support [#41](https://github.com/PKSHATechnology-Research/camphr/pull/41)
- [**bug**] Fix misuse `Vocab()` in Language instantiation [#42](https://github.com/PKSHATechnology-Research/camphr/pull/42)

---

## 0.5.17 (09/04/2020)
- [**enhancement**] Revert sentencepiece lang from v0.4 [#40](https://github.com/PKSHATechnology-Research/camphr/pull/40)

---

## 0.5.16 (09/04/2020)
- [**enhancement**] add `functools.lru_cache` to knp extensions. [#39](https://github.com/PKSHATechnology-Research/camphr/pull/39)

---

## 0.5.15 (08/04/2020)
*No changelog for this release.*

---

## 0.5.15.dev0 (08/04/2020)


---

## 0.5.14 (08/04/2020)
- [**enhancement**] tag and bunsetsu can be directly got from token [#38](https://github.com/PKSHATechnology-Research/camphr/pull/38)
- [**enhancement**] Feature/knp para noun chunks [#37](https://github.com/PKSHATechnology-Research/camphr/pull/37)
- [**bug**] fix noun chunker for para phrase [#36](https://github.com/PKSHATechnology-Research/camphr/pull/36)
- [**enhancement**][**refactor**] Refactor/knp noun chunker [#35](https://github.com/PKSHATechnology-Research/camphr/pull/35)

---

## 0.5.13 (06/04/2020)
# Bug fix

- Separate parallel clause in noun chunks into two or more chunks #34
---

## 0.5.12 (06/04/2020)
# New Features

- Support knp noun chunker and knp dependency parser #33
---

## 0.5.11 (27/03/2020)
# New features

- It is now possible to retrieve KNP result from `spacy.doc` (#31)
---

## 0.5.10 (18/03/2020)
Removed the version restriction `python<3.8`. This will allow users to install camphr with `python3.8`, but macos users will fail. see (#29) for details.
---

## 0.5.9 (03/03/2020)
# Improvements

- juman and knp now accepts longer text (#23)
---

## 0.5.8 (03/03/2020)
# Bug fix

- fix transformers requirements (#24)
---

## 0.5.5 (21/02/2020)
# New features

- Multi labels textcat pipe for transformers (#14)
---

## 0.5.7 (21/02/2020)
# bug fix

- fix `camphr.utils.get_requirements_line`
---

## 0.5.3 (17/02/2020)
# New Features

- Computing val loss in TorchLanguage.evaluate` #13