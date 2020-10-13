# Changelog

## 0.7.0 (21/08/2020)
- Separeted `udify` and `elmo` pipelines to [camphr_allennlp](https://github.com/PKSHATechnology-Research/camphr-allennlp) - now camphr doesn't depend on allennlp.
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