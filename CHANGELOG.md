# Changelog

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