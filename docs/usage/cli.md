# split_gold

Gold dataをsplitします．  
BERT等，入力長に制限があるモデルを使用する際に有効です．

```bash
$ python -m camphr.cli split_gold data.jsonl output.jsonl --sep 。
```
