from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

pretrained_model_name_or_path = "./tests/fixtures/ner_bert"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)
