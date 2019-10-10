download: # from bedore-rannd account
	mkdir -p data
	aws s3 cp s3://bedoner/trf_models/bert/bert-ja-juman.tar.gz data/
	cd data && tar xzvf bert-ja-juman.tar.gz

download_ner:
	mkdir -p ~/datasets
	aws s3 cp s3://bedoner/datasets/ ~/datasets --recursive
