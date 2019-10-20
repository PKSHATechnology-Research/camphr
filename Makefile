download: download_bert download_xlnet # from bedore-rannd account

download_bert: 
	mkdir -p data
	aws s3 cp s3://bedoner/trf_models/bert/bert-ja-juman.tar.gz data/
	cd data && tar xzvf bert-ja-juman.tar.gz

download_xlnet: 
	mkdir -p data/xlnet
	aws s3 cp --recursive s3://bedoner/trf_models/xlnet/ data/xlnet

download_ner:
	mkdir -p ~/datasets
	aws s3 cp s3://bedoner/datasets/ ~/datasets --recursive
