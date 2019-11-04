download: download_bert download_xlnet # from bedore-rannd account

download_bert: 
	mkdir -p data/bert-ja-juman
	aws s3 cp s3://bedoner/trf_models/bert/bert-ja-juman.tar.gz data/
	cd data && tar xzvf bert-ja-juman.tar.gz -C bert-ja-juman/

download_xlnet: 
	mkdir -p data/xlnet
	aws s3 cp --recursive s3://bedoner/trf_models/xlnet/ data/xlnet

download_udify:
	mkdir -p data/udify
	aws s3 cp s3://bedoner/models/udify.tar.gz data/
	cd data && tar xzvf udify.tar.gz -C udify/

download_ner:
	mkdir -p ~/datasets
	aws s3 cp s3://bedoner/datasets/ ~/datasets --recursive
