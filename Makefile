download: download_bert download_xlnet download_udify download_elmo # from bedore-rannd account

download_bert: 
	mkdir -p data/bert-ja-juman
	aws s3 cp s3://camphr/trf_models/bert/bert-ja-juman.tar.gz data/
	cd data && tar xzvf bert-ja-juman.tar.gz -C bert-ja-juman/

download_xlnet: 
	mkdir -p data/xlnet
	aws s3 cp --recursive s3://camphr/trf_models/xlnet/ data/xlnet

download_udify:
	mkdir -p data/udify
	aws s3 cp s3://camphr/models/udify.tar.gz data/ \
	cd data && tar xzvf udify.tar.gz -C udify/ && \
	mv udify/udify-model tmp && rm -r udify && mv tmp udify

download_elmo:
	mkdir -p data/elmo
	aws s3 cp --recursive s3://camphr/models/elmo data/elmo

download_dataset:
	mkdir -p ~/datasets
	aws s3 cp s3://camphr/datasets/ ~/datasets --recursive
download_ldcc:
	mkdir -p ~/datasets
	aws s3 cp s3://camphr/datasets/ldcc.jsonl ~/datasets/ldcc.jsonl
	aws s3 cp s3://camphr/datasets/ldcc_labels.json ~/datasets/ldcc_labels.json
