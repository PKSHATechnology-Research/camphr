packaging:
	cd scripts; pipenv run papermill packaging.ipynb output.ipynb
	
download: # from bedore-rannd account
	mkdir -p data
	aws s3 cp s3://bedoner/pytt_models/bert/bert-ja-juman.tar.gz data/
	cd data && tar xzvf bert-ja-juman.tar.gz

requirements:
	pipenv lock -r > requirements.txt