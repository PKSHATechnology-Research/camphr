packaging:
	cd scripts; pipenv run papermill packaging.ipynb output.ipynb
	
download:
	aws s3 cp s3://bedoner/data.tar.gz .
	tar xzvf data.tar.gz
