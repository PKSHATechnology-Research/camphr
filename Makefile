download_data:
	aws s3 cp s3://bedoner/data.tar.gz .
	tar xzvf data.tar.gz
