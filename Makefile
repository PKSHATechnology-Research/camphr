lock:
	pipenv lock -r > requirements.txt
	pipenv lock -r --dev >> requirements.txt
