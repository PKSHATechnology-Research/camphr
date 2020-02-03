download: download_udify download_elmo # from bedore-rannd account

UDIFY_PATH := en_udify-0.5
ELMO_PATH := en_elmo_medium-0.5

download_udify:
	mkdir -p data && \
	cd data && github-asset get ${UDIFY_PATH}.tar.gz && \
	tar xzvf ${UDIFY_PATH}.tar.gz && \
	poetry run pip install -e ${UDIFY_PATH} --no-deps

download_elmo:
	mkdir -p data && \
	cd data && github-asset get ${ELMO_PATH}.tar.gz && \
	tar xzvf ${ELMO_PATH}.tar.gz && \
	poetry run pip install -e ${ELMO_PATH} --no-deps
