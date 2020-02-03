download: download_udify download_elmo # from bedore-rannd account

UDIFY_PATH := en_udify-0.5
ELMO_PATH := en_elmo_medium-0.5
REPO := PKSHATechnology/agsnlp_camphr

download_udify:
	mkdir -p data && \
	cd data && github-asset get ${UDIFY_PATH}.tar.gz --repo ${REPO} --token ${GITHUB_TOKEN} && \
	tar xzvf ${UDIFY_PATH}.tar.gz && \
	poetry run pip install -e ${UDIFY_PATH} --no-deps

download_elmo:
	mkdir -p data && \
	cd data && github-asset get ${ELMO_PATH}.tar.gz --repo ${REPO} --token ${GITHUB_TOKEN} && \
	tar xzvf ${ELMO_PATH}.tar.gz && \
	poetry run pip install -e ${ELMO_PATH} --no-deps
