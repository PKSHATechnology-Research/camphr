rm -rf build
poetry run make html
aws s3 rm --recursive s3://camphr-doc
aws s3 cp --recursive build/html/ s3://camphr-doc
