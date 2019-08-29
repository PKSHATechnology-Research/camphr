# convert yamls to Python class
# requirements: yq, quicktype
# quicktype: brew install quicktype
convert:
	ls *.yml | xargs -I{} basename {} .yml  | xargs -I{} sh -c "cat {}.yml | yq . | quicktype -o label_{}.py --just-types"
