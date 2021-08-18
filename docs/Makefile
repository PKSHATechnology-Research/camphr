PACKAGES := camphr camphr_transformers
dev:
	watchmedo shell-command --patterns='*.rst;source/conf.py;../packages/**/*py' --ignore-pattern=_build/* --recursive --command='make html' &
	cd build/html && python -m http.server 5000

source/packages/%.rst: DIR = ../packages/$(basename $(notdir $@))
source/packages/%.rst: $(shell find $(DIR) -type f -name "*.py" -not -path '*/.venv/*' -not -path '*/tests/*')
	sphinx-apidoc -M -d 2 -f -o source/packages/ $(DIR) $(DIR)/tests '$(DIR)/*/VERSION.py'
	

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: Makefile $(addsuffix .rst, $(addprefix source/packages/,$(PACKAGES)))
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
