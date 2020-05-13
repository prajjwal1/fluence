SRC = $(wildcard ./*.ipynb)

all: fluence docs

fluence: $(SRC)
	nbdev_build_lib
	touch fluence

style:
	isort --recursive fluence 
	black --line-length 119 --target-version py35 fluence 

quality:
	black --check --line-length 119 --target-version py35 fluence 
	isort --check-only --recursive fluence 
	flake8 fluence examples 

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist
