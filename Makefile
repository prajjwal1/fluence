SRC = $(wildcard ./*.ipynb)

all: fluence docs

fluence: $(SRC)
	nbdev_build_lib
	touch fluence

style:
	black fluence 
	isort --recursive --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 fluence

quality:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check fluence
	isort --check-only --recursive --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 fluence

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
