style:
	isort --recursive --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 fluence tests examples
	black fluence tests examples

quality:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check fluence tests examples

