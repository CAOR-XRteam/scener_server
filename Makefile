install:
	pip install -e .

run_agent:
	python src/agent/__main__.py

run_library:
	python src/library/__main__.py

run_server:
	python src/server/__main__.py
