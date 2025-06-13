install:
	pip install -e .

run_agent:
	python -m agent

run_library:
	python -m library

run_server:
	python -m server

run_client:
	python src/server/protobuf/test_client.py

run_proto:
	python src/server/protobuf/test_once.py

run_gesture:
	python -m gesture
