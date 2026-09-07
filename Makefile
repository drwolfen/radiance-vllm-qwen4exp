IMAGE_NAME ?= drwolfen/radiance-vllm-qwen4exp:0.1.0

.PHONY: help build run stop test bench push logs

help:
	@echo "radiance-vllm-qwen4exp Makefile commands:"
	@echo "  make build      Build Docker image"
	@echo "  make run        Start isolated staging container on port 8085"
	@echo "  make stop       Stop staging container"
	@echo "  make logs       Tail container logs"
	@echo "  make test       Run functional & tool-calling tests"
	@echo "  make bench      Run throughput benchmark"
	@echo "  make push       Push image to Docker registry"

build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

test:
	python3 tests/test_tool_calling.py --port 8085
	python3 tests/test_262k_retrieval.py --port 8085

bench:
	python3 tests/bench_throughput.py --port 8085 --tokens 128

push:
	docker push $(IMAGE_NAME)
