# Radiance-vLLM Qwen4exp (Dual AMD Radeon AI PRO R9700 gfx1201)
# Based on Dyluhn/R9V architecture with dedicated FP8 MTP-2 Drafter

SHELL := /bin/bash
MODEL_DIR ?= /path/to/models/qwen38-r9v
DATA_DIR ?= $(HOME)/r9v-data
HOST_PORT ?= 8085
MAX_JOBS ?= 32

.PHONY: help doctor verify build ple run-staging stop clean test

help:
	@echo "Radiance-vLLM Qwen4exp Management"
	@echo "  make doctor      - Run hardware and environment preflight checks"
	@echo "  make verify      - Verify model package and hash integrity"
	@echo "  make build       - Build the R9V gfx1201 Docker runtime image"
	@echo "  make ple         - Extract and verify the PLE embedding table"
	@echo "  make run-staging - Launch container in staging mode on port $(HOST_PORT)"
	@echo "  make stop        - Stop running container"

doctor:
	R9V_MODEL_DIR=$(MODEL_DIR) R9V_PLE_PATH=$(DATA_DIR)/per_layer_token_embd.iq4_nl.bin ./r9v doctor qwen38

verify:
	./r9v verify qwen38 --model-dir $(MODEL_DIR)

build:
	R9V_MAX_JOBS=$(MAX_JOBS) ./r9v build qwen38

ple:
	mkdir -p $(DATA_DIR)
	python3 tools/prepare_ple.py \
		$(MODEL_DIR)/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
		$(MODEL_DIR)/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf \
		$(MODEL_DIR)/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf \
		--output $(DATA_DIR)/per_layer_token_embd.iq4_nl.bin

run-staging:
	R9V_MODEL_DIR=$(MODEL_DIR) \
	R9V_DATA_DIR=$(DATA_DIR) \
	R9V_PLE_PATH=$(DATA_DIR)/per_layer_token_embd.iq4_nl.bin \
	R9V_CACHE_DIR=$(DATA_DIR)/cache \
	R9V_HOST_PORT=$(HOST_PORT) \
	R9V_VISIBLE_DEVICES=0,1 \
	./r9v run qwen38

stop:
	docker stop r9v-qwen38-flash-next || true
