CC ?= cc

CPPFLAGS ?= -Isrcs/includes
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -Werror -O2
LDFLAGS ?=
LDLIBS ?=

# Per-TU AVX flags. Global CFLAGS stay generic so the binary loads on pre-AVX2
# hardware; -mavx2 -mfma -mf16c attach only to cpu_avx.o.
AVX_CFLAGS := -mavx2 -mfma -mf16c

# SIMD build switch. `make SIMD=off` drops cpu_avx.c from the link and defines
# LIS_DISABLE_AVX so cpu_dispatch.c compiles without referencing AVX symbols.
SIMD ?= on

BUILD_DIR := srcs/libs
OBJ_DIR := $(BUILD_DIR)/obj
BIN := $(BUILD_DIR)/lis
TEST_BIN := $(BUILD_DIR)/test_core
TEST_LOADER_BIN := $(BUILD_DIR)/test_loader
TEST_BACKEND_BIN := $(BUILD_DIR)/test_backend
TEST_CPU_AVX_BIN := $(BUILD_DIR)/test_cpu_avx
TEST_RUNTIME_BIN := $(BUILD_DIR)/test_runtime
TEST_CLI_BIN := $(BUILD_DIR)/test_cli
TEST_TOKENIZER_BIN := $(BUILD_DIR)/test_tokenizer
TEST_HF_IMPORT_BIN := $(BUILD_DIR)/test_hf_import
TEST_THREADING_BIN := $(BUILD_DIR)/test_threading

APP_SRCS := \
	srcs/cli/main.c \
	srcs/cli/cli.c \
	srcs/cli/driver.c

CORE_SRCS := \
	srcs/core/status.c \
	srcs/core/dtype.c \
	srcs/core/tensor.c \
	srcs/core/context.c \
	srcs/core/model.c \
	srcs/core/perf.c \
	srcs/core/artifact.c \
	srcs/core/cpu_features.c \
	srcs/core/trace.c \
	srcs/core/layer_trace.c

LOADER_SRCS := \
	srcs/loader/source.c \
	srcs/loader/safetensors.c \
	srcs/loader/config.c \
	srcs/loader/hf_model.c

BACKEND_SRCS := \
	srcs/backend/operator.c \
	srcs/backend/cpu_reference.c \
	srcs/backend/cpu_kernels_reference.c \
	srcs/backend/cpu_dispatch.c

ifeq ($(SIMD),off)
CPPFLAGS += -DLIS_DISABLE_AVX
else
BACKEND_SRCS += srcs/backend/cpu_avx.c
endif

RUNTIME_SRCS := \
	srcs/runtime/batch.c \
	srcs/runtime/kv_cache.c \
	srcs/runtime/runtime.c \
	srcs/runtime/llama.c \
	srcs/runtime/qwen3.c \
	srcs/runtime/thread_pool.c

TOKENIZER_SRCS := \
	srcs/tokenizer/token_ids.c \
	srcs/tokenizer/tokenizer.c \
	srcs/tokenizer/vocab.c \
	srcs/tokenizer/bpe.c \
	srcs/tokenizer/json_parse.c \
	srcs/tokenizer/hf_import.c

TEST_SRCS := \
	tests/core/test_core.c

TEST_LOADER_SRCS := \
	tests/loader/test_loader.c

TEST_BACKEND_SRCS := \
	tests/backend/test_backend.c

TEST_CPU_AVX_SRCS := \
	tests/backend/test_cpu_avx.c

TEST_RUNTIME_SRCS := \
	tests/runtime/test_runtime.c

TEST_CLI_SRCS := \
	tests/cli/test_cli.c

TEST_TOKENIZER_SRCS := \
	tests/tokenizer/test_tokenizer.c

TEST_HF_IMPORT_SRCS := \
	tests/tokenizer/test_hf_import.c

TEST_THREADING_SRCS := \
	tests/runtime/test_threading.c

APP_OBJS := $(APP_SRCS:%.c=$(OBJ_DIR)/%.o)
CORE_OBJS := $(CORE_SRCS:%.c=$(OBJ_DIR)/%.o)
LOADER_OBJS := $(LOADER_SRCS:%.c=$(OBJ_DIR)/%.o)
BACKEND_OBJS := $(BACKEND_SRCS:%.c=$(OBJ_DIR)/%.o)
RUNTIME_OBJS := $(RUNTIME_SRCS:%.c=$(OBJ_DIR)/%.o)
TOKENIZER_OBJS := $(TOKENIZER_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_OBJS := $(TEST_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_LOADER_OBJS := $(TEST_LOADER_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_BACKEND_OBJS := $(TEST_BACKEND_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_CPU_AVX_OBJS := $(TEST_CPU_AVX_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_RUNTIME_OBJS := $(TEST_RUNTIME_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_CLI_OBJS := $(TEST_CLI_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_TOKENIZER_OBJS := $(TEST_TOKENIZER_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_HF_IMPORT_OBJS := $(TEST_HF_IMPORT_SRCS:%.c=$(OBJ_DIR)/%.o)
TEST_THREADING_OBJS := $(TEST_THREADING_SRCS:%.c=$(OBJ_DIR)/%.o)
CLI_DRIVER_OBJS := \
	$(OBJ_DIR)/srcs/cli/cli.o \
	$(OBJ_DIR)/srcs/cli/driver.o
OBJS := $(APP_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) \
	$(RUNTIME_OBJS) $(TEST_OBJS) $(TEST_LOADER_OBJS) $(TEST_BACKEND_OBJS) \
	$(TEST_CPU_AVX_OBJS) $(TEST_RUNTIME_OBJS) $(TOKENIZER_OBJS) \
	$(TEST_CLI_OBJS) $(TEST_TOKENIZER_OBJS) $(TEST_HF_IMPORT_OBJS) \
	$(TEST_THREADING_OBJS)
DEPS := $(OBJS:.o=.d)

.PHONY: all build test docs clean bench verify verify-kernels verify-cli \
	verify-token-parity verify-qwen3-sanity verify-perf-smoke

# Default build/test do not require model artifacts. Model-backed validation
# and benchmarks are optional local/manual gates driven by explicit env vars.
# Examples:
#   make verify-qwen3-sanity VERIFY_QWEN3_MODEL=/path/to/qwen3-dense
#   make bench BENCH_MODEL=/path/to/plain-rope-llama
BENCH_MODEL ?=
VERIFY_MODEL ?=
VERIFY_CONFIG ?=
VERIFY_HF_TOKENIZER ?=
VERIFY_QWEN3_MODEL ?=
VERIFY_OUT_DIR ?= tests/verification
VERIFY_CONFIG_PATH = $(if $(strip $(VERIFY_MODEL)),$(if $(strip $(VERIFY_CONFIG)),$(VERIFY_CONFIG),$(VERIFY_MODEL)/config.json),)
VERIFY_HF_TOKENIZER_PATH = $(if $(strip $(VERIFY_MODEL)),$(if $(strip $(VERIFY_HF_TOKENIZER)),$(VERIFY_HF_TOKENIZER),$(VERIFY_MODEL)/tokenizer.json),)

all: build

build: $(BIN)

$(BIN): $(APP_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(TOKENIZER_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(APP_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(TOKENIZER_OBJS) $(LDLIBS) -lm -lpthread -o $@

$(TEST_BIN): $(TEST_OBJS) $(CORE_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_OBJS) $(CORE_OBJS) $(LDLIBS) -o $@

$(TEST_LOADER_BIN): $(TEST_LOADER_OBJS) $(CORE_OBJS) $(LOADER_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_LOADER_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(LDLIBS) -o $@

$(TEST_BACKEND_BIN): $(TEST_BACKEND_OBJS) $(CORE_OBJS) $(BACKEND_OBJS) $(OBJ_DIR)/srcs/runtime/thread_pool.o | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_BACKEND_OBJS) $(CORE_OBJS) $(BACKEND_OBJS) $(OBJ_DIR)/srcs/runtime/thread_pool.o $(LDLIBS) -lm -lpthread -o $@

$(TEST_CPU_AVX_BIN): $(TEST_CPU_AVX_OBJS) $(CORE_OBJS) $(BACKEND_OBJS) $(OBJ_DIR)/srcs/runtime/thread_pool.o | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_CPU_AVX_OBJS) $(CORE_OBJS) $(BACKEND_OBJS) $(OBJ_DIR)/srcs/runtime/thread_pool.o $(LDLIBS) -lm -lpthread -o $@

$(TEST_RUNTIME_BIN): $(TEST_RUNTIME_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_RUNTIME_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(LDLIBS) -lm -lpthread -o $@

$(TEST_CLI_BIN): $(TEST_CLI_OBJS) $(CLI_DRIVER_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(TOKENIZER_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_CLI_OBJS) $(CLI_DRIVER_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(TOKENIZER_OBJS) $(LDLIBS) -lm -lpthread -o $@

$(TEST_TOKENIZER_BIN): $(TEST_TOKENIZER_OBJS) $(CORE_OBJS) $(TOKENIZER_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_TOKENIZER_OBJS) $(CORE_OBJS) $(TOKENIZER_OBJS) $(LDLIBS) -lm -o $@

$(TEST_HF_IMPORT_BIN): $(TEST_HF_IMPORT_OBJS) $(CORE_OBJS) $(TOKENIZER_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_HF_IMPORT_OBJS) $(CORE_OBJS) $(TOKENIZER_OBJS) $(LDLIBS) -o $@ -lm

$(TEST_THREADING_BIN): $(TEST_THREADING_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(TEST_THREADING_OBJS) $(CORE_OBJS) $(LOADER_OBJS) $(BACKEND_OBJS) $(RUNTIME_OBJS) $(LDLIBS) -lm -lpthread -o $@

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

# Per-TU AVX flags. Only cpu_avx.c compiles with -mavx2 -mfma -mf16c so the
# rest of the binary is safe to load on pre-AVX2 hardware.
$(OBJ_DIR)/srcs/backend/cpu_avx.o: srcs/backend/cpu_avx.c | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(AVX_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

test: build $(TEST_BIN) $(TEST_LOADER_BIN) $(TEST_BACKEND_BIN) $(TEST_CPU_AVX_BIN) $(TEST_RUNTIME_BIN) $(TEST_CLI_BIN) $(TEST_TOKENIZER_BIN) $(TEST_HF_IMPORT_BIN) $(TEST_THREADING_BIN)
	$(BIN) --help >/dev/null
	$(TEST_BIN)
	$(TEST_LOADER_BIN)
	$(TEST_BACKEND_BIN)
	$(TEST_CPU_AVX_BIN)
	$(TEST_RUNTIME_BIN)
	$(TEST_CLI_BIN)
	$(TEST_TOKENIZER_BIN)
	$(TEST_HF_IMPORT_BIN)
	$(TEST_THREADING_BIN)

verify: verify-kernels verify-cli

# Kernel verification is backend-neutral. The current implementation reuses
# the existing AVX diff binary behind this broader entry point.
verify-kernels: $(TEST_CPU_AVX_BIN)
	$(TEST_CPU_AVX_BIN)

verify-cli: $(TEST_CLI_BIN)
	$(TEST_CLI_BIN)

verify-token-parity: $(BIN)
	@if [ -z "$(strip $(VERIFY_MODEL))" ]; then \
		printf '%s\n' "error: VERIFY_MODEL is required for $@. Set VERIFY_MODEL=/path/to/plain-rope-llama."; \
		exit 2; \
	fi
	python3 tests/verification/run_token_parity.py \
		--bin $(BIN) \
		--model $(VERIFY_MODEL) \
		--config $(VERIFY_CONFIG_PATH) \
		--hf-tokenizer $(VERIFY_HF_TOKENIZER_PATH) \
		--out-json $(VERIFY_OUT_DIR)/token_parity_snapshot.json

verify-qwen3-sanity: $(BIN)
	@if [ -z "$(strip $(VERIFY_QWEN3_MODEL))" ]; then \
		printf '%s\n' "error: VERIFY_QWEN3_MODEL is required for $@. Set VERIFY_QWEN3_MODEL=/path/to/qwen3-dense."; \
		exit 2; \
	fi
	python3 tests/verification/run_qwen3_sanity.py \
		--bin $(BIN) \
		--model $(VERIFY_QWEN3_MODEL) \
		--out-json $(VERIFY_OUT_DIR)/qwen3_sanity_snapshot.json

verify-perf-smoke: $(BIN)
	@if [ -z "$(strip $(VERIFY_MODEL))" ]; then \
		printf '%s\n' "error: VERIFY_MODEL is required for $@. Set VERIFY_MODEL=/path/to/plain-rope-llama."; \
		exit 2; \
	fi
	python3 tests/perf/run_perf_matrix.py \
		--bin $(BIN) \
		--model $(VERIFY_MODEL) \
		--config $(VERIFY_CONFIG_PATH) \
		--hf-tokenizer $(VERIFY_HF_TOKENIZER_PATH) \
		--prompt-sizes short \
		--generates 1 \
		--threads 1 \
		--measured-runs 1 \
		--out-csv $(VERIFY_OUT_DIR)/perf_smoke.csv \
		--out-md $(VERIFY_OUT_DIR)/perf_smoke.md \
		--out-json $(VERIFY_OUT_DIR)/perf_smoke.json

docs:
	@printf 'Public documentation:\n'
	@printf '  docs/performance_measurement.md\n'
	@printf '  docs/repro_execution_artifacts.md\n'
	@printf '  docs/loader_format_scope.md\n'
	@printf '  docs/hf_tokenizer_compat.md\n'
	@printf '  docs/verification_framework.md\n'
	@printf '  docs/precision_policy.md\n'
	@printf '  docs/qwen3_dense_scope.md\n'

bench: $(BIN)
	@if [ -z "$(strip $(BENCH_MODEL))" ]; then \
		printf '%s\n' "error: BENCH_MODEL is required for $@. Set BENCH_MODEL=/path/to/plain-rope-llama."; \
		exit 2; \
	fi
	python3 tests/perf/run_perf_matrix.py \
		--bin $(BIN) \
		--model $(BENCH_MODEL) \
		--config $(BENCH_MODEL)/config.json \
		--hf-tokenizer $(BENCH_MODEL)/tokenizer.json \
		--out-json tests/perf/results.json

clean:
	rm -rf $(OBJ_DIR) $(BIN) $(TEST_BIN) $(TEST_LOADER_BIN) $(TEST_BACKEND_BIN) $(TEST_CPU_AVX_BIN) $(TEST_RUNTIME_BIN) $(TEST_CLI_BIN) $(TEST_TOKENIZER_BIN) $(TEST_HF_IMPORT_BIN) $(TEST_THREADING_BIN)

-include $(DEPS)
