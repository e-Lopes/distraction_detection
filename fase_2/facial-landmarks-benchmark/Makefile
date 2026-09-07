.PHONY: build download cuda-check smoke public custom analysis

RUN_ID ?= benchmark
REPEATS ?= 5
MAXFRAMES ?= 0
MAXVIDEOS ?= 0

build:
	docker compose build

download:
	docker compose run --rm downloader drozy yawdd nitymed

cuda-check:
	docker compose run --rm --no-deps --entrypoint python3 insightface -m src.diagnostics.check_cuda

smoke:
	python src/experiment/run_matrix.py --run-id $(RUN_ID)_smoke --repetitions 1 --max-frames 300 --max-videos 1

public:
	python src/experiment/run_matrix.py --run-id $(RUN_ID) --repetitions $(REPEATS) --max-frames $(MAXFRAMES) --max-videos $(MAXVIDEOS)

custom:
	python src/experiment/run_matrix.py --run-id $(RUN_ID)_custom --datasets custom --repetitions $(REPEATS) --max-frames $(MAXFRAMES) --max-videos $(MAXVIDEOS)

analysis:
	docker compose run --rm --no-deps analysis
