PY ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip

.PHONY: install data train eval api ui test docker

install:
	$(PIP) install -r requirements.txt

data:
	$(PY) -m src.data --dataset simulate --n 50000 --out data/processed/experiment.parquet

hillstrom:
	$(PY) -m src.data --dataset hillstrom --campaign any --outcome visit --out data/processed/experiment.parquet

criteo:  # 311MB download; --sample-frac keeps it laptop-sized (~1.25M rows)
	$(PY) -m src.data --dataset criteo --sample-frac 0.05 --outcome visit --out data/processed/experiment.parquet

train:
	$(PY) -m src.train --data data/processed/experiment.parquet --learner xlearner

eval:
	$(PY) -m src.evaluate --data data/processed/experiment.parquet --model artifacts/xlearner.pkl

crosscheck:  # validate the from-scratch R-learner against econml (needs econml installed)
	$(PY) -m src.crosscheck --data data/processed/experiment.parquet

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

ui:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

docker:
	docker build -t uplift-engine . && docker run -p 8000:8000 uplift-engine
