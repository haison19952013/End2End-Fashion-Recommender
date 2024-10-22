python = venv/bin/python
pip = venv/bin/pip
.PHONY: run_apps

setup:
	python3 -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

run_api_apps:
	@PYTHONPATH=$$(pwd):$$(pwd)/src/training_pipeline; \
	export PYTHONPATH=$$PYTHONPATH; \
	echo "Running with PYTHONPATH=$$PYTHONPATH"; \
	uvicorn src.apps.my_api:app --reload

run_web_apps:
	@PYTHONPATH=$$(pwd):$$(pwd)/src/training_pipeline; \
	export PYTHONPATH=$$PYTHONPATH; \
	echo "Running with PYTHONPATH=$$PYTHONPATH"; \
	streamlit run src/web_apps/my_apps.py

clone_data:
	$(dvc) pull data.dvc
	
clone_model:
	$(dvc) pull model.dvc

clone_embedding:
	$(dvc) pull embedding_data.dvc

make_recommendation:
	$(python) src/inference/make_recommendations.py --scene_path data_test/emoi.jpg 



run:
	$(python) src/training_pipeline/train.py

mlflow:
	venv/bin/mlflow ui

test:
	$(python) -m pytest
		
clean:
	rm -rf steps/__pycache__
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf tests/__pycache__

remove:
	rm -rf venv
	rm -rf mlruns