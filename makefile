python = venv/bin/python
pip = venv/bin/pip

setup:
	python3 -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

get_data:
	$(python) src/fetch_images.py --input_file ../STL-Dataset/fashion.json --output_dir ../database --max_lines 5

run:
	$(python) src/train.py --input_file=STL-Dataset/fashion.json --image_dir=F:\database\shop_the_look-v1 --max_steps=30000 --learning_rate=0.0001618 --regularization=0.2076  --output_size=64 --checkpoint_every_steps=10000 --restore_checkpoint=False --model_name=pinterest_stl_model_rc1

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