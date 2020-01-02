init:
	virtualenv .env
	source .env/bin/activate
	pip install /home/rferrari/software/tensorflow/tensorflow/tensorflow_pkg/tensorflow*.whl
	pip install -r requirements.txt

test:
	py.test tests

.PHONY: init test
