format:
	black .
	isort .

lint:
	pylint your_package_name/ orchestration/ gradio_app/

test:
	pytest

all: format lint test
