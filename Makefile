MAKEFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_PATH)/)
PROJECT_NAME=mofuchan
IMAGE_TAG=v0.1
HOST_JUPYTER_PORT=8888
APP_PORT=7860
DOCKER_ADDOPTS=\
	-v $(PROJECT_ROOT)/notebook:/workdir/notebook \
	-v $(PROJECT_ROOT)/src:/workdir/src \
	-v $(PROJECT_ROOT)/script:/workdir/script \
	--env-file .env \
	$(OPTS)

conda_lock:
	conda-lock -f environment.yml -p linux-64 -k explicit --filename-template "conda-{platform}.lock"

build:
	docker build -t $(PROJECT_NAME):$(IMAGE_TAG) -f $(PROJECT_ROOT)/dockerfiles/Dockerfile $(PROJECT_ROOT)

run-apps::
	@echo "########################################"
	@echo ""
	@echo "Access"
	@echo "http://localhost:$(APP_PORT)"
	@echo ""
	@echo "########################################"

run-apps::
	docker run --rm $(DOCKER_ADDOPTS) -p $(APP_PORT):7860 --name mofuchan-apps $(PROJECT_NAME):$(IMAGE_TAG) python src/app/app.py

run-bash:
	docker run --rm -it $(DOCKER_ADDOPTS) -p $(APP_PORT):7860 --name mofuchan-bash $(PROJECT_NAME):$(IMAGE_TAG) bash

run-notebook::
	docker run -d $(DOCKER_ADDOPTS) -p $(HOST_JUPYTER_PORT):8888 --name mofuchan-notebook $(PROJECT_NAME):$(IMAGE_TAG) jupyter lab --no-browser

run-notebook::
	@echo "########################################"
	@echo ""
	@echo "Access"
	@echo "http://localhost:$(HOST_JUPYTER_PORT)"
	@echo ""
	@echo "########################################"

# remove the container
rm-notebook:
	docker rm -f mofuchan-notebook || echo

lint:
	@black .
