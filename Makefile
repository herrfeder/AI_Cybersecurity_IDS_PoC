## The Makefile includes instructions on environment setup and lint tests
# Create and activate a virtual environment
# Install dependencies in requirements.txt
# Dockerfile should pass hadolint
# app.py should pass pylint
# (Optional) Build a simple integration test

SHELL := /bin/bash

setup:
	# Create python virtualenv & source it
	python3 -m venv .devops


install:
	pip3 install -r requirements.txt


devinstall:
	.circleci/scripts/install_requirements.sh


statictest:
	python -m pytest -vv tests/*.py

staticscan:
	bandit -r ./app -lll


dynamicscan:
	# doing some stuff with selenium for the frontend


sourcelint:
	autopep8 --aggressive --in-place --recursive -v app/
	find app -name '*.py' -print0 | xargs -I '{}' -0 pylint -E --disable=E1101 '{}'


dockerlint:
	.devops/hadolint --ignore DL3008 --ignore DL3013 deploy/broai_compose/broai-docker/Dockerfile
	.devops/hadolint --ignore DL3008 --ignore DL3013 deploy/broai_compose/zeek-docker/Dockerfile


all: install lint test
