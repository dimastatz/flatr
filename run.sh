#!/bin/bash

abort()
{
    echo "*** FAILED ***" >&2
    exit 1
}

if [ "$#" -eq 0 ]; then
    echo "No arguments provided. Usage: 
    1. '-local' to build local environment
    2. '-test' to run local tests"
elif [ $1 = "-local" ]; then
    trap 'abort' 0
    set -e
    echo "Running format, linter and tests"
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements.txt

    black flatr tests
    pylint --fail-under=9.9 flatr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov flatr -v tests
elif [ $1 = "-test" ]; then
    trap 'abort' 0
    set -e
    
    echo "Running format, linter and tests"
    source .venv/bin/activate
    black flatr tests
    pylint --fail-under=9.9 flatr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov --log-cli-level=INFO flatr -v tests
elif [ $1 = "-docker" ]; then
    echo "Building and running docker image"
    docker stop flatr-container
    docker rm flatr-container
    docker rmi flatr-image
    # build docker
    docker build --tag flatr-image --build-arg CACHEBUST=$(date +%s) . --file Dockerfile.test
elif [ $1 = "-deploy-package" ]; then
    echo "Running WhisperFlow package setup"
    pip install twine
    pip install wheel
    python setup.py sdist bdist_wheel
    rm -rf .venv_test
    python3 -m venv .venv_test
    source .venv_test/bin/activate
    pip install ./dist/flatr-0.1-py3-none-any.whl
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov whisperflow -v tests
    # twine upload ./dist/*
else
  echo "Wrong argument is provided. Usage:
    1. '-local' to build local environment
    3. '-test' to run linter, formatter and tests"
fi

trap : 0
echo >&2 '*** DONE ***'