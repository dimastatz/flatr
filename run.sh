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

    black flattr tests
    pylint --fail-under=9.9 flattr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov flattr -v tests
elif [ $1 = "-test" ]; then
    trap 'abort' 0
    set -e
    
    echo "Running format, linter and tests"
    source .venv/bin/activate
    black flattr tests
    pylint --fail-under=9.9 flattr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov --log-cli-level=INFO flattr -v tests

else
  echo "Wrong argument is provided. Usage:
    1. '-local' to build local environment
    3. '-test' to run linter, formatter and tests"
fi

trap : 0
echo >&2 '*** DONE ***'