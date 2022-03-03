export PYTHONPATH=$PYTHONPATH:$PWD
conda activate ms4 && python3 -m pytest -s tests/integration
