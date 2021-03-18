export DATA_FOLDER=~/to_sort/recordings/M6_D3_2020-10-31_16-04-25
export DATASTORE_LOC=$(cat $DATA_FOLDER/parameters.txt | awk 'NR==2')
echo $DATASTORE_LOC
docker run -it -v ${PWD}/regression:/code \
    -v $DATA_FOLDER:/home/nolanlab/to_sort/recordings/M1_D1 \
    -v ${PWD}/tests/regression/output/openfield:/ActiveProjects/${DATASTORE_LOC}/ \
    -v ${PWD}/regression/sorting_files:/home/nolanlab/to_sort/sorting_files \
    -e SERVER_PATH_FIRST_HALF='/ActiveProjects/' \
    -e DEBUG=1 \
    -e SINGLE_RUN=1 \
    -w /code teristam/ephys_ms3  /bin/bash -l ./run_analysis.sh