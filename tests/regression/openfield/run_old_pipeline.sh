tar xvf ~/testdata/M5_2018-03-06_15-34-44_of.tar.gz -C ~/testdata
docker run -it -v ${PWD}/regression:/code \
    -v ~/testdata:/home/nolanlab/to_sort/recordings \
    -v ${PWD}/tests/regression/output/openfield:/ActiveProjects/Klara/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of/ \
    -v ${PWD}/regression/sorting_files:/home/nolanlab/to_sort/sorting_files \
    -e SERVER_PATH_FIRST_HALF='/ActiveProjects/' \
    -e DEBUG=1 \
    -e SINGLE_RUN=1 \
    -w /code teristam/ephys_ms3  /bin/bash -l ./run_analysis.sh