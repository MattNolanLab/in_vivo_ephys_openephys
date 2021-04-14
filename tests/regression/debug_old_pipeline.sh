# tar xvf ~/testdata/M1_D31_short.tar.gz -C ~/testdata
docker run -it -v ${PWD}/regression:/code \
    -v ~/testdata:/home/nolanlab/to_sort/recordings \
    -v ${PWD}/tests/regression/output:/ActiveProjects/Junji/Data/2021cohort1/vr/M1_D2_2021-01-12_13-28-48 \
    -v ${PWD}/regression/sorting_files:/home/nolanlab/to_sort/sorting_files \
    -e SERVER_PATH_FIRST_HALF='/ActiveProjects/' \
    -e DEBUG=1 \
    -e SINGLE_RUN=1 \
    -w /code teristam/ephys_ms3  /bin/bash -l ./run_analysis.sh