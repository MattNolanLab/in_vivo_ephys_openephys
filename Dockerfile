FROM ubuntu:xenial-20190720

# System packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends wget bzip2 software-properties-common

# Conda
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda2-4.7.10-Linux-x86_64.sh -O miniconda.sh && \
    [ "3bc6ffc6cda8efa467926dfd92a30bca" = "$(md5sum miniconda.sh | cut -f 1 -d ' ')" ] && \
    bash miniconda.sh -b -p /opt/conda/ && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

SHELL ["/bin/bash", "-lc"]
RUN conda config --append channels conda-forge && \
    conda create -yn env python=3.6.5 pandas joblib scipy matplotlib numba xlrd rpy2 cmocean scikit-image pytest && \
    echo "conda activate env" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# MountainSort 3
RUN add-apt-repository -y ppa:magland/mountainlab && \
    apt-get update && \
    apt-get install -y --no-install-recommends mountainlab mlpipeline mountainsort mountainview

### Add new dependencies after here ###

RUN conda activate env && \
    conda install -y astropy
    conda install -c anaconda statsmodels
