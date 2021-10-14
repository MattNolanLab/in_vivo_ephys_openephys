FROM ubuntu:xenial-20190720

# System packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends wget bzip2 software-properties-common

# Conda
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O miniconda.sh && \
    [ "b4e46fcc8029e2cfa731b788f25b1d36" = "$(md5sum miniconda.sh | cut -f 1 -d ' ')" ] && \
    bash miniconda.sh -b -p /opt/conda/ && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

COPY environment.yml environment.yml
SHELL ["/bin/bash", "-lc"]
RUN conda env create -f environment.yml && \
    echo "conda activate ms4" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN mkdir -p /home/ubuntu/to_sort/recordings  # Lazy convenience folder to match Eleanor