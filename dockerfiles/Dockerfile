FROM continuumio/miniconda3:24.5.0-0
USER root
WORKDIR /workdir
#Configure Poetry
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.8.3
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"
RUN apt update \
    && apt install -y --no-install-recommends \
    bzip2 \
    build-essential \
    curl \
    gcc \
    git \
    wget \
    ca-certificates \
    curl \
    procps \
    gettext-base \
    # clean
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC

# install conda environment
ADD conda-linux-64.lock /tmp/conda-linux-64.lock
RUN conda update -n base -c defaults conda \
    && conda update --all \
    && conda clean -afy
RUN conda install --freeze-installed --name base --file /tmp/conda-linux-64.lock \
        && conda clean -afy

# install dependencies
COPY poetry.lock pyproject.toml ./
RUN pip install poetry \
    && poetry config virtualenvs.in-project true \
    && poetry config virtualenvs.path /opt/conda/envs \
    && poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi -vvv

SHELL ["/bin/bash", "-c"]


# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
ADD .jupyter /root/.jupyter

ENV PATH /opt/conda/envs:$PATH

ENV PYTHONPATH $PYTHONPATH:/workdir/src:/workdir

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
# WORKDIR /workdir
