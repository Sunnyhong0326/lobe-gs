FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6+PTX;8.9;9.0"
ENV CUDA_ARCHITECTURES="70;89;90"
ENV DEBIAN_FRONTEND=noninteractive

ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        tar \
        wget \
        ffmpeg \
        exiftool \
        libsm6 \
        libxext6 \
        tree \
        g++ \
        openssh-client \
        cmake \
        ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN if getent group ${GID} > /dev/null 2>&1; then \
        groupmod -n appgroup $(getent group ${GID} | cut -d: -f1); \
    else \
        groupadd -g ${GID} appgroup; \
    fi && \
    if id -u ${UID} > /dev/null 2>&1; then \
        usermod -l appuser -g ${GID} -d /home/appuser -m $(id -un ${UID}); \
    else \
        useradd -r -u ${UID} -m -s /bin/bash -g appgroup appuser; \
    fi

RUN mkdir -p /app /opt/venv && \
    chown -R appuser:appgroup /app /opt/venv

WORKDIR /app

COPY --chown=appuser:appgroup ./requirements.txt ./requirements.txt
COPY --chown=appuser:appgroup ./submodules ./submodules

USER appuser

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv --system-site-packages $VIRTUAL_ENV

RUN pip install --no-cache-dir --upgrade pip setuptools

RUN pip install --no-cache-dir ninja numpy==1.26.4 jaxtyping rich

RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation submodules/simple-knn && \
    pip install --no-cache-dir --no-build-isolation submodules/fused-ssim 
    
RUN pip install --no-cache-dir --no-build-isolation submodules/fused-bilagrid

RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --upgrade "Brotli>=1.2.0"
RUN pip install --no-cache-dir "huggingface-hub==2.3.0"

RUN rm -rf ./requirements.txt ./submodules
