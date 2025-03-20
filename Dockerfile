FROM python:3.10-slim

LABEL maintainer="CNES"

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    binutils \
    libproj-dev \
    gdal-bin \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir bulldozer-dtm \
    && rm -rf /var/lib/apt/lists/*;

# Adding a default user to prevent root access
RUN groupadd bulldozer && useradd bulldozer -g bulldozer;

USER bulldozer

ENTRYPOINT ["bulldozer"] 
CMD ["-lu"]
