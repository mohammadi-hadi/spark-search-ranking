# Counterfactual learning-to-rank on marketplace search logs, JVM included.
#   docker run --rm ghcr.io/mohammadi-hadi/spark-search-ranking
#   docker run --rm ghcr.io/mohammadi-hadi/spark-search-ranking --searches 20000
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/mohammadi-hadi/spark-search-ranking" \
      org.opencontainers.image.description="Counterfactual learning-to-rank for marketplace search logs in PySpark" \
      org.opencontainers.image.licenses="MIT"

# Spark runs on the JVM, and not having to install one is the point of this
# image. JAVA_HOME is deliberately left unset: spark-submit falls back to java
# on PATH, and a wrong JAVA_HOME fails harder than no JAVA_HOME at all.
# procps is here because Spark's launcher scripts call ps.
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home app
USER app
WORKDIR /home/app

# The library defaults are a full 20,000-search run, which is not what someone
# trying the image for the first time wants. Bare `docker run` is demo scale;
# pass --searches for the real thing.
ENTRYPOINT ["spark-search-ranking"]
CMD ["--searches", "2000"]
