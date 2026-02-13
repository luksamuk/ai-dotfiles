FROM apache/tika:3.2.3.0-full
USER root
RUN set -eux \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
       tesseract-ocr-por \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY ./tika-config.xml /tika-config.xml
USER $UID_GID

ENTRYPOINT [ "/bin/sh", "-c", "exec java -cp \"/tika-server-standard-${TIKA_VERSION}.jar:/tika-extras/*\" org.apache.tika.server.core.TikaServerCli -c /tika-config.xml -h 0.0.0.0 $0 $@"]

