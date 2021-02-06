# bro
#
# VERSION               0.1

# Checkout and build Zeek
FROM debian:buster as builder
MAINTAINER Justin Azoff <justin.azoff@gmail.com>

ENV WD /scratch

RUN mkdir ${WD}
WORKDIR /scratch

RUN apt-get update && apt-get upgrade -y && echo 2010-05-15
RUN apt-get -y install build-essential git bison flex gawk cmake swig libssl-dev libmaxminddb-dev libpcap-dev python3.7-dev libcurl4-openssl-dev wget libncurses5-dev ca-certificates zlib1g-dev --no-install-recommends

ARG ZEEK_VER=3.2.0
ARG BUILD_TYPE=Release
ENV VER ${ZEEK_VER}
ADD ./common/buildbro ${WD}/common/buildbro
RUN ${WD}/common/buildbro zeek ${VER} ${BUILD_TYPE}

# For testing
ADD ./common/getmmdb.sh /usr/local/getmmdb.sh
ADD ./common/bro_profile.sh /usr/local/bro_profile.sh

# Get geoip data
FROM debian:buster as geogetter
ARG MAXMIND_LICENSE_KEY
RUN apt-get update && apt-get -y install wget ca-certificates --no-install-recommends

# For testing
#ADD ./common/getmmdb.sh /usr/local/bin/getmmdb.sh
COPY --from=builder /usr/local/getmmdb.sh /usr/local/bin/getmmdb.sh
RUN mkdir -p /usr/share/GeoIP
RUN /usr/local/bin/getmmdb.sh ${MAXMIND_LICENSE_KEY}
# This is a workaround for the case where getmmdb.sh does not create any files.
RUN touch /usr/share/GeoIP/.notempty

# Make final image
FROM debian:buster
ARG ZEEK_VER=3.2.0
#install runtime dependencies and dependencies for zkg and metron-kafka-plugin
RUN apt-get update \
    && apt-get -y install --no-install-recommends libpcap0.8 libssl1.1 libmaxminddb0 python3.7-minimal \
    && apt-get -y install git python3-pip cmake librdkafka-dev libssl-dev libpcap-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


#copy source for beeing able to compile plugins
COPY --from=builder /usr/src/ /usr/src
ENV ZEEK_DIST /usr/src/zeek-${ZEEK_VER}


COPY --from=builder /usr/local/zeek-${ZEEK_VER} /usr/local/zeek-${ZEEK_VER}
COPY --from=geogetter /usr/share/GeoIP/* /usr/share/GeoIP/
RUN rm -f /usr/share/GeoIP/.notempty
RUN ln -s /usr/local/zeek-${ZEEK_VER} /bro
RUN ln -s /usr/local/zeek-${ZEEK_VER} /zeek

# zeek-path is necessary for compiling plugin
ENV PATH /zeek/bin/:$PATH

RUN pip3 install zkg
#COPY ./config/zkg_config /root/.zkg/config
RUN zkg autoconfig
RUN zkg install --force --skiptests apache/metron-bro-plugin-kafka --version master

# copy local.zeek with Kafka configuration
COPY ./config/local.zeek /zeek/share/zeek/site/local.zeek

# For testing
#ADD ./common/bro_profile.sh /etc/profile.d/zeek.sh
COPY --from=builder /usr/local/bro_profile.sh /etc/profile.d/zeek.sh

COPY ./docker-entrypoint.sh /tmp/docker-entrypoint.sh
RUN chmod +x /tmp/docker-entrypoint.sh

RUN ln -s /usr/local/zeek-${ZEEK_VER}/spool/zeek /logs

# Entrypoint script from https://github.com/xx-zhang/docker-zeek/ with my individual adaptions
ENTRYPOINT ["/tmp/docker-entrypoint.sh"]

CMD /bin/bash -l
