#!/bin/bash

# exit script if an error is encountered
set -e
export PATH=/zeek/bin:$PATH


if [ ! -f /zeek/etc/node.cfg ] || [ ! -s /zeek/etc/node.cfg ]; then
	# node.cfg doesn't exist or is empty
	if [ -t 0 ]; then
	    # at a tty, so start the config wizard
		zeekcfg -o /zeek/etc/node.cfg --type afpacket --processes 0 --no-pin
	fi
	if [ ! -f /zeek/etc/node.cfg ] || [ ! -s /zeek/etc/node.cfg ]; then
		# if still doesn't exist
		echo
		echo "You must first create a node.cfg file and mount it into the container."
		exit 1
	fi
fi

### install and run Apache for having test Web-Server

apt update
apt install apache2


### set interface in node.cfg
interfaces="$(ip link | awk -F: '$0 !~ "lo|vir|br-|docker|^[^0-9]"{print $2;getline}' | sed -z 's/\n/ /g;s/ $/\n/')"
default_interface="$(ip route get 8.8.8.8 | awk '{print $5}' )"
sed -i "s/^interface=eth0/interface='$default_interface'/" /zeek/etc/node.cfg 

### set environment variables into config files
# KAFKA_TOPIC KAFKA_HOST KAFKA_PORT

if [[ -z $KAFKA_TOPIC || -z $KAFKA_HOST || -z $KAFKA_PORT ]]; then
  echo 'one or more variables are undefined'
  echo 'Initialiaze as empty'
  export KAFKA_TOPIC=" "; export KAFKA_HOST=" "; export KAFKA_PORT=" "
fi

sed -i "s/\$KAFKA_TOPIC/$KAFKA_TOPIC/" /zeek/share/zeek/site/local.zeek
sed -i "s/\$KAFKA_HOST/$KAFKA_HOST/" /zeek/share/zeek/site/local.zeek
sed -i "s/\$KAFKA_PORT/$KAFKA_PORT/" /zeek/share/zeek/site/local.zeek

### set timezone

ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
echo $TZ > /etc/timezone



# do final log rotation
stop() {
	echo "Stopping zeek..."
	zeekctl stop
	trap - SIGINT SIGTERM
	exit
}

# run zeekctl diag on error
diag() {
	echo "Running zeekctl diag for debugging"
	zeekctl diag
	trap - ERR
}
trap 'diag' ERR

# ensure Zeek has a valid, updated config, and then start Zeek
echo "Checking your Zeek configuration..."
zeekctl deploy
zeekctl check >/dev/null
zeekctl install
zeekctl start

# ensure spool logs are rotated when container is stopped
trap 'stop' SIGINT SIGTERM

# periodically run the Zeek cron monitor to restart any terminated processes
zeekctl cron enable
# disable the zeekctl ERR trap as there are no more zeek commands to fail
trap - ERR

# daemonize cron but log output to stdout
#cron -b -L /dev/fd/1

# infinite loop to prevent container from exiting and allow this script to process signals
while :; do sleep 1s; done
