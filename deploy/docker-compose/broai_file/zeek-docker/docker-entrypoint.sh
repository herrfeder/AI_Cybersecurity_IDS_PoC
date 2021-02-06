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

interfaces="$(ip link | awk -F: '$0 !~ "lo|vir|br-|docker|^[^0-9]"{print $2;getline}' | sed -z 's/\n/ /g;s/ $/\n/')"
default_interface="$(ip route get 8.8.8.8 | awk '{print $5}' )"
sed -i "s/^interface=eth0/interface='$default_interface'/" /zeek/etc/node.cfg 

cat /zeek/etc/node.cfg

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
