
if [[ ! -z $( pip3 -V | grep "/.devops/") ]] 
then
	pip3 install -r requirements.txt
	pip3 install -r dev-requirements.txt

	wget https://github.com/hadolint/hadolint/releases/download/v1.22.1/hadolint-Linux-x86_64 -O .devops/hadolint
	chmod +x .devops/hadolint
else
	echo "Please run first 'make setup'"
fi
