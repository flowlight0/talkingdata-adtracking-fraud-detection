#! /bin/bash


docker_host_name=talkingdata

function host_exists() {
    host_count=$(docker-machine ls --filter "name=${docker_host_name}" | wc -l)
    if [ ${host_count} -gt 1 ]; then
        return 0;
    else
        return 1;
    fi;
}

if host_exists; then
    echo "Docker host \"${docker_host_name}\" already exists";
else
    echo "Create docker host \"${docker_host_name}\"";
    docker-machine create --driver amazonec2 \
        --amazonec2-vpc-id vpc-5eac2b3b \
        --amazonec2-ami ami-ea4eae8c \
        --amazonec2-region ap-northeast-1  \
        --amazonec2-zone a  \
        --amazonec2-root-size 512 \
        --amazonec2-instance-type r3.4xlarge \
        --amazonec2-request-spot-instance \
        --amazonec2-spot-price 1.28 \
        ${docker_host_name} || exit 1;
fi;

eval $(docker-machine env ${docker_host_name})

if [ -z ACCESS_TOKEN ]; then
    echo "ACCESS_TOKEN must have access token for https://github.com/flowlight0/talkingdata-adtracking-fraud-detection";
    exit 1;
fi;

if [ -z JUPYTER_PASSWORD_HASH ]; then
    echo "JUPYTER_PASSWORD_HASH must have password hash generated from \"from IPython.lib import passwd; passwd()\"";
    exit 1;
fi;
echo ${JUPYTER_PASSWORD_HASH}
docker build . \
     --build-arg ACCESS_TOKEN=${ACCESS_TOKEN} \
     --build-arg JUPYTER_PASSWORD_HASH=${JUPYTER_PASSWORD_HASH} \
     -t kaggle/flowlight
