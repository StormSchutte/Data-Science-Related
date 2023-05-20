#!/bin/bash

#   chmod +x build_and_run.sh
#   ./build_and_run.sh


# Build the Docker image
docker build -t hadoop-docker .

# Run the Docker container
docker run -it --rm \
    -p 9870:9870 -p 9864:9864 -p 9866:9866 -p 9867:9867 -p 19888:19888 \
    -p 8088:8088 -p 8888:8888 -p 8030:8030 -p 8031:8031 -p 8032:8032 -p 8033:8033 \
    -p 8040:8040 -p 8042:8042 -p 8080:8080 \
    --name hadoop-docker \
    hadoop-docker
