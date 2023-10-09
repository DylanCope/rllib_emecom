docker run --gpus all -it --rm --name rllib_emecom -p :8888:8888 -p :8265:8265 -v $(pwd)/:/home/ray/project --entrypoint=/bin/bash drc/rllib_emecom:latest
