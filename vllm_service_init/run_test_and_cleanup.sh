#!/bin/bash

export no_proxy="127.0.0.1,localhost"

bash vllm_service_init/start_server.sh &
SERVER_PID=$!

sleep 240

python vllm_service_init/test.py

kill $SERVER_PID