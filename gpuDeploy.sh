docker run --runtime=nvidia -it -p 8501:8501 \
-v "$(pwd)/model:/models/docker_test" \
-e MODEL_NAME=docker_test tensorflow/serving:latest-gpu