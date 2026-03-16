# Triton ROCm + vLLM Backend build & publish
# Usage: just build-all

server_repo := "../triton-inference-server-server"
image := "nudibranches/tritonserver-rocm-vllm"
gpu_arch := "gfx1201"
triton_version := "2.64.0"
rocm_version := "7.2"
version := triton_version + "-rocm" + rocm_version + "-" + gpu_arch

# Build everything and push to Docker Hub
build-all: build-base build-triton build-image push

# Step 1: Build ROCm base image
build-base:
    cd {{server_repo}} && bash scripts/build_ubuntu24.04_rocm_72_base.sh

# Step 2: Build Triton Server
build-triton:
    cd {{server_repo}} && python3 build.py \
        --no-container-pull \
        --enable-logging \
        --enable-stats \
        --enable-tracing \
        --enable-rocm \
        --enable-metrics \
        --endpoint=grpc \
        --endpoint=http \
        --backend=python \
        --linux-distro=ubuntu \
        --target-platform=linux

# Step 3: Build final image with vLLM + backend
build-image:
    docker build -f Dockerfile.rocm \
        --build-arg BASE_IMAGE=tritonserver:latest \
        --build-arg PYTORCH_ROCM_ARCH={{gpu_arch}} \
        -t {{image}}:{{version}} \
        -t {{image}}:latest .

# Step 4: Push to Docker Hub
push:
    docker push {{image}}:{{version}}
    docker push {{image}}:latest

# Login to Docker Hub
login:
    docker login -u nudibranches

# Show version info
info:
    @echo "Image: {{image}}:{{version}}"
    @echo "Triton: {{triton_version}}"
    @echo "ROCm: {{rocm_version}}"
    @echo "GPU arch: {{gpu_arch}}"
