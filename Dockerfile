FROM ubuntu:latest as builder

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    nlohmann-json3-dev \
    libopencv-dev \
    libdlib-dev \
    g++ \
    git \
    curl \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set C++ compiler to g++ (optional, but good practice)
ENV CC=g++
ENV CXX=g++

# Verify g++ version (optional)
RUN g++ --version




# Set the working directory inside the container
WORKDIR /app

# Copy the source code into the container
COPY . .

# Build the Rust application in release mode
RUN cargo build --release

# Use a smaller base image for the final stage
#FROM debian:buster-slim
FROM ubuntu:latest

# Install dependencies for dlib, OpenCV
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    libdlib-data \
    libdlib19.1t64 \
    libopencv-imgcodecs406t64 \
    libopencv-imgproc406t64 \
    libopencv-core406t64 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/smugmug-syncer .

# Set the default command to run the application
ENTRYPOINT ["./smugmug-syncer"]