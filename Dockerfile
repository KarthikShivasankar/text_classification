FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Python won’t try to write .pyc files on the import of source modules and won't buffer stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /usr/src/app
WORKDIR /usr/src/app

# Install necessary packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    wget \
    python3.9 \
    python3-pip \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    libxft-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip to its latest version
RUN pip3 install --no-cache-dir --upgrade pip

# Set the working directory to /code for the application code
WORKDIR /code

# Copy the requirements.txt file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Create a non-root user for running the application
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables for the non-root user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1

# Set the working directory to the user's home directory/app
WORKDIR $HOME/app

# Copy the application code into the container, setting the owner to the non-root user
COPY --chown=user . $HOME/app

# Default command to run the application
CMD ["python3", "app.py"]

#docker build -t finetune .
#docker run --gpus all -p 7077:7077 finetune
#ft_td 