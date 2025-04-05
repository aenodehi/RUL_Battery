FROM python:3.12-slim

# Switch to root for installations
USER root

# Set working directory
WORKDIR /home/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    wget \
    git \
    libnss3 \
    libatk1.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pipenv globally
RUN pip3 install --no-cache-dir --upgrade pip pipenv

# Create and activate a Python virtual environment
RUN python3 -m venv /home/workspace/venv && \
    /home/workspace/venv/bin/pip install --no-cache-dir --upgrade pip

# Add the virtual environment to the PATH
ENV PATH="/home/workspace/venv/bin:$PATH"

# Install VSCode server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Copy requirements and install dependencies
COPY requirements.txt /home/workspace/
RUN pipenv install --dev --ignore-pipfile

# Copy the application code
COPY . /home/workspace/

# Expose the port for the VSCode server
EXPOSE 3000

# Start the VSCode server
CMD ["code-server", "--bind-addr", "0.0.0.0:3000", "--auth", "none", "/home/workspace"]
