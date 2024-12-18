# Use the official Python image as a parent image
FROM python:3.9-slim

# Create a directory for the app
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requierments.txt .
RUN pip install --no-cache-dir -r requierments.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Set the default command to bash
CMD ["python server.py"]