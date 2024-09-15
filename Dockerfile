# Use an official Python 3.12 runtime as the base image
FROM python:3.12-slim

# Install system dependencies for h5py and other Python packages
RUN apt-get update || true \
    && apt-get install -y apt-transport-https ca-certificates \
    && apt-get update \
    && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    build-essential \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8080 (required by Google Cloud Run)
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "app.py"]
