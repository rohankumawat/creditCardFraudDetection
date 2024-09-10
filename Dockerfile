# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port (if you're running a web service or dashboard, adjust as needed)
EXPOSE 5000

# Run the model training script or any other starting point
CMD ["python", "src/train.py"]