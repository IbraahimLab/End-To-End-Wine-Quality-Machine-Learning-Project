# Base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 👇 Make sure your package is installed in the container
RUN pip install -e .

# Expose Flask/FastAPI port
EXPOSE 8080

# Run the application
CMD ["python3", "app.py"]
