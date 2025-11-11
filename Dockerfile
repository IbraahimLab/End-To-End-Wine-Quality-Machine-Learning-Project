# Base image
FROM python:3.10-slim-buster

# Create app directory
WORKDIR /app

# Copy everything
COPY . /app

# Upgrade build tools
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Directly install the wine_quality package (not editable mode)
RUN pip install ./wine_quality

# Add /app to PYTHONPATH (so Python always sees it)
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8080

CMD ["python3", "app.py"]
