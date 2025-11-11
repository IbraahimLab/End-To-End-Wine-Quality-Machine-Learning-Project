FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

# Ensure build tools are present
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Force local package installation (not editable)
RUN pip install .

EXPOSE 8080

CMD ["python3", "app.py"]
