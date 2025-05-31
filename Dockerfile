# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data to a specific directory
RUN mkdir -p /usr/share/nltk_data && \
    python -c "import nltk; \
    nltk.download('punkt', download_dir='/usr/share/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/usr/share/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/share/nltk_data'); \
    nltk.download('wordnet', download_dir='/usr/share/nltk_data')"

# Set the NLTK_DATA environment variable to point to the directory where NLTK data is stored
ENV NLTK_DATA=/usr/share/nltk_data

# Copy the rest of the application
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]