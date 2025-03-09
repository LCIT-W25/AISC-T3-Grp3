# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from your project into the container
COPY . /app

# Install dependencies. Ensure you have a requirements.txt in your repo.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit uses (default is 8501)
EXPOSE 8501

# Set the entrypoint to run your Streamlit app.
CMD ["streamlit", "run", "Deploy.py", "--server.enableCORS=false", "--server.port=8501"]
