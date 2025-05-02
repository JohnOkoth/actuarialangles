# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into container
COPY . .

# Ensure Streamlit doesn’t try to open a browser (important for cloud environments)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit's default port
EXPOSE 8501

# Run your Streamlit app
CMD ["streamlit", "run", "enhancing.py"]
