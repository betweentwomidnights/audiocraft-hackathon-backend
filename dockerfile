# Use Python image
FROM python:3.9

# Create app directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy other project files
COPY . .

# Your default command here, e.g.,
CMD [ "python", "your_script.py" ]
