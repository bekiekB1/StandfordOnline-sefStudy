# Source from fastapi - Doceker+Poetry
# Use a Python 3.10 slim base image as the requirements stage
FROM python:3.10-slim as requirements-stage

# Set working directory
WORKDIR /tmp

# Install Poetry
RUN pip install poetry

# Copy poetry files for dependencies resolution
COPY ./pyproject.toml ./poetry.lock* /tmp/

# # Resolve dependencies and export them to requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --with cs224n --with dev --without-hashes

# # Use a new Python 3.9 slim base image for the final stage
FROM python:3.10-slim as final-stage

# Set working directory
WORKDIR /code

COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

# Install additional system dependencies for JupyterLab
RUN apt-get update && \
    apt-get install -y make && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean


# Install dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Install JupyterLab
RUN pip install jupyterlab

# Copy the rest of the application code
COPY ./cs224n /code/

EXPOSE 8888

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
