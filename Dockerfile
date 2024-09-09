FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Install additional cloud-related packages and requirements
RUN pip install --no-cache-dir google-cloud-monitoring \
                               google-cloud-logging \
                               firebase-admin \
                               streamlit \
                               psutil \
                               sqlalchemy \
                               pymysql \
                               torch \
                               transformers

COPY . .

EXPOSE 8080

CMD ["poetry", "run", "streamlit", "run", "--server.port", "8080", "--server.headless", "true", "--server.address", "0.0.0.0", "app.py"]