FROM python:3.10

# prepare user "app" and home directory "/app"
RUN mkdir /app
RUN useradd --home-dir /app app
RUN chown app:app /app
WORKDIR /app
USER app
ENV PATH="${PATH}:/app/.local/bin"

# install
RUN pip install --user poetry
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install --no-root
COPY . .
RUN poetry install --only-root

# run
EXPOSE 8888
CMD poetry run jupyter notebook --ServerApp.ip=0.0.0.0
