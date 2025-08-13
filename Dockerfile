FROM alpine:3.22.1

WORKDIR /app

COPY . .

RUN apk update && apk upgrade && \
    apk add --no-cache python3 py3-pip build-base python3-dev && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt 

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python3", "main.py"]

