FROM ghcr.io/fedlearnnet/python-tool-api/base-python-dep:0.7.12

USER 0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R nonroot:nonroot /app /opt/venv \
 && chmod -R u+rwX,g+rwX /app

USER nonroot:nonroot
CMD ["python", "-m", "main"]

