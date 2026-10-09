FROM gitlab.cosy.bio:5050/cosybio/federated-learning/federated_db/pyfedappwrap/base-python-dep:0.7.11

USER 0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R nonroot:nonroot /app /opt/venv \
 && chmod -R u+rwX,g+rwX /app

USER nonroot:nonroot
CMD ["python", "-m", "main"]

