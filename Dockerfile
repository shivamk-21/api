FROM python:3.7.9

RUN python -m pip install --upgrade pip
# Install production dependencies.
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image.
WORKDIR /app
COPY . .

ENV PORT 80
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 0 server:app