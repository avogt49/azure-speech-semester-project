#!/bin/bash
apt-get update && apt-get install -y ffmpeg
gunicorn -k uvicorn.workers.UvicornWorker --bind=0.0.0.0 --timeout 300 main:app
