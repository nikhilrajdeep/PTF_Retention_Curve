# Container image: runs anywhere (Hugging Face Spaces, Render, Google Cloud Run,
# a lab server).  docker build -t ptf-explorer . && docker run -p 8501:8501 ptf-explorer
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# compile the Numba kernels at build time so the first user waits for nothing
RUN python -c "import numpy as np; from ptf_engine import core; \
    [core.Ensemble.load('models/ptf_model_bundle', f, 'SSC').predict_area(np.array([[90.,5.,5.,5.]]), [1.], [0., 1.], [2., 4.2]) for f in core.MODEL_FAMILIES]"
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
