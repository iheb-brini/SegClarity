FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# JupyterLab
RUN pip install jupyter notebook
RUN pip install jupyterlab ipywidgets

RUN pip uninstall -y opencv-python opencv opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
RUN pip install --no-cache-dir "opencv-contrib-python==4.8.0.74"

EXPOSE 8888

# Copy only code (datasets/models excluded via .dockerignore)
COPY Modules ./Modules
COPY Notebooks ./Notebooks
COPY README.md LICENSE ./

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
