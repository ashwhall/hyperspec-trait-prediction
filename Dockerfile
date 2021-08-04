FROM anibali/pytorch:1.4.0-cuda10.1

# Install python dependencies
COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt

CMD ["/bin/bash"]
