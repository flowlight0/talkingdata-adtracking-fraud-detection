FROM kaggle/python

WORKDIR /root/
ARG ACCESS_TOKEN
RUN git clone https://${ACCESS_TOKEN}:x-oauth-basic@github.com/flowlight0/talkingdata-adtracking-fraud-detection.git

WORKDIR /root/talkingdata-adtracking-fraud-detection
RUN apt-get install awscli -y
RUN pip install --upgrade pip
RUN pip install --upgrade awscli
RUN pip install kaggle
RUN conda install arrow-cpp=0.9.* -c conda-forge
RUN conda install numba

# setup jupyter notebook configuration
RUN pip install --upgrade jupyter
RUN jupyter-notebook --generate-config --allow-root
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
ARG JUPYTER_PASSWORD_HASH
RUN echo "c.NotebookApp.password = \"${JUPYTER_PASSWORD_HASH}\"" >> /root/.jupyter/jupyter_notebook_config.py
