FROM kaggle/python

WORKDIR /root/
RUN git clone https://github.com/flowlight0/talkingdata-adtracking-fraud-detection.git

WORKDIR /root/talkingdata-adtracking-fraud-detection
RUN apt-get install awscli -y
RUN pip install --upgrade pip
RUN pip install --upgrade awscli
RUN pip install kaggle
RUN conda install arrow-cpp=0.9.* -c conda-forge
RUN conda install numba
