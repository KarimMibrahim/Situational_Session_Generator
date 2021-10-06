FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/Situational_Session_Generator.git /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo


# install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /src_code/repo
ENTRYPOINT ["bash"]

