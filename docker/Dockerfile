FROM rayproject/ray-ml:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV RAY_RESULTS_PATH=/home/ray/project/ray_results
ENV PYTHONWARNINGS="ignore::DeprecationWarning"

RUN pip install --upgrade pip setuptools
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt

# supersuit needs to be installed after for some reason
RUN pip install supersuit==3.9.0

RUN git config --global --add safe.directory /home/ray

ENV PYTHONPATH=/home/ray/project:$PYTHONPATH

# RUN sudo chmod 0777 /home/ray/project

# LaTeX for matplotlib
# RUN apt-get install texlive-binaries texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y
# RUN pip install latex
