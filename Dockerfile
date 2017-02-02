FROM python:3.6-alpine
MAINTAINER Ivan Miniailenko <discrete.west@gmail.com>

RUN pip3 install \
    jupyter \
    matplotlib \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    seaborn \
    sqlalchemy \
    ujson

RUN jupyter notebook --generate-config && \
    sed -i "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = ''/" ~/.jupyter/jupyter_notebook_config.py

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--notebook-dir=/notebooks"]