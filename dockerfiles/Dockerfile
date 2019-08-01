FROM python:3-onbuild
RUN pip3 install --upgrade pip

RUN  git clone https://git.asi.ru/remembername/datahacksd.git /API

EXPOSE 3333

ENTRYPOINT ["python3", "app.py"]