FROM rocker/r-ver:4.2.0

RUN apt-get update && apt-get install -y \
    libssl-dev \
    libcurl4-gnutls-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libcairo2-dev

COPY . /home/rstudio/

RUN Rscript /home/rstudio/install.R

EXPOSE 3838

CMD ["R", "-e", "shiny::runApp('/home/rstudio/', host = '0.0.0.0', port = 3838)"]