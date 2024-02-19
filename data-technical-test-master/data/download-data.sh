#!/bin/bash

echo "Downloading data for database..."

mkdir -p data

# Download decisions data
curl https://s3.eu-central-1.amazonaws.com/data-technical-test/CIV.zip --output data/CIV.csv.zip ; unzip data/CIV.csv.zip -d data/
curl https://s3.eu-central-1.amazonaws.com/data-technical-test/SOC.zip --output data/SOC.csv.zip ; unzip data/SOC.csv.zip -d data/
curl https://s3.eu-central-1.amazonaws.com/data-technical-test/COM.zip --output data/COM.csv.zip ; unzip data/COM.csv.zip -d data/
curl https://s3.eu-central-1.amazonaws.com/data-technical-test/CRIM.zip --output data/CRIM.csv.zip ; unzip data/CRIM.csv.zip -d data/
