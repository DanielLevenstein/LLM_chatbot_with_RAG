#!/bin/bash
rm -rf data
rm config/features_current.json
rm config/features_downloaded.json
cp config/features_default.json config/features_current.json
mkdir data
python3 extract.py
python3 ingest.py