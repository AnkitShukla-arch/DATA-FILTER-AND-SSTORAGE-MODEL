# Hackathon Data Filter

This project filters, cleans, and prepares raw structured data for storage in star/snowflake schemas.  
Built for OpenAI Hackathon 🚀

## Features
- Handles large CSVs (up to ~5GB) with chunked processing
- Cleans data (nulls, types, duplicates)
- Detects anomalies using IsolationForest
- Stores output in Parquet format for downstream warehouses
- Configurable rules via `config.yaml`

## Setup
```bash
git clone https://github.com/<your-username>/hackathon-data-filter.git
cd hackathon-data-filter
pip install -r requirements.txt
python filter.py --input ./data/input.csv --output ./data/curated/ --config config.yaml

