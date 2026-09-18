# House Price + Buyer Classification System (R + Flask)

This project is an integrated Machine Learning system. It predicts house prices, classifies them into buyer segments, and groups similar houses using a customizable dendrogram.

## Architecture
- **Machine Learning**: R (Models, Data Generation, Preprocessing)
- **Web Interface**: Python (Flask + HTML/CSS for a minimal look)

## Setup Instructions

### 1. R Environment
Make sure you have R installed on your system.
Run the data generator to create `house_data.csv`:
```bash
Rscript generate_data.R
```

### 2. Python Environment (Flask)
Install the Flask requirements:
```bash
pip install -r requirements.txt
```

Run the web app:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.
