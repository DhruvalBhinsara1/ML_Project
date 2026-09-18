from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

import subprocess

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        size = request.form.get('size')
        bhk = request.form.get('bhk')
        age = request.form.get('age')
        city = request.form.get('city', 'Pune')

        result = subprocess.run(
            ['Rscript', 'src/predict_regression.R', size, bhk, age, city],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr}"})
            
        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line for line in output_lines if line.startswith('RESULT:')), None)
        
        if pred_line:
            pred_value = pred_line.replace('RESULT:', '').strip()
            prediction = f"Predicted Price: ₹{pred_value} Lakhs (Linear Regression)"
        else:
            prediction = "Error: Could not parse prediction from R output."

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        size = request.form.get('size')
        bhk = request.form.get('bhk')
        age = request.form.get('age')
        city = request.form.get('city', 'Pune')
        
        result = subprocess.run(
            ['Rscript', 'src/predict_classification.R', size, bhk, age, city],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr}"})
            
        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line for line in output_lines if line.startswith('RESULT:')), None)
        
        if pred_line:
            preds = pred_line.replace('RESULT:', '').strip().split('|')
            dt_pred = preds[0].strip() if len(preds) > 0 else "N/A"
            rf_pred = preds[1].strip() if len(preds) > 1 else "N/A"
            prediction = f"Decision Tree: {dt_pred} | Random Forest: {rf_pred}"
        else:
            prediction = "Error: Could not parse prediction from R output."
            
        return jsonify({'prediction': prediction, 'image_url': 'static/decision_tree.png'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/generate_dendrogram', methods=['POST'])
def generate_dendrogram():
    try:
        linkage = request.form.get('linkage')
        clusters = request.form.get('clusters')
        
        result = subprocess.run(
            ['Rscript', 'src/predict_dendrogram.R', linkage, clusters],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr}"})
            
        image_path = result.stdout.strip().split('\n')[-1]
        return jsonify({'image_url': f"/{image_path}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

