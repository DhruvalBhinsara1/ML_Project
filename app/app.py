import os
import subprocess
from flask import Flask, render_template, request, jsonify

# Compute absolute paths to project directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


def get_r_script_path(script_name: str) -> str:
    """Find script in SRC or src directory."""
    src_upper = os.path.join(PROJECT_ROOT, 'SRC', script_name)
    src_lower = os.path.join(PROJECT_ROOT, 'src', script_name)
    if os.path.exists(src_upper):
        return src_upper
    if os.path.exists(src_lower):
        return src_lower
    return src_upper


def format_inr_price(lakhs_val: float) -> str:
    """Normalize price in Lakhs into clean Crores, Lakhs, or Thousands."""
    if lakhs_val >= 100.0:
        crores = lakhs_val / 100.0
        return f"₹{crores:.2f} Crores"
    elif lakhs_val < 1.0:
        thousands = lakhs_val * 100.0
        return f"₹{thousands:.2f} Thousands"
    else:
        return f"₹{lakhs_val:.2f} Lakhs"


@app.route('/')
def home():
    return render_template('home.html', active_page='home')


@app.route('/regression')
def regression():
    return render_template('regression.html', active_page='regression')


@app.route('/classification')
def classification():
    return render_template('classification.html', active_page='classification')


@app.route('/clustering')
def clustering():
    return render_template('clustering.html', active_page='clustering')

@app.route('/recommendation')
def recommendation():
    return render_template(
        'recommendation.html',
        active_page='recommendation'
    )


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        size = request.form.get('size', '1500')
        bhk = request.form.get('bhk', '3')
        age = request.form.get('age', '5')
        city = request.form.get('city', 'Pune')

        script_path = get_r_script_path('predict_regression.R')
        result = subprocess.run(
            ['Rscript', script_path, str(size), str(bhk), str(age), str(city)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"})

        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line for line in output_lines if line.startswith('RESULT:')), None)

        if pred_line:
            pred_raw = pred_line.replace('RESULT:', '').strip()
            try:
                lakhs_val = float(pred_raw)
                formatted_price = format_inr_price(lakhs_val)
            except ValueError:
                lakhs_val = None
                formatted_price = f"₹{pred_raw} Lakhs"
            prediction = f"Predicted Price: {formatted_price} (Linear Regression)"
            return jsonify({
                'prediction': prediction,
                'formatted_price': formatted_price,
                'raw_lakhs': lakhs_val,
                'size': size,
                'bhk': bhk,
                'age': age,
                'city': city
            })
        else:
            prediction = "Error: Could not parse prediction from R output."

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        size = request.form.get('size', '1500')
        bhk = request.form.get('bhk', '3')
        age = request.form.get('age', '5')
        city = request.form.get('city', 'Pune')

        script_path = get_r_script_path('predict_classification.R')
        result = subprocess.run(
            ['Rscript', script_path, str(size), str(bhk), str(age), str(city)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"})

        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line for line in output_lines if line.startswith('RESULT:')), None)

        if pred_line:
            preds = pred_line.replace('RESULT:', '').strip().split('|')
            dt_pred = preds[0].strip() if len(preds) > 0 else "N/A"
            rf_pred = preds[1].strip() if len(preds) > 1 else "N/A"
            prediction = f"Decision Tree: {dt_pred} | Random Forest: {rf_pred}"
            return jsonify({
                'prediction': prediction,
                'dt_prediction': dt_pred,
                'rf_prediction': rf_pred,
                'size': size,
                'bhk': bhk,
                'age': age,
                'city': city,
                'image_url': '/static/decision_tree.png'
            })
        else:
            prediction = "Error: Could not parse prediction from R output."

        return jsonify({'prediction': prediction, 'image_url': '/static/decision_tree.png'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/generate_dendrogram', methods=['POST'])
def generate_dendrogram():
    try:
        linkage = request.form.get('linkage', 'ward')
        clusters = request.form.get('clusters', '3')

        script_path = get_r_script_path('predict_dendrogram.R')
        result = subprocess.run(
            ['Rscript', script_path, str(linkage), str(clusters)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"})

        output_lines = result.stdout.strip().split('\n')
        res_line = next((line for line in output_lines if line.startswith('RESULT:')), None)

        if res_line:
            image_path = res_line.replace('RESULT:', '').strip()
        else:
            image_path = 'static/dendrogram.png'

        clean_path = image_path.lstrip('/')
        return jsonify({
            'image_url': f"/{clean_path}",
            'linkage': linkage,
            'clusters': clusters
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_knn', methods=['POST'])
@app.route('/predict_recommendation', methods=['POST'])
def predict_knn():
    try:
        bhk = request.form.get('bhk', '3')
        size = request.form.get('size', '1800')
        year_built = request.form.get('year_built', '2020')
        floor_no = request.form.get('floor_no', '3')
        total_floors = request.form.get('total_floors', '10')
        age = request.form.get('age', '5')
        nearby_schools = request.form.get('nearby_schools', '5')
        nearby_hospitals = request.form.get('nearby_hospitals', '3')

        script_path = get_r_script_path('predict_knn.R')
        result = subprocess.run(
            [
                'Rscript', script_path,
                str(bhk), str(size), str(year_built), str(floor_no),
                str(total_floors), str(age), str(nearby_schools), str(nearby_hospitals)
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"})

        output_lines = result.stdout.strip().split('\n')
        res_line = next((line for line in output_lines if line.startswith('RESULT:')), None)

        if res_line:
            parts = [p.strip() for p in res_line.replace('RESULT:', '').split('|')]
            category = parts[0] if len(parts) > 0 else "Unknown"
            budget_prob = parts[1] if len(parts) > 1 else "0.0"
            mid_prob = parts[2] if len(parts) > 2 else "0.0"
            premium_prob = parts[3] if len(parts) > 3 else "0.0"
            best_k = parts[4] if len(parts) > 4 else "11"

            return jsonify({
                'category': category,
                'budget_prob': f"{budget_prob}%",
                'mid_prob': f"{mid_prob}%",
                'premium_prob': f"{premium_prob}%",
                'best_k': best_k,
                'image_url': '/static/knn_accuracy_vs_k.png',
                'bhk': bhk,
                'size': size,
                'year_built': year_built,
                'floor_no': floor_no,
                'total_floors': total_floors,
                'age': age,
                'nearby_schools': nearby_schools,
                'nearby_hospitals': nearby_hospitals
            })
        else:
            return jsonify({'error': "Could not parse prediction from R output."})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f" * Server running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)


