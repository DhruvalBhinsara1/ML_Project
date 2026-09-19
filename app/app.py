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


def get_param(key: str, default=None):
    """Retrieve parameter from request, supporting both JSON payloads and form data."""
    if request.is_json:
        json_data = request.get_json(silent=True) or {}
        if key in json_data:
            return json_data[key]
    val = request.form.get(key)
    if val is not None:
        return val
    return default


def parse_numeric(val, field_name: str, min_val=None, max_val=None, is_int=False):
    """Validate and convert numeric input with optional range constraints."""
    try:
        converted = int(val) if is_int else float(val)
    except (ValueError, TypeError):
        raise ValueError(f"'{field_name}' must be a valid {'integer' if is_int else 'number'}.")
    
    if min_val is not None and converted < min_val:
        raise ValueError(f"'{field_name}' must be greater than or equal to {min_val}.")
    if max_val is not None and converted > max_val:
        raise ValueError(f"'{field_name}' must be less than or equal to {max_val}.")
    return converted


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
    return render_template('recommendation.html', active_page='recommendation')


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        size = get_param('size', '1500')
        bhk = get_param('bhk', '3')
        age = get_param('age', '5')
        city = str(get_param('city', 'Pune')).strip()

        # Validate numeric inputs
        try:
            val_size = parse_numeric(size, 'Size', min_val=50, max_val=50000)
            val_bhk = parse_numeric(bhk, 'BHK', min_val=1, max_val=30, is_int=True)
            val_age = parse_numeric(age, 'Age', min_val=0, max_val=200)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        script_path = get_r_script_path('predict_regression.R')
        result = subprocess.run(
            ['Rscript', script_path, str(val_size), str(val_bhk), str(val_age), city],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"}), 500

        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line.strip() for line in output_lines if line.strip().startswith('RESULT:')), None)

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
                'size': val_size,
                'bhk': val_bhk,
                'age': val_age,
                'city': city
            })
        else:
            return jsonify({'error': "Could not parse prediction from R output."}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        size = get_param('size', '1500')
        bhk = get_param('bhk', '3')
        age = get_param('age', '5')
        city = str(get_param('city', 'Pune')).strip()

        # Validate numeric inputs
        try:
            val_size = parse_numeric(size, 'Size', min_val=50, max_val=50000)
            val_bhk = parse_numeric(bhk, 'BHK', min_val=1, max_val=30, is_int=True)
            val_age = parse_numeric(age, 'Age', min_val=0, max_val=200)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        script_path = get_r_script_path('predict_classification.R')
        result = subprocess.run(
            ['Rscript', script_path, str(val_size), str(val_bhk), str(val_age), city],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"}), 500

        output_lines = result.stdout.strip().split('\n')
        pred_line = next((line.strip() for line in output_lines if line.strip().startswith('RESULT:')), None)

        if pred_line:
            preds = [p.strip() for p in pred_line.replace('RESULT:', '').strip().split('|')]
            dt_pred = preds[0] if len(preds) > 0 else "N/A"
            rf_pred = preds[1] if len(preds) > 1 else "N/A"
            prediction = f"Decision Tree: {dt_pred} | Random Forest: {rf_pred}"
            return jsonify({
                'prediction': prediction,
                'dt_prediction': dt_pred,
                'rf_prediction': rf_pred,
                'size': val_size,
                'bhk': val_bhk,
                'age': val_age,
                'city': city,
                'image_url': '/static/decision_tree.png'
            })
        else:
            return jsonify({'error': "Could not parse prediction from R output."}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_dendrogram', methods=['POST'])
def generate_dendrogram():
    try:
        linkage = str(get_param('linkage', 'ward')).strip().lower()
        clusters = get_param('clusters', '3')

        valid_linkages = {'ward', 'complete', 'average', 'single'}
        if linkage not in valid_linkages:
            linkage = 'ward'

        try:
            val_clusters = parse_numeric(clusters, 'Clusters', min_val=2, max_val=15, is_int=True)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        script_path = get_r_script_path('predict_dendrogram.R')
        result = subprocess.run(
            ['Rscript', script_path, linkage, str(val_clusters)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"}), 500

        output_lines = result.stdout.strip().split('\n')
        res_line = next((line.strip() for line in output_lines if line.strip().startswith('RESULT:')), None)

        if res_line:
            image_path = res_line.replace('RESULT:', '').strip()
        else:
            image_path = 'static/dendrogram.png'

        clean_path = image_path.lstrip('/')
        return jsonify({
            'image_url': f"/{clean_path}",
            'linkage': linkage,
            'clusters': val_clusters
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_knn', methods=['POST'])
@app.route('/predict_recommendation', methods=['POST'])
def predict_knn():
    try:
        bhk = get_param('bhk', '3')
        size = get_param('size', '1800')
        year_built = get_param('year_built', '2020')
        floor_no = get_param('floor_no', '3')
        total_floors = get_param('total_floors', '10')
        age = get_param('age', '5')
        nearby_schools = get_param('nearby_schools', '5')
        nearby_hospitals = get_param('nearby_hospitals', '3')

        # Validate all numeric inputs
        try:
            val_bhk = parse_numeric(bhk, 'BHK', min_val=1, max_val=30, is_int=True)
            val_size = parse_numeric(size, 'Size', min_val=50, max_val=50000)
            val_year = parse_numeric(year_built, 'Year Built', min_val=1800, max_val=2100, is_int=True)
            val_floor = parse_numeric(floor_no, 'Floor Number', min_val=0, max_val=200, is_int=True)
            val_total_floors = parse_numeric(total_floors, 'Total Floors', min_val=1, max_val=250, is_int=True)
            val_age = parse_numeric(age, 'Age', min_val=0, max_val=200)
            val_schools = parse_numeric(nearby_schools, 'Nearby Schools', min_val=0, max_val=100, is_int=True)
            val_hospitals = parse_numeric(nearby_hospitals, 'Nearby Hospitals', min_val=0, max_val=100, is_int=True)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        script_path = get_r_script_path('predict_knn.R')
        result = subprocess.run(
            [
                'Rscript', script_path,
                str(val_bhk), str(val_size), str(val_year), str(val_floor),
                str(val_total_floors), str(val_age), str(val_schools), str(val_hospitals)
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            return jsonify({'error': f"R Error: {result.stderr.strip() or result.stdout.strip()}"}), 500

        output_lines = result.stdout.strip().split('\n')
        res_line = next((line.strip() for line in output_lines if line.strip().startswith('RESULT:')), None)

        if res_line:
            parts = [p.strip() for p in res_line.replace('RESULT:', '').split('|')]
            category = parts[0] if len(parts) > 0 else "Unknown"

            def safe_pct(part_idx: int) -> float:
                if len(parts) > part_idx:
                    try:
                        return float(parts[part_idx])
                    except ValueError:
                        pass
                return 0.0

            budget_pct = safe_pct(1)
            mid_pct = safe_pct(2)
            premium_pct = safe_pct(3)
            best_k = parts[4] if len(parts) > 4 else "11"

            return jsonify({
                'category': category,
                'budget_prob': f"{budget_pct:.1f}%",
                'mid_prob': f"{mid_pct:.1f}%",
                'premium_prob': f"{premium_pct:.1f}%",
                'budget_pct': budget_pct,
                'mid_pct': mid_pct,
                'premium_pct': premium_pct,
                'best_k': best_k,
                'image_url': '/static/knn_accuracy_vs_k.png',
                'bhk': val_bhk,
                'size': val_size,
                'year_built': val_year,
                'floor_no': val_floor,
                'total_floors': val_total_floors,
                'age': val_age,
                'nearby_schools': val_schools,
                'nearby_hospitals': val_hospitals
            })
        else:
            return jsonify({'error': "Could not parse prediction from R output."}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f" * Server running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)


