import os
import sys
import unittest

# Ensure app package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from app import app


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    # --- GET Page Routes ---
    def test_home_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'House Predictor', response.data)
        self.assertIn(b'Property Recommendation', response.data)

    def test_regression_route(self):
        response = self.client.get('/regression')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Linear Regression Price Predictor', response.data)

    def test_classification_route(self):
        response = self.client.get('/classification')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Segment Classification', response.data)

    def test_clustering_route(self):
        response = self.client.get('/clustering')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hierarchical Clustering', response.data)

    def test_recommendation_route(self):
        response = self.client.get('/recommendation')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'KNN Property Recommendation', response.data)

    def test_about_route(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About House Predictor', response.data)

    # --- POST Linear Regression ---
    def test_predict_regression_form(self):
        response = self.client.post('/predict_regression', data={
            'size': '1800',
            'bhk': '3',
            'age': '5',
            'city': 'Pune'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('formatted_price', data)
        self.assertIn('raw_lakhs', data)
        self.assertGreater(data['raw_lakhs'], 0)

    def test_predict_regression_json(self):
        response = self.client.post('/predict_regression', json={
            'size': 2200,
            'bhk': 4,
            'age': 2,
            'city': 'Mumbai'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('formatted_price', data)
        self.assertEqual(data['city'], 'Mumbai')

    def test_predict_regression_invalid(self):
        response = self.client.post('/predict_regression', json={
            'size': 'not-a-number',
            'bhk': 3
        })
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    # --- POST Classification ---
    def test_predict_classification_form(self):
        response = self.client.post('/predict_classification', data={
            'size': '1800',
            'bhk': '3',
            'age': '5',
            'city': 'Pune'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('dt_prediction', data)
        self.assertIn('rf_prediction', data)
        self.assertIn('image_url', data)

    def test_predict_classification_json(self):
        response = self.client.post('/predict_classification', json={
            'size': 1200,
            'bhk': 2,
            'age': 8,
            'city': 'Bangalore'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('dt_prediction', data)
        self.assertIn('rf_prediction', data)

    def test_predict_classification_invalid(self):
        response = self.client.post('/predict_classification', json={
            'size': 1500,
            'bhk': 'abc'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    # --- POST Clustering Dendrogram ---
    def test_generate_dendrogram_form(self):
        response = self.client.post('/generate_dendrogram', data={
            'linkage': 'ward',
            'clusters': '3'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('image_url', data)
        self.assertEqual(data['linkage'], 'ward')

    def test_generate_dendrogram_json(self):
        response = self.client.post('/generate_dendrogram', json={
            'linkage': 'complete',
            'clusters': 4
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('image_url', data)
        self.assertEqual(data['clusters'], 4)

    def test_generate_dendrogram_invalid(self):
        response = self.client.post('/generate_dendrogram', json={
            'linkage': 'ward',
            'clusters': 'invalid'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    # --- POST KNN Recommendation ---
    def test_predict_knn_form(self):
        response = self.client.post('/predict_knn', data={
            'bhk': '3',
            'size': '1800',
            'year_built': '2020',
            'floor_no': '3',
            'total_floors': '10',
            'age': '5',
            'nearby_schools': '5',
            'nearby_hospitals': '3'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('category', data)
        self.assertIn('budget_prob', data)
        self.assertIn('mid_prob', data)
        self.assertIn('premium_prob', data)
        self.assertIn('budget_pct', data)
        self.assertIn('best_k', data)
        self.assertIn('image_url', data)

    def test_predict_knn_json(self):
        response = self.client.post('/predict_knn', json={
            'bhk': 4,
            'size': 2500,
            'year_built': 2022,
            'floor_no': 5,
            'total_floors': 15,
            'age': 2,
            'nearby_schools': 6,
            'nearby_hospitals': 4
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('category', data)
        self.assertEqual(data['category'], 'Mid-Range')

    def test_predict_knn_invalid(self):
        response = self.client.post('/predict_knn', json={
            'bhk': 'three',
            'size': 1800
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())


if __name__ == '__main__':
    unittest.main()
