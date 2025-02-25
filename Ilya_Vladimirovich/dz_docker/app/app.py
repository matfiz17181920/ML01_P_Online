from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    # Получаем параметры из URL
    features_str = request.args.get('features')
    
    if not features_str:
        return jsonify({'error': 'Missing features parameter'}), 400
    
    try:
        features = list(map(float, features_str.split(',')))
        features_array = np.array(features).reshape(1, -1)
            
        prediction = model.predict(features_array)
        return jsonify({'prediction': int(prediction[0])})
    
    except ValueError:
        return jsonify({'error': 'Invalid features format'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)