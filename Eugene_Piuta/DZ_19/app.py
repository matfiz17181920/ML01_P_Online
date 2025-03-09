from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Загрузка обученной модели
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])   #:/predict
def predict():
    # Получаем данные для предсказания из POST запроса
    data = request.get_json(force=True)
    predict_request = [data['gender'],data['age'],data['marital_status'],data['job_position'],data['credit_sum'],data['credit_month'],data['tariff_id'],data['score_shk'],data['education'],data['living_region'],data['monthly_income'],data['credit_count'],data['overdue_credit_count']]
    
    # Делаем предсказание с помощью модели
    prediction = model.predict([predict_request])[0]

    # Возвращаем результат
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
