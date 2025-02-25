from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Загрузка обученной модели
model_regression = joblib.load('model_regression.joblib')

model_tree = joblib.load('model_tree.joblib')

@app.route('/predict', methods=['POST'])   #:/predict
def predict():
    try:     
        # Получаем данные для предсказания из POST запроса
        data = request.get_json(force=True)
        predict_request = [data['Pclass'],data['Sex'],data['Age'],data['SibSp'],data['Parch'],data['Fare'],data['Embarked']]    
        
        # Делаем предсказание с помощью модели
        
        prediction_regression_proba = (model_regression.predict_proba([predict_request])[0])[0]
        prediction_tree_proba = (model_tree.predict_proba([predict_request])[0])[0]
        #return jsonify({'Pclass':prediction_regression, 'Sex':data['Sex']})
        return jsonify({'prediction_tree':  prediction_tree_proba, '   prediction_regression ': prediction_regression_proba})        
        
        
    except Exception as ex:
        return jsonify({'error': type(ex).__name__, 'args': ex.args})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

