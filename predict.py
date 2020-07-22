import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = tf.keras.models.load_model('model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def Predict():
    input_data = [item for item in request.form.values()]
    if input_data[1]=='Male':
        input_data[1]=1
    else:
        input_data[1]=0
    if input_data[-2]=='No':
        input_data[-2]=0
    else:
        input_data[-2]=1
    # print(np.reshape(input_data,(-1,1)))
    order = [3, 2, -3, 4, -1, -2, -4, 1]
    input_data = [int(input_data[j]) for j in order] 
    data = scaler.transform([input_data])

    pred = model.predict(data)

    if pred > 0.5:
        return render_template('index.html', prediction_text='YOU ARE WORTHY')
    else:
        return render_template('index.html', prediction_text='YOU MORTAL. YOU ARE NOT WORTHY')


# def main():
#     predict_input_file = sys.argv[1]
#     predict_input = pd.read_csv(predict_input_file, header = None)
#     Predict(predict_input)



if __name__ == "__main__":
    app.run(debug=True)
