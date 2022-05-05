import numpy as np
from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)
model = load('model.joblib')

@app.route("/")
def interface():
    return render_template('interface.html')

@app.route('/predict', methods = ['POST'])
def process():
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    patal_length = request.form['patal_length']
    patal_width = request.form['patal_width']
    input_values = np.array([[
        sepal_length,
        sepal_width,
        patal_length,
        patal_width
    ]])
    # processing the inputs to account for missing values and errors
    for idx, n in enumerate(input_values):
        try:
            float(n)
        except:
            input_values[idx] = 0
    output_value = model.predict(input_values)
    return render_template('result.html', data = output_value)

if __name__ == '__main__':
    app.run(debug = True)
