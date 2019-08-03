from flask import Flask, render_template, url_for, request
from flask_material import Material

# EDA Pkgs

import pandas as pd
import numpy as np

# ML Pkg
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preview')
def preview():
    df = pd.read_csv("data/Dataset.csv")
    return render_template('preview.html', df_view=df)


@app.route('/', methods=["POST"])
def analyze():

    if request.method == 'POST':
        rooms_per_dwelling = request.form['rooms_per_dwelling']
        lower_status = request.form['lower_status']
        pupil_teacher_ratio = request.form['pupil_teacher_ratio']
        linear_regression = request.form['linear_regression']

        # Clean the data by convert from unicode to float
        sample_data = [rooms_per_dwelling, lower_status, pupil_teacher_ratio]
        clean_data = [float(i) for i in sample_data]

        # reshape the data as a sample not individual features
        client_data = np.array(clean_data).reshape(1, -1)
        model = joblib.load('data/lin_reg_model.joblib')
        result_prediction = model.predict(client_data)

    return render_template('index.html', rooms_per_dwelling=rooms_per_dwelling,
                           lower_status=lower_status,
                           pupil_teacher_ratio=pupil_teacher_ratio,
                           clean_data=clean_data,
                           result_prediction=int(result_prediction),
                           model_selected=linear_regression
                           )


if __name__ == '__main__':
    app.run(debug=True)
