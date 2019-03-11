

from flask import request
from flask import Flask
from flask import jsonify
from flask_cors import CORS


import pandas as pd
import numpy as np
import itertools
import pickle
from flask import Flask, render_template
from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np
#app = Flask('using bokeh')



app = Flask(__name__)
CORS(app)

test = pd.read_csv('test.csv')
test = test.drop(["id"], axis=1)



model_names = ["Catboost3", "Catboost_w", "Catboost", ]
for model_name in model_names:
     with open(f"{model_name}", "rb") as pfile:
        exec(f"{model_name} = pickle.load(pfile)")






CATEGORIES = ["Rejected", "Approved"]

def voting_classifer_r(X):

    result = []
    i = 0
    for i in range(X.shape[0]):
        A = Catboost3.predict_proba(X.iloc[[i]])[:, 1]
        B = Catboost_w.predict_proba(X.iloc[[i]])[:, 1]
        C = Catboost.predict_proba(X.iloc[[i]])[:, 1]

        score = np.round_((A + B + C) / 3)

        result.append(score)
        i+=1
    return result

def make_df(index):
    #cols = list(test)
    df = pd.DataFrame([index], columns=['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'])
    return df



@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    index = message['name']

    #gets the index location for the test set and than passing the dataframe with on row to the classifer
    response = (CATEGORIES[(int(np.max(voting_classifer_r(test.iloc[[int(index)]]))))])
    return jsonify(response)

@app.route('/predict2', methods=['POST'])
def predict2():
    message = request.get_json(force=True)
    index = message['name']


    print(index)

    z = make_df(index)

    response = (CATEGORIES[(int(np.max(voting_classifer_r(z))))])

    return jsonify(response)



@app.route('/plots')
def index():
    df = pd.read_csv('run_test-tag-AUC.csv')
    # prepare  data
    x = df.Step
    y = df.Value

    # output to static HTML file
    #output_file("lines.html")

    # create a new plot with a title and axis labels
    fig = figure(title="Catboost Score Progress", x_axis_label='Tree Steps',
               y_axis_label='ROC_AUC Score', tools="hover", tooltips="Tree @x:Score @y",plot_width=400, plot_height=400)
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None

    # add a line renderer with legend and line thickness
    #p.line(x, y, line_width=2)

    # show the results
    # show(p)


    #hist, edges = np.histogram(measured, density=True, bins=50)


    fig.line(x, y, line_width=4)
    javascript, div = components(fig)

    return render_template('index.html', javascript=javascript, div=div)
