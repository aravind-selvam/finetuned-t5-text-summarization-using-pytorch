from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

from ts.pipeline.prediction_pipeline import SinglePrediction
from ts.pipeline.training_pipeline import TrainingPipeline

# Creating a Flask object.
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['POST'])
@cross_origin()
def train():
    """
    The function `train()` runs the training pipeline.
    
    Returns:
      The index.html page is being returned.
    """
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def getsummary():
    """
    1. The function gets the input text from the user.
    2. It then calls the predict function of the SinglePrediction class.
    3. The predict function returns the summary.
    4. The summary is then rendered in the summary.html file
    
    Returns:
      The result of the prediction is being returned.
    """
    input_text = request.form['data']
    single_prediction = SinglePrediction()
    result = single_prediction.predict(input_text)
    return render_template('summary.html',result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)