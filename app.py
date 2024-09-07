import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, session
from stock_prediction import fetch_data, prepare_data, train_model, evaluate_model, plot_predictions
import io
import json
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    data = fetch_data(ticker)
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    mse, prediction_results = evaluate_model(model, X_test, y_test, data)

    # Extract dates for plotting
    dates = data.index[-len(prediction_results):]

    # Convert data to JSON
    data_json = data.to_json(date_format='iso')

    # Store in session
    session['mse'] = mse
    session['prediction_results'] = prediction_results
    session['dates'] = [date.strftime('%Y-%m-%d') for date in dates]  # Convert dates to string format
    session['data'] = data_json

    # Create plot and save to BytesIO
    img = io.BytesIO()
    plot_predictions(data, [pr[1] for pr in prediction_results], dates)
    plt.savefig(img, format='png')
    img.seek(0)

    return render_template('result.html', ticker=ticker, mse=mse, prediction_results=prediction_results, plot_url='/plot')

@app.route('/plot')
def plot():
    # Retrieve from session
    data_json = session.get('data')
    prediction_results = session.get('prediction_results')
    dates = session.get('dates')

    # Reconstruct data from JSON
    data = pd.read_json(data_json)

    # Create the plot again
    img = io.BytesIO()
    plot_predictions(data, [pr[1] for pr in prediction_results], pd.to_datetime(dates))
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
