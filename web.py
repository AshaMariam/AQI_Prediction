from flask import Flask, render_template, request, session
import pickle
import numpy as np
import os   


app = Flask(__name__)
app.secret_key = 'aqiprediction'


model_path = 'C:/Users/susmi/OneDrive/Desktop/ICTAK_DSA_2023/Web_AQI/model.pkl'
model = pickle.load(open(model_path, 'rb'))

scaler_path = 'C:/Users/susmi/OneDrive/Desktop/ICTAK_DSA_2023/Web_AQI/scaler.pkl'
try:
    scaler = pickle.load(open(scaler_path, 'rb'))
except FileNotFoundError:
    print("FileNotFoundError: Scaler file not found.")
except Exception as e:
    print(f"An error occurred while loading the scaler: {e}")


def get_aqi_bucket(aqi):
    if 0 <= aqi <= 50:
        return 'Good'
    elif 51 <= aqi <= 100:
        return 'Satisfactory'
    elif 101 <= aqi <= 200:
        return 'Moderate'
    elif 201 <= aqi <= 300:
        return 'Poor'
    elif 301 <= aqi <= 400:
        return 'Very Poor'
    elif aqi >= 401:
        return 'Severe'
    else:
        return 'None'


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        city    = request.form['city']
        pm2p5   = float(request.form['pm2p5'])
        pm10    = float(request.form['pm10'])
        no      = float(request.form['no'])
        no2     = float(request.form['no2'])
        nox     = float(request.form['nox'])
        nh3     = float(request.form['nh3'])
        co      = float(request.form['co'])
        so2     = float(request.form['so2'])
        o3      = float(request.form['o3'])
        benzene = float(request.form['benzene'])
        toluene = float(request.form['toluene'])
        xylene  = float(request.form['xylene'])
        
        print(f"Input values: pm2p5={pm2p5}, pm10={pm10}, no={no}, no2={no2}, nox={nox}, nh3={nh3}, co={co}, so2={so2}, o3={o3}, benzene={benzene}, toluene={toluene}, xylene={xylene}")

        
        aqi_data = np.array([[pm2p5, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]])
        features = ['pm2p5', 'pm10', 'no', 'no2', 'nox', 'nh3', 'co', 'so2', 'o3', 'benzene', 'toluene', 'xylene']
        aqi_data = np.array([[float(request.form[feature]) for feature in features]])
        print("Input data array:", aqi_data)
    
        
        placeholder_values = np.zeros((aqi_data.shape[0], model.n_features_in_ - aqi_data.shape[1]))
        aqi_data = np.concatenate([aqi_data, placeholder_values], axis=1)
        
        
        aqi_data_scaled = scaler.transform(aqi_data)
        print("Scaled input data array:", aqi_data_scaled)
        
        prediction = model.predict(aqi_data_scaled)[0]        
        aqi_level = get_aqi_bucket(prediction)
        
        session['prediction_result'] = {
            'city': city,
            'prediction': prediction,
            'aqi_bucket': aqi_level
        }
        
        print(f"Prediction: {prediction}")
        
        return render_template('res.html', prediction_text=f"The AQI of { city } is : { prediction }", aqi_bucket=aqi_level)
    else:
        if 'prediction_result' in session:
            result = session['prediction_result']
            return render_template('res.html', prediction_text=f"The AQI of {result['city']} is : {result['prediction']}", aqi_bucket=result['aqi_bucket'])
        else:
            return render_template('res.html')
               
if __name__ == '__main__':
    app.run(port=8000, debug=True)