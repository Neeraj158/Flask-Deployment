
import os
import numpy as np
import sqlite3
import librosa

from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename, send_from_directory
import sqlite3
import pickle, joblib
import sqlite3
import random

import smtplib 
from email.message import EmailMessage





app = Flask(__name__)

# Load the trained Random Forest model and label encoder
model1 = joblib.load("models/model_beats.sav")
model2 = joblib.load("models/model_hs.sav")
model3 = joblib.load("models/model_multi.sav")

label_encoder1 = joblib.load("models/beats_encoder.pkl")
label_encoder2 = joblib.load("models/hs_encoder.pkl")
label_encoder3 = joblib.load("models/multi_encoder.pkl")

# Function to extract audio features
# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    return np.hstack([mfccs, chroma, mel, zcr, rms])
 


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']
    file_path = "static/temp_audio.wav"
    file.save(file_path)

    # Extract features
    features = extract_features(file_path)
    print(f"Extracted Features Shape: {features.shape}")  # Debugging

    # Check if feature size matches model input
    expected_features = 155  # Change this based on model training
    if features.shape[0] != expected_features:
        os.remove(file_path)
        return render_template('index.html', error=f"Feature mismatch: expected {expected_features}, got {features.shape[0]}")

    # Reshape for prediction
    features = features.reshape(1, -1)
    prediction1 = model1.predict(features)[0]
    predicted_label1 = label_encoder1.inverse_transform([prediction1])[0]

    if prediction1 == 0:
        prediction2 = model2.predict(features)[0]
        predicted_label2 = label_encoder2.inverse_transform([prediction2])[0]    
        if prediction2 == 0:
            prediction3 = model3.predict(features)[0]
            predicted_label3 = label_encoder3.inverse_transform([prediction3])[0] 
            return render_template('result1.html', output1 = predicted_label1, output2=predicted_label2,output3=predicted_label3)  
        else:
            return render_template('result2.html', output1 = predicted_label1, output2=predicted_label2)  
    else:
        return render_template('result3.html', output1 = predicted_label1)

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(100000, 999999)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "myprojectstp@gmail.com"
    msg['To'] = email
    
    
    s =  smtplib.SMTP('smtp.gmail.com', 587)     
    s.starttls()
    s.login("myprojectstp@gmail.com", "paxgxdrhifmqcrzn")
    s.send_message(msg)
    s.quit()
    return render_template("otp.html") 


@app.route('/otp', methods=['POST'])
def otp():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signin.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route('/notebook')
def notebook1():
    return render_template('Notebook.html')


if __name__ == '__main__':
    app.run(debug=True)