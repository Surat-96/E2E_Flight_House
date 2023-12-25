from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np
import pandas as pd

#from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

app = Flask(__name__)

#flight price model read
filename = open('poff/FIPmodel.pkl', 'rb')
clf = pickle.load(filename)
filename.close()

filename = open('House_Price/hpmodel.pkl', 'rb')
model = pickle.load(filename)
filename.close()

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/fppredict', methods=['GET','POST'])
def fppredict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # print("Journey Date : ",Journey_day, Journey_month)

         # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        Duration_hours = abs(Arrival_hour - Dep_hour)
        Duration_mins = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_Stops = int(request.form["stops"])
        # print(Total_stops)

         # Airline
        Airline=int(request.form['airline'])

        # Source
        Source = int(request.form["Source"])

        # Destination
        Destination = int(request.form["Destination"])

        data = np.array([[Airline,Source,Destination,Total_Stops,Journey_day,Journey_month,Dep_hour,Dep_min,Arrival_hour,Arrival_min,Duration_hours,Duration_mins]])
        prediction = clf.predict(data)
        output=round(prediction[0],2)

        return render_template('fpshow.html',output=output)
    return render_template('fp.html')



@app.route('/hppredict', methods=['GET','POST'])
def hppredict():
    if request.method == "POST":

        na = request.form['na']
        CRIM = float(request.form['crim'])
        ZN = float(request.form['zn'])
        INDUS = float(request.form['indus'])
        CHAS = int(request.form['chas'])
        NOX = float(request.form['nox'])
        RM = float(request.form['rm'])
        AGE = float(request.form['age'])
        DIS = float(request.form['dis'])
        RAD = int(request.form['rad'])
        TAX = float(request.form['tax'])
        PTRATIO = float(request.form['ptratio'])
        B = float(request.form['b'])
        LSTAT = float(request.form['lstat'])

        dat = np.array([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
        my_prediction = model.predict(dat)
        result=round(my_prediction[0],2)

        return render_template('hpshow.html',name=na,result=result)
    return render_template('hp.html')




if __name__ == '__main__':
	app.run(debug=True)

