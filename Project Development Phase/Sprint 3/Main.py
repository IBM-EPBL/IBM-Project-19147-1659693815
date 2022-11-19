from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request

import smtplib
#from PIL import Image
import pickle


import numpy as np






import ibm_db
import pandas
import ibm_db_dbi
from sqlalchemy import create_engine

engine = create_engine('sqlite://',
                       echo = False)

dsn_hostname = "98538591-7217-4024-b027-8baa776ffad1.c3n41cmd0nqnrk39u98g.databases.appdomain.cloud"
dsn_uid = "hyr41683"
dsn_pwd = "UMhb2FYOgfQmG5Cv"

dsn_driver = "{IBM DB2 ODBC DRIVER}"
dsn_database = "BLUDB"
dsn_port = "30875"
dsn_protocol = "TCPIP"
dsn_security = "SSL"

dsn = (
    "DRIVER={0};"
    "DATABASE={1};"
    "HOSTNAME={2};"
    "PORT={3};"
    "PROTOCOL={4};"
    "UID={5};"
    "PWD={6};"
    "SECURITY={7};").format(dsn_driver, dsn_database, dsn_hostname, dsn_port, dsn_protocol, dsn_uid, dsn_pwd,dsn_security)



try:
    conn = ibm_db.connect(dsn, "", "")
    print ("Connected to database: ", dsn_database, "as user: ", dsn_uid, "on host: ", dsn_hostname)

except:
    print ("Unable to connect: ", ibm_db.conn_errormsg() )



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

app.config['DEBUG']


@app.route("/")
def homepage():
    return render_template('index.html')
@app.route("/Home")
def Home():
    return render_template('index.html')
@app.route("/AdminLogin")

@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')
@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewQuery1")
def NewQuery1():
    return render_template('NewQueryReg.html')









@app.route("/reg", methods=['GET', 'POST'])
def reg():
    if request.method == 'POST':
        n = request.form['name']

        address = request.form['address']
        age = request.form['age']
        pnumber = request.form['phone']
        email = request.form['email']
        zip = request.form['zip']
        uname = request.form['uname']
        password = request.form['psw']


        conn = ibm_db.connect(dsn, "", "")

        insertQuery = "INSERT INTO REGTB VALUES ('" + n + "','" + age + "','" + email + "','" + pnumber + "','" + zip + "','" + address + "','" + uname + "','" + password + "')"
        insert_table = ibm_db.exec_immediate(conn, insertQuery)
        print(insert_table)


        # return 'file register successfully'
        return render_template('UserLogin.html')

@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    error = None
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']


        conn = ibm_db.connect(dsn, "", "")
        pd_conn = ibm_db_dbi.Connection(conn)

        selectQuery = "SELECT * from regtb where UserName='" + username + "' and password='" + password + "'"
        dataframe = pandas.read_sql(selectQuery, pd_conn)

        if dataframe.empty:

            return 'Username or Password is wrong'
        else:
            print("Login")


            return render_template('NewQueryReg.html')


@app.route("/newquery", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':
        uname = session['uname']
        msg = ""

        age = request.form['age']
        gender = request.form['gender']
        height = request.form['height']
        weight = request.form['weight']
        aphi = request.form['aphi']
        aplo = request.form['aplo']
        choles = request.form['choles']
        glucose = request.form['glucose']
        smoke = request.form['smoke']
        alcohol = request.form['alcohol']

        age = float(age)
        gender = float(gender)
        height = float(height)
        weight = float(weight)
        aphi = float(aphi)
        aplo = float(aplo)
        choles = float(choles)
        glucose = float(glucose)
        smoke = float(smoke)
        alcohol = float(alcohol)

        filename = './Heart/heart-prediction-rfc-model.pkl'
        classifier = pickle.load(open(filename, 'rb'))

        data = np.array([[age, gender, height, weight, aphi, aplo, choles, glucose, smoke, alcohol]])
        my_prediction = classifier.predict(data)

        if my_prediction == 1:

            msg = 'Hello! According to our Calculations, You have  Heart Disease'

        else:

            msg = 'Congratulations!!  You DO NOT have  Heart Disease'




        return render_template('NewQueryReg.html', data=msg)










if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
