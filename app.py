from flask import Flask
from flask import render_template
from flask import request


import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route("/")
def home():
    return render_template("browser1.html")  
@app.route('/login',methods = ['POST'])  
def login():  
      uname=request.form['files']
      rr=pd.read_csv(uname)
      type(rr)
      y_pre=classifier.predict(rr)
      if y_pre[0]==0:
          return render_template('index1.html')
      elif y_pre[0]==1:
          return render_template('index2.html')
      elif y_pre[0]==2:
          return render_template('index3.html')
      elif y_pre[0]==3:
          return render_template('index4.html')
      elif y_pre[0]==4:
          return render_template('index5.html')
      elif y_pre[0]==5:
          return render_template('index6.html')
      elif y_pre[0]==6:
          return render_template('index7.html')  
if __name__ == '__main__':
   app.run()  

