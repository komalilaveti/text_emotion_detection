from flask import Flask ,render_template ,request 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle 
import json 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
import neattext.functions as nfx
import nltk 

app = Flask(__name__) 

def step1(text):
    # text = [text]
    # text = text.apply(nfx.remove_userhandles) 
    # text = text.apply(nfx.remove_stopwords)
    model = pickle.load(open('model.pkl','rb')) 
    ans = str(model.predict([text])) 
    return ans



@app.route("/",methods = ["GET","POST"]) 
def home():
    return render_template('index.html') 

@app.route("/submit",methods = ["GET","POST"]) 

def submit():
    res = []
    if request.method == "POST":
        text = request.form.get("text","") 
        res.append(text)
        # print(text)
        t = step1(text)
        t = t[2:-2]
        res.append(t)
    return render_template("result.html",res = res) 

if __name__ == "__main__":
    app.run(debug = True)


