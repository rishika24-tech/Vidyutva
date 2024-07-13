import pandas as pd

from flask import Flask, render_template,request
import pickle

app=Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    num_rooms = request.form.get('num_rooms')
    num_people = request.form.get('num_people')
    is_ac = request.form.get('is_ac')
    is_tv= request.form.get('is_tv')
    print(num_rooms,num_people,is_ac,is_tv)

    input = pd.DataFrame([[num_rooms,num_people,is_ac,is_tv]],columns=['num_rooms','num_people','is_ac','is_tv'])
    prediction = pipe.predict(input)[0]


    return str(prediction)
if __name__=="__main__":
    app.run(debug=True,port=5001)