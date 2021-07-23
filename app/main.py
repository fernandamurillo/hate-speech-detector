# import basics
import os
import pickle
from text_cleaner import clean_text
from flask_cors import cross_origin

# preprocessing imports
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

# import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask import jsonify
from utils import get_base_url, allowed_file, and_syntax

'''
Coding center code - comment out the following 4 lines of code when ready for production
'''
# load up the model into memory
# you will need to have all your trained model in the app/ directory.
model = pickle.load(open("log_reg_model.sav", 'rb'))
cv = pickle.load(open("countvec.pkl",'rb'))

# preprocessing
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
# port = 12335
# base_url = get_base_url(port)
# app = Flask(__name__, static_url_path=base_url+'static')

app = Flask(__name__)

@app.route('/')
# @app.route(base_url)
def home():
    return render_template('home.html', generated=None)

@app.route('/results', methods=["GET","POST"])
@cross_origin()
def results():
    if request.method=="POST":
        
        normalized_review = []
        review =(request.form["Review"])
        
        review = [review]
        cleaned_texts = normalize_texts(review)
        cleaned_texts = cv.transform(cleaned_texts)                

        prediction=model.predict(cleaned_texts)
        
        output=""
        if prediction[0]==0:
            output="Negative"

        else:
            output="Positive"

        return render_template('home.html',prediction_text=f'This is a {output} Review')

    return render_template("home.html")

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc8.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
