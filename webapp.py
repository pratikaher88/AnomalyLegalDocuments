from flask import Flask,render_template,flash,redirect,url_for,session ,logging,request
from werkzeug.utils import secure_filename

import analyzerSKL
import analyzer

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('totets.html')

@app.route('/submit', methods=['POST'])

def submit():

    InputFromBrowser=request.form['text']

    if InputFromBrowser=='':
        f = request.files['file']
        f.filename = 'myinputdata'
        f.save(secure_filename(f.filename))
        filee = open('myinputdata', 'r')

        output = analyzer.getFinalText(filee.read())
    else:
        output=analyzerSKL.getFinalText(InputFromBrowser)
    # output=analyzer.getFinalKmeans(InputFromBrowser)

    return render_template('outputpage.html',text=output)

if __name__ == "__main__":
    app.run()