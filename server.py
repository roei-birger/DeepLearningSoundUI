import os

from flask import Flask, render_template, request
from model.inference_orya_roi_gender import inf_rec, inf_upload
# from model.inference_dor_dolev import
import torch

app = Flask(__name__)


def load_model(name):
    model = torch.load(f"model/{name}", map_location=torch.device('cpu'))
    model.eval()

    return model


model_orya_roi = load_model("model_orya_roi.pth")
model2 = load_model("model_orya_roi.pth")

# model_dor_dolev = load_model("model_dor_dolev.pth")


@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/index', methods=['POST', 'GET'])
def send_text():
    if request.method == 'POST':
        if request.form['submit_button'] == 'rec':
            result = render_template("index.html", pred=inf_rec(model_orya_roi, model2))
            os.remove('recording.wav')
            return result

        elif request.form['submit_button'] == 'file':
            print("Upload file")
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save(uploaded_file.filename)
            result = render_template("index.html",
                                     pred=inf_upload(model_orya_roi, model2, name=uploaded_file.filename))
            if uploaded_file.filename != '':
                os.remove(f'{uploaded_file.filename}')

            return result

        elif request.form['submit_button'] == 'clean':
            return render_template("index.html", pred="")

        else:
            print("error")
            return render_template("index.html", pred="error")


def upload_file():
    print("file")


if __name__ == '__main__':
    app.run(debug=True)
