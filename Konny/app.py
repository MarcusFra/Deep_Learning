
from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from load_model import predict
from PIL import Image


app = Flask(__name__)

from image_proc import img_to_b64, score_word, b64_remove_header, b64_preprocess

@app.route('/submit_image', methods=['POST'])
def submit_image():
    print('IMAGE RECEIVED')
    error = None
    if request.method == 'POST':
        img_in = request.form['imgBase64']

        # Ensure image has no header
        img_in = b64_remove_header(img_in)

        # Convert image to process-able format
        img = b64_preprocess(img_in)

        img.save('image.png')

        small = img.resize((28,28))

        pred = str(predict(small))

        return jsonify({'response': f"You drew a {pred}."})

@app.route("/")
def main():
    return render_template("index.html")
