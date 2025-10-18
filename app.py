"""Flask Image Recognition App"""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)

@app.route("/")
def main():
    """Flask Image Recognition App"""
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Handle Image Prediction Request"""
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))
    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)

if __name__ == "__main__":
    app.run(port=9000, debug=True)
