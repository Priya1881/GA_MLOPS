from flask import Flask, render_template, request, jsonify
import os
from prediction_service import prediction
from tabulate import tabulate
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                path = request.form['filename']
                response = prediction.form_response(path)
                return render_template("index.html", response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ =="__main__":
    app.run(debug=True)
