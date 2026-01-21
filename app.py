from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model from model folder
model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": int(request.form["GrLivArea"]),
            "TotalBsmtSF": int(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "YearBuilt": int(request.form["YearBuilt"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        # Ensure columns match training data
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
