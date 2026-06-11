from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/admission_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    gre = float(request.form['gre'])
    toefl = float(request.form['toefl'])
    university = float(request.form['university'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = float(request.form['research'])

    features = np.array([
        [gre,
         toefl,
         university,
         sop,
         lor,
         cgpa,
         research]
    ])

    prediction = model.predict(features)[0]

    chance = round(prediction * 100, 2)

    profile_score = round(
        ((gre / 340) * 30) +
        ((toefl / 120) * 20) +
        ((cgpa / 10) * 30) +
        (research * 20),
        2
    )

    if chance >= 80:
        category = "Safe Universities"
        universities = [
            "Arizona State University",
            "University of Texas Dallas",
            "University at Buffalo"
        ]

    elif chance >= 60:
        category = "Moderate Universities"
        universities = [
            "Northeastern University",
            "University of Illinois Chicago",
            "SUNY Albany"
        ]

    else:
        category = "Ambitious Universities"
        universities = [
            "UCLA",
            "UC Berkeley",
            "Carnegie Mellon University"
        ]

    suggestions = []

    if gre < 320:
        suggestions.append("Improve GRE Score")

    if toefl < 105:
        suggestions.append("Improve TOEFL Score")

    if cgpa < 8.5:
        suggestions.append("Improve CGPA")

    if research == 0:
        suggestions.append("Gain Research Experience")

    if sop < 4:
        suggestions.append("Strengthen SOP")

    if lor < 4:
        suggestions.append("Improve LOR Quality")

    return render_template(
        'result.html',
        chance=chance,
        category=category,
        profile_score=profile_score,
        universities=universities,
        suggestions=suggestions
    )

if __name__ == '__main__':
    app.run(debug=True)