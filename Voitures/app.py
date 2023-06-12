from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)

label_mapping = joblib.load('label_mapping.pkl')
regr = joblib.load('regr.pkl')

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        # Get values through input bars
        Couleur = request.form.get("couleur")
        Etat = request.form.get("etat")
        Boite = request.form.get("boite")
        Cylindré = request.form.get("cylindre")
        Marque = request.form.get("marque")
        Carrosserie = request.form.get("carrosserie")
        Carburant = request.form.get("carburant")
        Gamme = request.form.get("gamme")
        Kilomètrage = request.form.get("Kilomètrage")
        Puissance = request.form.get("Puissance")
        Annee = request.form.get("Annee")

        # Map the input values to their corresponding codes using label_mapping
        Couleur_code = label_mapping['Couleur'][Couleur]
        Etat_code = label_mapping['Etat'][Etat]
        Boite_code = label_mapping['Boite'][Boite]
        Cylindré_code = label_mapping['Cylindré'][Cylindré]
        Marque_code = label_mapping['Marque'][Marque]
        Carrosserie_code = label_mapping['Carrosserie'][Carrosserie]
        Carburant_code = label_mapping['Carburant'][Carburant]
        Gamme_code = label_mapping['Gamme'][Gamme]

        # Put inputs to dataframe
        X = pd.DataFrame([[Kilomètrage, Couleur_code, Etat_code, Boite_code, Cylindré_code, Marque_code, Gamme_code,
                           Puissance, Carrosserie_code, Carburant_code, Annee]],
                         columns=["Kilomètrage", "Couleur", "Etat", "Boite", "Cylindré", "Marque", "Gamme", "Puissance",
                                  "Carrosserie", "Carburant", "Annee"])

        # Get prediction
        prediction = regr.predict(X)

    else:
        prediction = ""

    return render_template("formulaire.html", output=prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug=True, port=3000)