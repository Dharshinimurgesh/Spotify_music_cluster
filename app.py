from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
kmeans = joblib.load("spotify_kmeans_model.pkl")
scaler = joblib.load("spotify_scaler.pkl")

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# If cluster column not present create it
features = df[['danceability','energy','tempo','loudness','valence']]
scaled = scaler.transform(features)

df["cluster"] = kmeans.predict(scaled)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    danceability = float(request.form["danceability"])
    energy = float(request.form["energy"])
    tempo = float(request.form["tempo"])
    loudness = float(request.form["loudness"])
    valence = float(request.form["valence"])

    user_song = np.array([[danceability, energy, tempo, loudness, valence]])

    scaled_song = scaler.transform(user_song)

    cluster = kmeans.predict(scaled_song)[0]

    # get songs from same cluster
    recommendations = df[df["cluster"] == cluster].sample(5)

    songs = recommendations["track_name"].tolist()

    return render_template(
        "index.html",
        prediction=cluster,
        songs=songs
    )


if __name__ == "__main__":
    app.run(debug=True)