import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Memuat model dari JSON
with open("mental_disorder_model.json", "r") as model_file:
    model_json = json.load(model_file)
model = tf.keras.models.model_from_json(model_json)
model.load_weights("mental_disorder.weights.h5") 

# Memuat scaler dari JSON
with open("scaler.json", "r") as scaler_file:
    scaler_data = json.load(scaler_file)

class Scaler:
    def __init__(self, data):
        self.mean_ = np.array(data["mean"])
        self.scale_ = np.array(data["scale"])
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_

scaler = Scaler(scaler_data)

# Memuat label encoder dari JSON
with open("label_encoder.json", "r") as le_file:
    label_encoder_data = json.load(le_file)

class LabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
    
    def inverse_transform(self, y):
        return self.classes_[y]

label_encoder = LabelEncoder(label_encoder_data["classes"])

# Mapping label ke Bahasa Indonesia
label_mapping = {
    'MDD': 'Gangguan Depresi Mayor (MDD)',  
    'ASD': 'Gangguan Spektrum Autisme (ASD)',
    'Loneliness': 'Kesepian',
    'bipolar': 'Gangguan Bipolar',
    'anexiety': 'Gangguan Kecemasan',
    'PTSD': 'Gangguan Stres Pascatrauma (PTSD)',
    'sleeping disorder': 'Gangguan Tidur',
    'psychotic deprission': 'Depresi Psikotik',
    'eating disorder': 'Gangguan Makan',
    'ADHD': 'Gangguan Pemusatan Perhatian dan Hiperaktivitas (ADHD)',
    'PDD': 'Gangguan Perkembangan Pervasif (PDD)',
    'OCD': 'Gangguan Obsesif-Kompulsif (OCD)',
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Mengambil input dari formulir
        ag = request.form.get("age")
        if ag and ag.isdigit():
            ag = int(ag)
        else:
            return jsonify({"error": "Harap masukkan usia Anda dengan benar."}), 400

        # Ambil data lainnya, misalnya feeling_nervous, panic, dll.
        feeling_nervous = request.form.get("feeling_nervous") == "Ya"
        panic = request.form.get("panic") == "Ya"
        breathing_rapidly = request.form.get("breathing_rapidly") == "Ya"
        sweating = request.form.get("sweating") == "Ya"
        trouble_in_concentration = request.form.get("trouble_in_concentration") == "Ya"
        having_trouble_in_sleeping = request.form.get("having_trouble_in_sleeping") == "Ya"
        having_trouble_with_work = request.form.get("having_trouble_with_work") == "Ya"
        hopelessness = request.form.get("hopelessness") == "Ya"
        anger = request.form.get("anger") == "Ya"
        over_react = request.form.get("over_react") == "Ya"
        change_in_eating = request.form.get("change_in_eating") == "Ya"
        suicidal_thought = request.form.get("suicidal_thought") == "Ya"
        feeling_tired = request.form.get("feeling_tired") == "Ya"
        close_friend = request.form.get("close_friend") == "Ya"
        social_media_addiction = request.form.get("social_media_addiction") == "Ya"
        weight_gain = request.form.get("weight_gain") == "Ya"
        introvert = request.form.get("introvert") == "Ya"
        popping_up_stressful_memory = request.form.get("popping_up_stressful_memory") == "Ya"
        having_nightmares = request.form.get("having_nightmares") == "Ya"
        avoids_people_or_activities = request.form.get("avoids_people_or_activities") == "Ya"
        feeling_negative = request.form.get("feeling_negative") == "Ya"
        trouble_concentrating = request.form.get("trouble_concentrating") == "Ya"
        blaming_yourself = request.form.get("blaming_yourself") == "Ya"
        hallucinations = request.form.get("hallucinations") == "Ya"
        repetitive_behaviour = request.form.get("repetitive_behaviour") == "Ya"
        seasonally = request.form.get("seasonally") == "Ya"
        increased_energy = request.form.get("increased_energy") == "Ya"

        # Konversi input ke array numpy
        user_data = np.array([[ 
            ag, feeling_nervous, panic, breathing_rapidly, sweating, 
            trouble_in_concentration, having_trouble_in_sleeping, having_trouble_with_work, 
            hopelessness, anger, over_react, change_in_eating, suicidal_thought, 
            feeling_tired, close_friend, social_media_addiction, weight_gain, introvert, 
            popping_up_stressful_memory, having_nightmares, avoids_people_or_activities, 
            feeling_negative, trouble_concentrating, blaming_yourself, hallucinations, 
            repetitive_behaviour, seasonally, increased_energy
        ]]).astype(float)

        # Melakukan skalasi
        user_data_scaled = scaler.transform(user_data)

        # Prediksi
        prediction = model.predict(user_data_scaled)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        confidence_score = np.max(prediction) * 100  # Konversi ke persen

        # Map ke Bahasa Indonesia
        predicted_class_in_indonesian = label_mapping.get(predicted_class[0], predicted_class[0])

        # Return hasil dalam format JSON
        return jsonify({
            "result": f"Hasil Diagnosa bahwa Anda mengalami {predicted_class_in_indonesian} dengan tingkat kepercayaan {confidence_score:.2f}%",
            "prediction": predicted_class_in_indonesian,
            "confidence": confidence_score
        })
    
    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
