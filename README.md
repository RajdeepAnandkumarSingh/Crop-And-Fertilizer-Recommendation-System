# Crop-And-Fertilizer-Recommendation-System
🌱 Crop & Fertilizer Recommendation System

This project is a machine learning–powered web application built with Flask that helps farmers and agricultural enthusiasts make informed decisions about:

Crop Recommendation – Suggests the most suitable crop to grow based on soil nutrients (N, P, K), weather conditions (temperature, humidity, rainfall), and soil pH.

Fertilizer Recommendation – Provides the best fertilizer suggestion depending on soil nutrient levels and gives detailed recommendations to improve soil health.

🚀 Features

Crop Prediction

Input soil and environmental parameters.

Model predicts the best crop to cultivate.

Supports crops like rice, maize, chickpea, banana, mango, coffee, etc.

Fertilizer Recommendation

Input Nitrogen (N), Phosphorous (P), and Potassium (K) levels.

Model recommends the best fertilizer (e.g., Urea, DAP, NPK variants).

Provides actionable suggestions for soil improvement (e.g., manure, compost, crop rotation).

Web Interface

🛠️ Tech Stack

Backend: Flask (Python)

Machine Learning: Scikit-learn, Pandas, Numpy

Models:

Crop_model.pkl – trained on crop_recommendation.csv

classifier1.pkl – trained on Fertilizer.csv

Frontend: HTML templates (index, crop, fertilizer pages)

📊 Datasets

Crop Dataset: Contains soil nutrients, weather, and pH values mapped to suitable crops.

Fertilizer Dataset: Contains N, P, K values mapped to fertilizer recommendations.

🎯 Use Cases

Helps farmers optimize crop selection.

Guides in sustainable fertilizer usage.

Can be extended into a full AgriTech solution with weather APIs & soil sensors.

Simple Flask-based UI with separate pages for crop and fertilizer predictions.

REST APIs available for integration.
