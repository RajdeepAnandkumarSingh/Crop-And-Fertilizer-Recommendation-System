from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the crop prediction model
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    print("Crop model loaded successfully.")
except Exception as e:
    print(f"Error loading crop model: {e}")
    crop_model = None

# Load the fertilizer recommendation model
try:
    fertilizer_model = pickle.load(open("classifier1.pkl", "rb"))
    print("Fertilizer model loaded successfully.")
except Exception as e:
    print(f"Error loading fertilizer model: {e}")
    fertilizer_model = None

# Label mapping for crop predictions
crop_labels = {
    1: "rice",
    2: "maize",
    3: "chickpea",
    4: "kidneybeans",
    5: "pigeonpeas",
    6: "mothbeans",
    7: "mungbean",
    8: "blackgram",
    9: "lentil",
    10: "pomegranate",
    11: "banana",
    12: "mango",
    13: "grapes",
    14: "watermelon",
    15: "muskmelon",
    16: "apple",
    17: "orange",
    18: "papaya",
    19: "coconut",
    20: "cotton",
    21: "jute",
    22: "coffee"
}

# Fertilizer recommendation messages
fertilizer_recommendations = {
    'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

    'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

    'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

    'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

    'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

    'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium"""
}

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Crop Prediction Page
@app.route('/crop')
def crop():
    return render_template('crop.html')

# Fertilizer Prediction Page
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

# Crop Prediction API
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Extract inputs from the form
        inputs = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        # Create a DataFrame for the model
        input_df = pd.DataFrame([inputs], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        # Make prediction
        prediction = crop_model.predict(input_df)[0]
        # Convert prediction to a native Python type
        prediction = int(prediction)
        # Map the numeric label to a crop name
        crop_name = crop_labels.get(prediction, "Unknown Crop")
        return jsonify({'prediction': crop_name})
    except KeyError as e:
        return jsonify({'error': f'Missing form field: {e}'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Fertilizer Prediction API
@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # Extract inputs from the form
        nitrogen = float(request.form['Nitrogen'])
        phosphorous = float(request.form['Phosphorous'])
        potassium = float(request.form['Potassium'])

        # Make prediction
        result = fertilizer_model.predict(np.array([[nitrogen, phosphorous, potassium]]))
        if result[0] == 0:
            result = 'TEN-TWENTY SIX-TWENTY SIX'
        elif result[0] == 1:
            result = 'Fourteen-Thirty Five-Fourteen'
        elif result[0] == 2:
            result = 'Seventeen-Seventeen-Seventeen'
        elif result[0] == 3:
            result = 'TWENTY-TWENTY'
        elif result[0] == 4:
            result = 'TWENTY EIGHT-TWENTY EIGHT'
        elif result[0] == 5:
            result = 'DAP'
        else:
            result = 'UREA'

        # Determine recommendations based on N, P, K values
        recommendations = []
        if nitrogen > 120:
            recommendations.append(fertilizer_recommendations['NHigh'])
        elif nitrogen < 80:
            recommendations.append(fertilizer_recommendations['Nlow'])

        if phosphorous > 50:
            recommendations.append(fertilizer_recommendations['PHigh'])
        elif phosphorous < 30:
            recommendations.append(fertilizer_recommendations['Plow'])

        if potassium > 200:
            recommendations.append(fertilizer_recommendations['KHigh'])
        elif potassium < 100:
            recommendations.append(fertilizer_recommendations['Klow'])

        return jsonify({'prediction': result, 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)