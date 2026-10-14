
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import onnxruntime as ort
import json
import os

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# =========================
# PATHS AND MODEL SETTINGS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(
    MODEL_DIR, "efficientnet_v2_s_best.onnx"
)
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

disease_session = None
disease_input_name = None
disease_classes = []
IMAGE_SIZE = 224

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(1, 1, 3)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(1, 1, 3)


# =========================
# LOAD DISEASE MODEL
# =========================

def load_disease_model():
    global disease_session
    global disease_input_name
    global disease_classes
    global IMAGE_SIZE
    global IMAGE_MEAN
    global IMAGE_STD

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not os.path.exists(CLASSES_PATH):
            raise FileNotFoundError(
                f"Classes file not found: {CLASSES_PATH}"
            )

        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        disease_classes = metadata["classes"]
        IMAGE_SIZE = int(metadata.get("image_size", 224))

        IMAGE_MEAN = np.array(
            metadata.get("mean", [0.485, 0.456, 0.406]),
            dtype=np.float32
        ).reshape(1, 1, 3)

        IMAGE_STD = np.array(
            metadata.get("std", [0.229, 0.224, 0.225]),
            dtype=np.float32
        ).reshape(1, 1, 3)

        if not disease_classes:
            raise ValueError("No disease classes found.")

        if np.any(IMAGE_STD == 0):
            raise ValueError("Invalid normalization standard deviation.")

        disease_session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        model_input = disease_session.get_inputs()[0]
        disease_input_name = model_input.name

        output_count = disease_session.get_outputs()[0].shape[-1]

        if output_count != len(disease_classes):
            raise ValueError(
                f"Model outputs {output_count} classes, "
                f"but classes.json contains {len(disease_classes)}."
            )

        print("Disease detection model loaded successfully.")
        print("Model:", os.path.basename(MODEL_PATH))
        print("Input:", model_input.shape)
        print("Classes:", len(disease_classes))

    except Exception as error:
        disease_session = None
        disease_input_name = None
        disease_classes = []
        print("Disease model loading failed:", error)


load_disease_model()


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image, dtype=np.float32
    ) / 255.0

    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD

    # HWC -> CHW -> NCHW
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return np.ascontiguousarray(image_array, dtype=np.float32)


# =========================
# PREDICTION HELPERS
# =========================

def get_probabilities(output):
    scores = np.asarray(output, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid prediction scores.")

    # Use existing probabilities if output is already normalized.
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=1e-3)
    ):
        return scores

    # Otherwise, convert logits to probabilities.
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def format_crop_name(raw_crop):
    crop_names = {
        "Apple": "Apple",
        "Blueberry": "Blueberry",
        "Cherry_(including_sour)": "Cherry",
        "Corn_(maize)": "Corn",
        "Grape": "Grape",
        "Orange": "Orange",
        "Peach": "Peach",
        "Pepper,_bell": "Bell Pepper",
        "Potato": "Potato",
        "Raspberry": "Raspberry",
        "Soybean": "Soybean",
        "Squash": "Squash",
        "Strawberry": "Strawberry",
        "Tomato": "Tomato"
    }

    return crop_names.get(
        raw_crop, raw_crop.replace("_", " ")
    )


def format_disease_name(raw_disease):
    disease = raw_disease.replace("_", " ")
    disease = disease.replace("  ", " ").strip()

    if disease.lower() == "healthy":
        return "Healthy"

    return disease


def normalize_crop_name(crop):
    crop = crop.lower().strip()

    aliases = {
        "corn": "corn",
        "maize": "corn",
        "bell pepper": "pepper",
        "pepper": "pepper",
        "cherry": "cherry",
        "potato": "potato",
        "tomato": "tomato",
        "apple": "apple",
        "blueberry": "blueberry",
        "grape": "grape",
        "orange": "orange",
        "peach": "peach",
        "raspberry": "raspberry",
        "soybean": "soybean",
        "squash": "squash",
        "strawberry": "strawberry"
    }

    return aliases.get(crop, crop)


# =========================
# MONGODB CONNECTION
# =========================

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/"
)

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=3000
    )

    client.admin.command("ping")

    db = client["agromind_ai"]

    users_collection = db["users"]
    crop_collection = db["crop_recommendations"]
    disease_collection = db["disease_detections"]
    irrigation_collection = db["irrigation"]
    fertilizer_collection = db["fertilizer_advice"]
    weather_collection = db["weather"]

    mongo_status = True

except Exception as error:
    print("MongoDB connection failed:", error)

    client = None
    db = None

    users_collection = None
    crop_collection = None
    disease_collection = None
    irrigation_collection = None
    fertilizer_collection = None
    weather_collection = None

    mongo_status = False


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "AgroMind AI Backend is running!",
        "project": "AgroMind AI",
        "status": "online"
    })


# =========================
# HEALTH CHECK
# =========================

@app.route("/api/health")
def health():
    return jsonify({
        "success": True,
        "backend": "online",
        "mongodb": "connected" if mongo_status else "not connected",
        "disease_model": (
            "loaded" if disease_session is not None
            else "not loaded"
        )
    })


# =========================
# TEST API
# =========================

@app.route("/api/test")
def test():
    return jsonify({
        "success": True,
        "message": "AgroMind API working correctly"
    })


# =========================
# REGISTER
# =========================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "message": "Name, email and password are required"
        }), 400

    if users_collection is None:
        return jsonify({
            "success": False,
            "message": "MongoDB is not connected"
        }), 500

    existing_user = users_collection.find_one({"email": email})

    if existing_user:
        return jsonify({
            "success": False,
            "message": "Email already registered"
        }), 409

    user = {
        "name": name,
        "email": email,
        "password": password,
        "role": "farmer",
        "created_at": datetime.utcnow()
    }

    result = users_collection.insert_one(user)

    return jsonify({
        "success": True,
        "message": "Registration successful",
        "user_id": str(result.inserted_id)
    }), 201


# =========================
# LOGIN
# =========================

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received"
        }), 400

    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required"
        }), 400

    if users_collection is None:
        return jsonify({
            "success": False,
            "message": "MongoDB is not connected"
        }), 500

    user = users_collection.find_one({
        "email": email,
        "password": password
    })

    if not user:
        return jsonify({
            "success": False,
            "message": "Invalid email or password"
        }), 401

    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "farmer")
        }
    })


# =========================
# CROP RECOMMENDATION
# =========================

@app.route("/api/crop-recommendation", methods=["POST"])
def crop_recommendation():
    data = request.get_json(silent=True) or {}

    try:
        nitrogen = float(data.get("nitrogen", 0))
        phosphorus = float(data.get("phosphorus", 0))
        potassium = float(data.get("potassium", 0))
        temperature = float(data.get("temperature", 25))
        humidity = float(data.get("humidity", 60))
        ph = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 100))

        values = [
            nitrogen, phosphorus, potassium, temperature,
            humidity, ph, rainfall
        ]

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Values must be finite numbers.")

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Enter valid numeric soil and weather values."
        }), 400

    # Demo recommendation logic
    if nitrogen > 80 and rainfall > 150:
        crop = "Rice"
    elif potassium > 60 and temperature > 25:
        crop = "Cotton"
    elif phosphorus > 50 and ph >= 6:
        crop = "Wheat"
    elif temperature > 25 and humidity > 60:
        crop = "Tomato"
    else:
        crop = "Maize"

    result = {
        "crop": crop,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "message": (
            f"{crop} may be suitable based on the entered conditions."
        )
    }

    if crop_collection is not None:
        try:
            crop_collection.insert_one({
                **result,
                "created_at": datetime.utcnow()
            })
        except Exception as error:
            print("Crop history save failed:", error)

    return jsonify({
        "success": True,
        "result": result
    })


# =========================
# DISEASE DETECTION - ONNX
# =========================

@app.route("/api/disease-detection", methods=["POST"])
def disease_detection():
    if disease_session is None:
        return jsonify({
            "success": False,
            "error": "Disease detection model is not loaded."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload a leaf image."
        }), 400

    image_file = request.files["image"]
    selected_crop = request.form.get("crop", "").strip()

    if not selected_crop:
        return jsonify({
            "success": False,
            "error": "Please select a crop."
        }), 400

    if not image_file.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    if image_file.mimetype not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        return jsonify({
            "success": False,
            "error": "Upload a JPG, PNG or WEBP image."
        }), 400

    image_data = image_file.read()

    if not image_data:
        return jsonify({
            "success": False,
            "error": "The uploaded image is empty."
        }), 400

    if len(image_data) > 10 * 1024 * 1024:
        return jsonify({
            "success": False,
            "error": "Image must be less than 10 MB."
        }), 400

    try:
        input_tensor = preprocess_image(image_data)

        outputs = disease_session.run(
            None,
            {disease_input_name: input_tensor}
        )

        probabilities = get_probabilities(outputs[0])

        if len(probabilities) != len(disease_classes):
            raise ValueError(
                "Prediction output does not match class labels."
            )

        predicted_index = int(np.argmax(probabilities))
        class_name = disease_classes[predicted_index]
        confidence = float(probabilities[predicted_index])

        if "___" in class_name:
            raw_crop, raw_disease = class_name.split("___", 1)
        else:
            raw_crop = "Unknown"
            raw_disease = class_name

        predicted_crop = format_crop_name(raw_crop)
        disease_name = format_disease_name(raw_disease)

        is_healthy = raw_disease.lower() == "healthy"

        crop_matches = (
            normalize_crop_name(selected_crop)
            == normalize_crop_name(predicted_crop)
        )

        result = {
            "crop": predicted_crop,
            "selected_crop": selected_crop,
            "disease": disease_name,
            "status": "Healthy" if is_healthy else "Possible Disease Detected",
            "confidence": round(confidence * 100, 2),
            "class_name": class_name,
            "crop_matches": crop_matches,
            "message": (
                "The model classified the leaf as healthy."
                if is_healthy
                else f"The model detected a possible condition: {disease_name}."
            )
        }

        if not crop_matches:
            result["warning"] = (
                "The predicted crop differs from your selected crop. "
                "Check that the uploaded image and crop selection match."
            )

        if disease_collection is not None:
            try:
                disease_collection.insert_one({
                    "selected_crop": selected_crop,
                    "predicted_crop": predicted_crop,
                    "disease": disease_name,
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "class_name": class_name,
                    "crop_matches": crop_matches,
                    "filename": os.path.basename(image_file.filename),
                    "created_at": datetime.utcnow()
                })
            except Exception as error:
                print("Disease history save failed:", error)

        return jsonify({
            "success": True,
            "message": "Disease prediction completed.",
            "result": result
        }), 200

    except (UnidentifiedImageError, OSError, ValueError) as error:
        print("Invalid image or prediction data:", error)

        return jsonify({
            "success": False,
            "error": "Unable to process this image. Upload a valid leaf image."
        }), 400

    except Exception as error:
        print("Disease prediction failed:", error)

        return jsonify({
            "success": False,
            "error": "Disease prediction failed. Check the backend terminal."
        }), 500


# =========================
# IRRIGATION
# =========================

@app.route("/api/irrigation", methods=["POST"])
def irrigation():
    data = request.get_json(silent=True) or {}

    try:
        soil_moisture = float(data.get("soil_moisture", 40))
        temperature = float(data.get("temperature", 30))
        rainfall = float(data.get("rainfall", 0))

        if not all(np.isfinite(value) for value in [
            soil_moisture, temperature, rainfall
        ]):
            raise ValueError()

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Enter valid numeric irrigation values."
        }), 400

    if rainfall > 20:
        recommendation = (
            "Avoid irrigation because rainfall is sufficient."
        )
        status = "Not Required"
    elif soil_moisture < 30:
        recommendation = (
            "Irrigation is recommended. Soil moisture is low."
        )
        status = "Required"
    elif soil_moisture < 50:
        recommendation = "Light irrigation may be required."
        status = "Moderate"
    else:
        recommendation = "No irrigation required at the moment."
        status = "Not Required"

    result = {
        "status": status,
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "rainfall": rainfall,
        "recommendation": recommendation
    }

    if irrigation_collection is not None:
        try:
            irrigation_collection.insert_one({
                **result,
                "created_at": datetime.utcnow()
            })
        except Exception as error:
            print("Irrigation history save failed:", error)

    return jsonify({
        "success": True,
        "result": result
    })


# =========================
# FERTILIZER ADVICE
# =========================

@app.route("/api/fertilizer-advice", methods=["POST"])
def fertilizer_advice():
    data = request.get_json(silent=True) or {}

    crop = data.get("crop", "Tomato")
    soil = data.get("soil", "Loamy")
    growth_stage = data.get("growthStage", "Vegetative")

    try:
        nitrogen = float(data.get("nitrogen", 0))
        phosphorus = float(data.get("phosphorus", 0))
        potassium = float(data.get("potassium", 0))

        if not all(np.isfinite(value) for value in [
            nitrogen, phosphorus, potassium
        ]):
            raise ValueError()

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Enter valid numeric NPK values."
        }), 400

    if nitrogen < 40:
        fertilizer = "Nitrogen-rich fertilizer"
        recommendation = (
            "Your soil has low nitrogen. "
            "Consider a nitrogen-rich fertilizer."
        )
    elif phosphorus < 30:
        fertilizer = "Phosphorus-rich fertilizer"
        recommendation = (
            "Your soil has low phosphorus. "
            "A phosphorus-rich fertilizer may help."
        )
    elif potassium < 30:
        fertilizer = "Potassium-rich fertilizer"
        recommendation = (
            "Your soil has low potassium. "
            "Consider a potassium-rich fertilizer."
        )
    else:
        fertilizer = "Balanced NPK fertilizer"
        recommendation = (
            "The entered NPK values are within the demo range."
        )

    result = {
        "crop": crop,
        "soil": soil,
        "growthStage": growth_stage,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "fertilizer": fertilizer,
        "recommendation": recommendation
    }

    if fertilizer_collection is not None:
        try:
            fertilizer_collection.insert_one({
                **result,
                "created_at": datetime.utcnow()
            })
        except Exception as error:
            print("Fertilizer history save failed:", error)

    return jsonify({
        "success": True,
        "result": result
    })


# =========================
# WEATHER - DEMO DATA
# =========================

@app.route("/api/weather", methods=["GET"])
def weather():
    city = request.args.get("city", "Udumalpet")

    result = {
        "city": city,
        "temperature": 29,
        "condition": "Partly Cloudy",
        "humidity": 68,
        "wind": 14,
        "rainfall": 12,
        "feelsLike": 31,
        "farmingAdvice": (
            "Good conditions for field activities. "
            "Check local rainfall before irrigation."
        )
    }

    if weather_collection is not None:
        try:
            weather_collection.insert_one({
                **result,
                "created_at": datetime.utcnow()
            })
        except Exception as error:
            print("Weather history save failed:", error)

    return jsonify({
        "success": True,
        "result": result
    })


# =========================
# CHATBOT
# =========================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "success": False,
            "message": "Please enter a message"
        }), 400

    message_lower = message.lower()

    if "weather" in message_lower:
        reply = (
            "You can check the Weather Updates section "
            "for temperature, humidity, rainfall and wind details."
        )
    elif "fertilizer" in message_lower:
        reply = (
            "Enter your crop, soil type and NPK values "
            "in Fertilizer Advice to get a recommendation."
        )
    elif "irrigation" in message_lower or "water" in message_lower:
        reply = (
            "Irrigation depends on soil moisture, rainfall "
            "and crop conditions. Check the Smart Irrigation tool."
        )
    elif "crop" in message_lower:
        reply = (
            "Use Crop Recommendation by entering soil and "
            "environmental parameters."
        )
    elif "disease" in message_lower or "leaf" in message_lower:
        reply = (
            "Upload a crop leaf image in Disease Detection "
            "to analyze possible crop health issues."
        )
    else:
        reply = (
            "Hello! I am AgroMind AI Assistant. "
            "I can help with crops, fertilizer, irrigation, "
            "weather and plant diseases."
        )

    return jsonify({
        "success": True,
        "reply": reply
    })


# =========================
# GLOBAL ERROR HANDLER
# =========================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image must be less than 10 MB."
    }), 413


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    print("----------------------------------------")
    print("         AgroMind AI Backend")
    print("----------------------------------------")
    print("Server: http://127.0.0.1:5000")
    print("MongoDB:", "Connected" if mongo_status else "Not Connected")
    print(
        "Disease Model:",
        "Loaded" if disease_session is not None else "Not Loaded"
    )
    print("----------------------------------------")

    # Disable the reloader to avoid starting the model twice.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
