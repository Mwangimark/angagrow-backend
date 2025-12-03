from PIL import Image
import numpy as np
import cv2  # make sure you have opencv-python installed

def analyze_drone_image(image_path):
    # Open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)

    # Split channels
    R = arr[:, :, 0].astype(float)
    G = arr[:, :, 1].astype(float)
    B = arr[:, :, 2].astype(float)

    eps = 1e-6  # avoid division by zero

    # Vegetation indices
    VARI = np.mean((G - R) / (G + R - B + eps))
    EXG = np.mean(2 * G - R - B)
    GLI = np.mean((2 * G - R - B) / (2 * G + R + B + eps))

    # Canopy cover (% green pixels)
    green_pixels = np.sum(G > R)
    total_pixels = arr.shape[0] * arr.shape[1]
    canopy_pct = (green_pixels / total_pixels) * 100

    # --- Stress detection using HSV ---
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Brown roughly: Hue 10-30, Saturation 50-255, Value 20-200
    brown_pixels = np.sum(
        (h >= 10) & (h <= 30) & 
        (s >= 50) & (s <= 255) & 
        (v >= 20) & (v <= 200)
    )
    stress_pct = (brown_pixels / total_pixels) * 100

    return {
        "vari": round(VARI, 3),
        "exg": round(EXG, 3),
        "gli": round(GLI, 3),
        "canopy_pct": round(canopy_pct, 2),
        "stress_pct": round(stress_pct, 2)
    }


def estimate_yield(canopy, stress):
    """
    Estimate yield based on canopy cover and stress levels.
    canopy: percentage (0–100)
    stress: percentage (0–100)
    """
    # health factor (0 to 1)
    health = (canopy / 100) * (1 - stress / 100)

    # assume maximum possible yield is 6 tons/ha (adjust per crop)
    max_yield = 6.0

    yield_estimate = health * max_yield
    return {
        "yield_estimate": round(yield_estimate, 2)
    }

def generate_chatbot_response(message, latest_data):
    response = ""

    # If no data available
    if latest_data is None:
        response += "I can answer general agriculture questions. "
        response += "Upload a drone image first to get crop-specific insights.\n\n"
    
    else:
        # Use image data
        response += f"Here is what I know from your latest drone image:\n"
        response += f"- **Canopy Cover:** {latest_data.canopy_cover}%\n"
        response += f"- **Stress Level:** {latest_data.stress_percentage}%\n"
        response += f"- **Yield Estimate:** {latest_data.yield_estimate} tons/acre\n\n"

    # Now add general knowledge-based response
    response += handle_general_query(message, latest_data)

    return response


def handle_general_query(message, data):
    msg = message.lower()

    if "stress" in msg:
        if data:
            if data.stress_percentage > 40:
                return "Your crop stress is high. Likely caused by nutrient deficiency, water stress, or pests."
            else:
                return "Your crop stress is moderate. Maintain good watering and monitor nutrient balance."
        else:
            return "Crop stress refers to conditions that reduce crop health — usually caused by water shortage, poor soil, pests, or disease."

    if "canopy" in msg:
        return "Canopy cover shows how much of the field is covered by crops. High canopy = healthy vegetation. Low canopy can indicate stunted growth."

    if "yield" in msg:
        return "Yield estimate is the predicted crop production. It's influenced by canopy cover, stress, soil health, and rainfall."

    if "improve" in msg or "increase" in msg:
        return "To improve yield: ensure adequate nitrogen, manage irrigation, monitor pests weekly, and avoid soil compaction."

    # fallback general response
    return "Great question! This system analyzes crop health. You can ask about stress, canopy cover, yield, or how to improve your crop."



# recommendations after analyzing drone images
def generate_recommendations(analysis_data):
    """
    analysis_data structure:
    {
        "vari": float,
        "exg": float,
        "gli": float,
        "canopy_pct": float,
        "stress_pct": float
    }
    """
    recommendations = []

    # --- Canopy coverage check ---
    canopy = analysis_data.get("avg_canopy_cover", 0)
    if canopy < 40:
        recommendations.append({
            "title": "Low Canopy Coverage",
            "severity": "warning",
            "message": "The canopy coverage is low. Consider improving planting density or checking for early-stage stress.",
            "actions": [
                "Add organic matter to improve soil health",
                "Ensure seeds are evenly spaced",
                "Increase irrigation if the soil is dry"
            ]
        })
    elif canopy > 70:
        recommendations.append({
            "title": "Healthy Canopy Coverage",
            "severity": "success",
            "message": "The canopy is dense and healthy. Maintain current farm management practices.",
            "actions": [
                "Continue regular monitoring",
                "Ensure balanced fertilization to avoid overgrowth"
            ]
        })

    # --- Stress level check ---
    stress = analysis_data.get("avg_stress_percentage", 0)
    if stress > 15:
        recommendations.append({
            "title": "High Vegetation Stress",
            "severity": "danger",
            "message": "Vegetation stress is high. Immediate action is needed.",
            "actions": [
                "Check for pests or diseases",
                "Ensure proper irrigation",
                "Consider nitrogen-rich fertilizer"
            ]
        })
    elif 5 < stress <= 15:
        recommendations.append({
            "title": "Moderate Vegetation Stress",
            "severity": "warning",
            "message": "Some stress detected. Monitor conditions and adjust management where necessary.",
            "actions": [
                "Inspect soil moisture levels",
                "Evaluate weed competition",
            ]
        })

    # --- VARI index check (green vegetation health indicator) ---
    vari = analysis_data.get("avg_vari", 0)
    print(f"VARI value for recommendations: {vari}")
    if vari < 0.2:
        recommendations.append({
            "title": "Low Vegetation Index (VARI)",
            "severity": "warning",
            "message": "Vegetation index is low. Growth may be limited.",
            "actions": [
                "Increase nutrient application",
                "Check water distribution",
                "Verify soil pH levels"
            ]
        })
    else:
        recommendations.append({
            "title": "Good Vegetation Health",
            "severity": "success",
            "message": "The vegetation index suggests healthy crop growth.",
            "actions": [
                "Maintain current practices",
                "Monitor weekly for changes"
            ]
        })

        # --- EXG check (chlorophyll & greenness) ---
    exg = analysis_data.get("avg_exg", 0)
    if exg < 20:
        recommendations.append({
            "title": "Low Greenness (EXG)",
            "severity": "warning",
            "message": "Crops show low greenness. Chlorophyll content may be low.",
            "actions": [
                "Apply nitrogen fertilizer",
                "Check irrigation frequency",
                "Inspect for nutrient deficiency symptoms"
            ]
        })
    elif exg > 50:
        recommendations.append({
            "title": "High Greenness Levels",
            "severity": "success",
            "message": "Your crops display strong green coloration — good sign of health.",
            "actions": [
                "Maintain fertilizer schedule",
                "Monitor for excessive nitrogen use"
            ]
        })

        # --- GLI check (leaf vigor) ---
    gli = analysis_data.get("avg_gli", 0)
    if gli < 0.1:
        recommendations.append({
            "title": "Weak Leaf Vigor",
            "severity": "warning",
            "message": "Leaves show low vigor. Growth may be slowed.",
            "actions": [
                "Check for pests on leaf surfaces",
                "Ensure adequate sunlight exposure",
                "Increase organic compost application"
            ]
        })
    else:
        recommendations.append({
            "title": "Strong Leaf Vigor",
            "severity": "success",
            "message": "Leaf vigor looks good. Plants are actively growing.",
            "actions": [
                "Continue current management",
                "Watch for seasonal stress changes"
            ]
        })
    
        # --- Yield estimate check ---
    yield_est = analysis_data.get("avg_yield_estimate", 0)
    if yield_est < 2:
        recommendations.append({
            "title": "Low Yield Projection",
            "severity": "danger",
            "message": "Expected yield is low. Production may be affected.",
            "actions": [
                "Increase fertilizer efficiency (NPK)",
                "Check plant spacing & density",
                "Inspect for early disease signs"
            ]
        })
    elif 2 <= yield_est < 5:
        recommendations.append({
            "title": "Moderate Yield Projection",
            "severity": "warning",
            "message": "Yield is average. Improvements are possible.",
            "actions": [
                "Improve irrigation uniformity",
                "Monitor nutrient uptake",
            ]
        })
    else:
        recommendations.append({
            "title": "High Yield Projection",
            "severity": "success",
            "message": "Expected yield is high. Great performance!",
            "actions": [
                "Maintain current care",
                "Prepare for upcoming harvest requirements"
            ]
        })
    
        # --- Weed presence probability ---
    if vari < 0.2 and exg > 40:
        recommendations.append({
            "title": "Possible Weed Presence",
            "severity": "warning",
            "message": "Patterns suggest weeds may be present in the field.",
            "actions": [
                "Conduct manual spot checks",
                "Use selective herbicides where needed",
                "Mulch to suppress future weed growth"
            ]
        })
    
        # --- Overall crop health summary ---
    good_signals = 0
    if canopy > 70: good_signals += 1
    if vari > 0.2: good_signals += 1
    if exg > 40: good_signals += 1
    if stress < 10: good_signals += 1

    if good_signals >= 3:
        recommendations.append({
            "title": "Overall Crop Condition: Healthy",
            "severity": "success",
            "message": "Most vegetation indicators show healthy crop performance.",
            "actions": [
                "Maintain your current field management",
                "Monitor stress indicators weekly"
            ]
        })
    else:
        recommendations.append({
            "title": "Overall Crop Condition: Needs Attention",
            "severity": "warning",
            "message": "Multiple indicators suggest your crops require intervention.",
            "actions": [
                "Review irrigation schedule",
                "Carry out a field inspection",
                "Check for pests, diseases, and nutrient deficiency"
            ]
        })

    return recommendations

# +++++++++++++++++++++++++++++++++++++++recommendation cards done+++++++++++++++++++++++++++++++++++++++++++++
