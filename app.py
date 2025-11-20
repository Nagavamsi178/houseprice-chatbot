from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting House Price Prediction API")

# Global variables
model = None
model_loaded = False
feature_names = []

# Load model
try:
    model_files = ['housing_price_model.pkl', 'housing_price_model (2).pkl', 'model.pkl']
    model_file = None
    
    for file in model_files:
        if os.path.exists(file):
            model_file = file
            break
    
    if model_file:
        print(f"📁 Loading model: {model_file}")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
        print("✅ Model loaded successfully!")
        
        # Get feature info
        if hasattr(model, 'n_features_in_'):
            print(f"🎯 Expected features: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            print(f"📝 Feature names: {feature_names}")
    else:
        print("❌ No model file found")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False

# Define your exact feature names
FEATURE_NAMES = [
    'Property_Type', 'BHK', 'Furnished_Status', 'Floor_No', 'Total_Floors', 
    'Age_of_Property', 'Public_Transport_Accessibility', 'Parking_Space', 
    'Security', 'Facing', 'Owner_Type', 'Availability_Status', 'Price_for_Area', 
    'School_and_Hospitals', 'State_City', 'clubhouse', 'garden', 'gym', 
    'playground', 'pool'
]

# Request model with all 20 features
class HouseFeatures(BaseModel):
    Property_Type: float
    BHK: float
    Furnished_Status: float
    Floor_No: float
    Total_Floors: float
    Age_of_Property: float
    Public_Transport_Accessibility: float
    Parking_Space: float
    Security: float
    Facing: float
    Owner_Type: float
    Availability_Status: float
    Price_for_Area: float
    School_and_Hospitals: float
    State_City: float
    clubhouse: float
    garden: float
    gym: float
    playground: float
    pool: float

class PredictionResponse(BaseModel):
    prediction: float
    status: str
    message: str

@app.get("/")
async def root():
    return {
        "message": "🏠 House Price Prediction API",
        "status": "running",
        "model_loaded": model_loaded,
        "features": FEATURE_NAMES,
        "total_features": len(FEATURE_NAMES)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """
    Predict house price based on 20 input features
    """
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    try:
        # Convert to dictionary
        input_data = features.dict()
        print("📨 Received prediction request")
        
        # Create feature array in exact order expected by model
        feature_values = []
        for feature_name in FEATURE_NAMES:
            value = input_data[feature_name]
            feature_values.append(float(value))
        
        print(f"✅ Features processed: {len(feature_values)} values")
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(feature_values, dtype=np.float64).reshape(1, -1)
        
        # Make prediction
        prediction_result = model.predict(features_array)
        predicted_price = float(prediction_result[0])
        
        print(f"💰 Prediction successful: ₹{predicted_price:,.2f}")
        
        return PredictionResponse(
            prediction=predicted_price,
            status="success",
            message=f"Predicted house price: ₹{predicted_price:,.2f}"
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        print(f"🔍 Error details: {error_details}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_type": str(type(model)) if model else "None",
        "features_expected": len(FEATURE_NAMES)
    }

@app.get("/features")
async def get_features():
    """Get information about all expected features"""
    return {
        "total_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "example_request": {
            "Property_Type": 1.0,
            "BHK": 3.0,
            "Furnished_Status": 2.0,
            "Floor_No": 5.0,
            "Total_Floors": 10.0,
            "Age_of_Property": 5.0,
            "Public_Transport_Accessibility": 8.0,
            "Parking_Space": 2.0,
            "Security": 1.0,
            "Facing": 2.0,
            "Owner_Type": 1.0,
            "Availability_Status": 1.0,
            "Price_for_Area": 8000.0,
            "School_and_Hospitals": 7.0,
            "State_City": 25.0,
            "clubhouse": 1.0,
            "garden": 1.0,
            "gym": 1.0,
            "playground": 1.0,
            "pool": 0.0
        }
    }

@app.get("/example-curl")
async def example_curl():
    """Get example curl command for testing"""
    example_data = {
        "Property_Type": 1.0,
        "BHK": 3.0,
        "Furnished_Status": 2.0,
        "Floor_No": 5.0,
        "Total_Floors": 10.0,
        "Age_of_Property": 5.0,
        "Public_Transport_Accessibility": 8.0,
        "Parking_Space": 2.0,
        "Security": 1.0,
        "Facing": 2.0,
        "Owner_Type": 1.0,
        "Availability_Status": 1.0,
        "Price_for_Area": 8000.0,
        "School_and_Hospitals": 7.0,
        "State_City": 25.0,
        "clubhouse": 1.0,
        "garden": 1.0,
        "gym": 1.0,
        "playground": 1.0,
        "pool": 0.0
    }
    
    curl_command = f"""curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -H "accept: application/json" \\
  -d '{str(example_data).replace("'", '"')}'"""
  
    return {
        "curl_command": curl_command,
        "example_data": example_data
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)