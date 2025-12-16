# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from predict import MealDemandForecaster  # This line must be working
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Initialize the FastAPI app
# THIS IS THE CRITICAL LINE THAT SOLVES THE ERROR
app = FastAPI(
    title="Meal Demand Forecasting API",
    description="API for predicting meal demand and getting insights."
)

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Load the model ONCE at startup ---
print("Starting API and loading models...")
forecaster = MealDemandForecaster(artifacts_path='artifacts')
print("API ready to accept requests.")

# --- Define request and response models ---
class PredictionInput(BaseModel):
    week: int
    center_id: int
    meal_id: int
    checkout_price: Optional[float] = Field(None, description="Current checkout price")
    base_price: Optional[float] = Field(None, description="Original base price")
    emailer: Optional[int] = Field(0, description="Email promotion sent (1 or 0)")
    homepage: Optional[int] = Field(0, description="Featured on homepage (1 or 0)")

class PredictionResponse(BaseModel):
    week: int
    center_id: int
    meal_id: int
    predicted_orders: float
    explanation: str
    meal_details: dict
    center_details: dict

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Meal Demand Forecasting API. Go to /docs for details."}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(input_data: PredictionInput):
    """
    Get a single demand forecast with an AI-generated explanation.
    """
    # 1. Make the raw prediction
    prediction = forecaster.predict_single(
        week=input_data.week,
        center_id=input_data.center_id,
        meal_id=input_data.meal_id,
        checkout_price=input_data.checkout_price,
        base_price=input_data.base_price,
        emailer=input_data.emailer,
        homepage=input_data.homepage
    )
    
    # 2. Get the AI explanation
    explanation = forecaster.explain_prediction_with_llm(
        center_id=input_data.center_id,
        meal_id=input_data.meal_id,
        week=input_data.week,
        predicted_orders=prediction
    )
    
    # 3. Get helper details
    meal_info = forecaster.meal_df[forecaster.meal_df['meal_id'] == input_data.meal_id].to_dict('records')
    center_info = forecaster.center_df[forecaster.center_df['center_id'] == input_data.center_id].to_dict('records')
    
    return {
        "week": input_data.week,
        "center_id": input_data.center_id,
        "meal_id": input_data.meal_id,
        "predicted_orders": prediction,
        "explanation": explanation,
        "meal_details": meal_info[0] if meal_info else {},
        "center_details": center_info[0] if center_info else {}
    }

@app.get("/insights/feature-importance")
def get_feature_importance():
    """
    Get the top 10 most important features for the model.
    """
    # Convert to JSON-serializable format
    importance_json = forecaster.feature_importance.head(10).to_dict('records')
    return {"feature_importance": importance_json}

@app.get("/insights/top-historical-meals")
def get_top_meals():
    """
    Get the top 5 historically high-demand combinations from train.csv.
    (This logic is from your interactive_query)
    """
    try:
        # We can pre-load this or load on the fly. For simplicity, load on fly.
        original_train = pd.read_csv('train.csv')
        original_train = original_train.merge(forecaster.center_df, on='center_id', how='left')
        original_train = original_train.merge(forecaster.meal_df, on='meal_id', how='left')

        top_predictions = original_train.nlargest(10, 'num_orders')[
            ['week', 'center_id', 'meal_id', 'category', 'cuisine', 'num_orders']
        ].drop_duplicates(subset=['center_id', 'meal_id']).head(5)
        
        return {"top_meals": top_predictions.to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/insights/category-stats")
def get_category_stats():
    """
    Get demand statistics grouped by category and cuisine.
    """
    try:
        original_train = pd.read_csv('train.csv')
        original_train = original_train.merge(forecaster.meal_df, on='meal_id', how='left')
        
        cat_stats = original_train.groupby('category')['num_orders'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
        cui_stats = original_train.groupby('cuisine')['num_orders'].agg(['mean', 'sum']).sort_values('sum', ascending=False)

        return {
            "category_stats": cat_stats.reset_index().to_dict('records'),
            "cuisine_stats": cui_stats.reset_index().to_dict('records')
        }
    except Exception as e:
        return {"error": str(e)}