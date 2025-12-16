# predict.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
import os
import joblib
import json

warnings.filterwarnings('ignore')

class MealDemandForecaster:
    def __init__(self, artifacts_path='artifacts'):
        self.model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.llm_tokenizer = None
        self.llm_model = None
        
        # --- Add all the stats properties ---
        self.center_stats = {}
        self.meal_stats = {}
        self.category_stats = {}
        self.cuisine_stats = {}
        self.center_type_stats = {}
        self.region_stats = {}
        
        # --- Add helper dataframes ---
        self.meal_df = None
        self.center_df = None
        self.feature_importance = None
        
        # --- Load all artifacts ---
        self.load_artifacts(artifacts_path)
        # --- This is the line that was failing ---
        self.load_llm() # Load LLM on init

    def load_artifacts(self, artifacts_path):
        print("Loading artifacts...")
        try:
            # 1. Load XGB model
            model_path = os.path.join(artifacts_path, 'xgb_model.json')
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            print("✓ Model loaded.")

            # 2. Load Label Encoders
            encoder_path = os.path.join(artifacts_path, 'label_encoders.joblib')
            self.label_encoders = joblib.load(encoder_path)
            print("✓ Encoders loaded.")

            # 3. Load feature columns
            features_path = os.path.join(artifacts_path, 'feature_cols.json')
            with open(features_path, 'r') as f:
                self.feature_cols = json.load(f)
            print("✓ Feature columns loaded.")

            # 4. Load historical stats
            stats_path = os.path.join(artifacts_path, 'feature_stats.joblib')
            stats = joblib.load(stats_path)
            self.center_stats = stats.get('center_stats', {})
            self.meal_stats = stats.get('meal_stats', {})
            self.category_stats = stats.get('category_stats', {})
            self.cuisine_stats = stats.get('cuisine_stats', {})
            self.center_type_stats = stats.get('center_type_stats', {})
            self.region_stats = stats.get('region_stats', {})
            print("✓ Feature stats loaded.")
            
            # 5. Load helper data
            self.meal_df = joblib.load(os.path.join(artifacts_path, 'meal_df.joblib'))
            self.center_df = joblib.load(os.path.join(artifacts_path, 'center_df.joblib'))
            self.feature_importance = joblib.load(os.path.join(artifacts_path, 'feature_importance.joblib'))
            print("✓ Helper data loaded.")
            print("Artifacts loading complete.")
        except FileNotFoundError as e:
            print(f"❌ ARTIFACT LOADING ERROR: {e}")
            print("Please make sure the 'artifacts' folder exists and contains all model files.")
            print("Run train.py first!")
        except Exception as e:
            print(f"❌ An error occurred during artifact loading: {e}")


    def load_data(self, train_path, center_path, meal_path):
        """Load all datasets"""
        # This function is not really used by the predict class, 
        # but we keep it for consistency. The data is loaded from artifacts.
        print("Loading datasets...")
        if self.center_df is None:
             self.center_df = pd.read_csv(center_path)
        if self.meal_df is None:
             self.meal_df = pd.read_csv(meal_path)
        print("✓ Datasets loaded (from artifacts or file)")


    def engineer_features(self, df, is_train=True):
        """Create advanced features including NLP-based features"""
        if is_train:
            print("\nEngineering features...")

        # Merge with center and meal info
        # Add checks for loaded dataframes
        if self.center_df is None or self.meal_df is None:
             print("Warning: center_df or meal_df not loaded. Merging might fail.")
             # Attempt to load them if they weren't
             if not os.path.exists('fulfilment_center_info.csv') or not os.path.exists('meal_info.csv'):
                 print("Error: Cannot find CSV files to merge.")
                 return df # return early
             self.center_df = pd.read_csv('fulfilment_center_info.csv')
             self.meal_df = pd.read_csv('meal_info.csv')
        
        df = df.merge(self.center_df, on='center_id', how='left')
        df = df.merge(self.meal_df, on='meal_id', how='left')

        # Time-based features
        df['week_mod_52'] = df['week'] % 52
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        df['is_holiday_season'] = ((df['week_mod_52'] >= 48) | (df['week_mod_52'] <= 2)).astype(int)
        df['quarter'] = (df['week'] % 52) // 13 + 1

        # Price features
        df['discount_pct'] = ((df['base_price'] - df['checkout_price']) / df['base_price']) * 100
        df['discount_pct'] = df['discount_pct'].clip(0, 100)
        df['price_per_unit'] = df['checkout_price']
        df['is_discounted'] = (df['discount_pct'] > 0).astype(int)

        # Marketing features
        df['total_promotion'] = df['emailer_for_promotion'] + df['homepage_featured']
        df['promo_and_discount'] = df['emailer_for_promotion'] * df['is_discounted']

        # Lag features and rolling statistics (only if num_orders exists)
        if 'num_orders' in df.columns:
            # Lag features (historical demand)
            for lag in [1, 2, 3, 4, 8]:
                df[f'lag_{lag}'] = df.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)

            # Rolling statistics
            for window in [3, 4, 8]:
                df[f'rolling_mean_{window}'] = df.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'rolling_std_{window}'] = df.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )

            # Exponential weighted moving average
            df['ewm_demand'] = df.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                lambda x: x.ewm(span=4, adjust=False).mean()
            )

            # Center-based aggregations
            df['center_avg_demand'] = df.groupby('center_id')['num_orders'].transform('mean')
            df['center_std_demand'] = df.groupby('center_id')['num_orders'].transform('std')

            # Meal-based aggregations
            df['meal_avg_demand'] = df.groupby('meal_id')['num_orders'].transform('mean')
            df['meal_std_demand'] = df.groupby('meal_id')['num_orders'].transform('std')

            # Category-based aggregations
            df['category_avg_demand'] = df.groupby('category')['num_orders'].transform('mean')
            df['cuisine_avg_demand'] = df.groupby('cuisine')['num_orders'].transform('mean')

            # Center type and region interactions
            df['center_type_avg'] = df.groupby('center_type')['num_orders'].transform('mean')
            df['region_avg'] = df.groupby('region_code')['num_orders'].transform('mean')

            # Demand density (orders per operational area)
            df['demand_density'] = df['num_orders'] / (df['op_area'] + 1)
        else:
            # For test data, use historical statistics from training data
            # Create lag features with zeros (will be filled with historical data if available)
            for lag in [1, 2, 3, 4, 8]:
                df[f'lag_{lag}'] = 0

            # Rolling statistics with zeros
            for window in [3, 4, 8]:
                df[f'rolling_mean_{window}'] = 0
                df[f'rolling_std_{window}'] = 0

            df['ewm_demand'] = 0

            # Use stored aggregations from training
            if hasattr(self, 'center_stats'):
                df['center_avg_demand'] = df['center_id'].map(self.center_stats.get('avg', {}))
                df['center_std_demand'] = df['center_id'].map(self.center_stats.get('std', {}))
            else:
                df['center_avg_demand'] = 0
                df['center_std_demand'] = 0

            if hasattr(self, 'meal_stats'):
                df['meal_avg_demand'] = df['meal_id'].map(self.meal_stats.get('avg', {}))
                df['meal_std_demand'] = df['meal_id'].map(self.meal_stats.get('std', {}))
            else:
                df['meal_avg_demand'] = 0
                df['meal_std_demand'] = 0

            if hasattr(self, 'category_stats'):
                df['category_avg_demand'] = df['category'].map(self.category_stats)
                df['cuisine_avg_demand'] = df['cuisine'].map(self.cuisine_stats)
            else:
                df['category_avg_demand'] = 0
                df['cuisine_avg_demand'] = 0

            if hasattr(self, 'center_type_stats'):
                df['center_type_avg'] = df['center_type'].map(self.center_type_stats)
                df['region_avg'] = df['region_code'].map(self.region_stats)
            else:
                df['center_type_avg'] = 0
                df['region_avg'] = 0

            df['demand_density'] = 0

        # NLP-based features: Encode category and cuisine combinations
        df['category_cuisine'] = df['category'] + '_' + df['cuisine']
        df['center_meal_combo'] = df['center_id'].astype(str) + '_' + df['meal_id'].astype(str)

        # Fill NaN values
        df = df.fillna(0)

        if is_train:
            # Store statistics for test data
            if 'num_orders' in df.columns:
                self.center_stats = {
                    'avg': df.groupby('center_id')['num_orders'].mean().to_dict(),
                    'std': df.groupby('center_id')['num_orders'].std().to_dict()
                }
                self.meal_stats = {
                    'avg': df.groupby('meal_id')['num_orders'].mean().to_dict(),
                    'std': df.groupby('meal_id')['num_orders'].std().to_dict()
                }
                self.category_stats = df.groupby('category')['num_orders'].mean().to_dict()
                self.cuisine_stats = df.groupby('cuisine')['num_orders'].mean().to_dict()
                self.center_type_stats = df.groupby('center_type')['num_orders'].mean().to_dict()
                self.region_stats = df.groupby('region_code')['num_orders'].mean().to_dict()

            print(f"✓ Features engineered: {df.shape[1]} total columns")

        return df

    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        categorical_cols = ['center_id', 'meal_id', 'city_code', 'region_code',
                           'center_type', 'category', 'cuisine', 'category_cuisine',
                           'center_meal_combo']

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    # Store original values before encoding
                    if col in ['category', 'cuisine']:
                        if not hasattr(self, 'original_mappings'):
                            self.original_mappings = {}
                        self.original_mappings[col] = dict(zip(
                            df[col].unique(),
                            df[col].unique()
                        ))
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        df[col] = df[col].astype(str).map(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_ else -1
                        )
        return df

    def prepare_training_data(self):
        """This method is part of train.py, not needed for predict.py"""
        pass

    def train_model(self, X, y):
        """This method is part of train.py, not needed for predict.py"""
        pass

    # --- THIS IS THE MISSING METHOD ---
    def load_llm(self):
        """Load FLAN-T5 for natural language interaction"""
        print("\nLoading FLAN-T5 model for natural language interaction...")
        try:
            # Using FLAN-T5 small for faster inference
            model_name = "google/flan-t5-small"
            self.llm_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.llm_model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("✓ FLAN-T5 loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load FLAN-T5: {e}")
            print("  Continuing without LLM features...")

    def predict_single(self, week, center_id, meal_id, checkout_price=None, base_price=None,
                      emailer=0, homepage=0):
        """Make prediction for a single combination on-the-fly"""

        # Create a single row dataframe
        single_row = pd.DataFrame({
            'week': [week],
            'center_id': [center_id],
            'meal_id': [meal_id],
            'checkout_price': [checkout_price if checkout_price else 200],  # Default price
            'base_price': [base_price if base_price else 250],
            'emailer_for_promotion': [emailer],
            'homepage_featured': [homepage]
        })

        # Add a temporary id
        single_row['id'] = [999999]

        # Engineer features
        single_row = self.engineer_features(single_row, is_train=False)

        # Encode categorical
        single_row = self.encode_categorical(single_row, fit=False)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in single_row.columns:
                single_row[col] = 0

        # Prepare features
        X_single = single_row[self.feature_cols]

        # Make prediction
        prediction = self.model.predict(X_single)[0]
        prediction = max(0, prediction)

        return prediction

    def predict(self, test_df):
        """Make predictions on test data"""
        print("\nMaking predictions on test data...")

        # Store original test data before transformations
        original_test = test_df.copy()

        # Engineer features (with is_train=False)
        test_df = self.engineer_features(test_df, is_train=False)

        # Encode categorical variables
        test_df = self.encode_categorical(test_df, fit=False)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in test_df.columns:
                test_df[col] = 0

        X_test = test_df[self.feature_cols]

        # Make predictions
        predictions = self.model.predict(X_test)
        predictions = np.maximum(0, predictions)  # Ensure non-negative

        # Add predictions to ORIGINAL test data (before encoding)
        original_test['predicted_orders'] = predictions

        print(f"✓ Predictions completed: {len(predictions)} records")
        print(f"✓ Average predicted demand: {predictions.mean():.2f}")
        print(f"✓ Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

        return original_test

    def explain_prediction_with_llm(self, center_id, meal_id, week, predicted_orders):
        """Use FLAN-T5 to generate natural language explanation"""

        # Get meal and center info
        meal_info = None
        center_info = None

        if hasattr(self, 'meal_df') and self.meal_df is not None:
            meal_matches = self.meal_df[self.meal_df['meal_id'] == meal_id]
            if len(meal_matches) > 0:
                meal_info = meal_matches.iloc[0]

        if hasattr(self, 'center_df') and self.center_df is not None:
            center_matches = self.center_df[self.center_df['center_id'] == center_id]
            if len(center_matches) > 0:
                center_info = center_matches.iloc[0]

        # Create detailed context
        category = meal_info['category'] if meal_info is not None else 'meal'
        cuisine = meal_info['cuisine'] if meal_info is not None else 'various cuisine'
        city = center_info['city_code'] if center_info is not None else 'the city'
        center_type = center_info['center_type'] if center_info is not None else 'standard'

        # Determine demand level
        if predicted_orders < 100:
            demand_level = "low"
            recommendation = "Consider reducing inventory and optimizing promotions"
        elif predicted_orders < 200:
            demand_level = "moderate"
            recommendation = "Maintain standard stock levels"
        elif predicted_orders < 400:
            demand_level = "high"
            recommendation = "Increase inventory and ensure adequate staffing"
        else:
            demand_level = "very high"
            recommendation = "Maximize inventory, schedule extra staff, and prepare for peak demand"

        context = f"""Week {week} forecast for {category} ({cuisine}) at center {center_id} in city {city}:
Expected orders: {int(predicted_orders)}
Demand level: {demand_level}
Center type: {center_type}
Action: {recommendation}"""

        if self.llm_model and self.llm_tokenizer:
            try:
                # Generate explanation using FLAN-T5
                prompt = f"Explain this restaurant delivery forecast clearly and professionally: {context}"
                inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.llm_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=False
                )
                explanation = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Add actionable insights
                full_explanation = f"{explanation}\n\n"
                full_explanation += f"📦 Procurement Planning: {recommendation}\n"
                full_explanation += f"🎯 Demand Level: {demand_level.upper()} ({int(predicted_orders)} orders)\n"
                full_explanation += f"📍 Location: Center {center_id}, City {city}\n"
                full_explanation += f"🍽️ Product: {category} - {cuisine}"

                return full_explanation
            except Exception as e:
                print(f"⚠ LLM generation error: {e}")
                # Fall through to fallback

        # Fallback explanation with rich details
        explanation = f"For week {week}, our model predicts {demand_level} demand with approximately {int(predicted_orders)} orders "
        explanation += f"for {category} ({cuisine} cuisine) at fulfillment center {center_id} in city {city}.\n\n"
        explanation += f"📦 Recommended Action: {recommendation}\n"
        explanation += f"🏢 Center Type: {center_type}\n"
        explanation += f"📊 This forecast helps optimize inventory, staffing, and operational planning."

        return explanation

    def interactive_query(self, user_input):
        """This method is not used by the API directly, but we keep it."""
        pass