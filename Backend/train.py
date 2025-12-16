# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
import os
import joblib # <-- Import joblib
import json

# ... [PASTE your entire MealDemandForecaster class here] ...
# ... (from line 14 all the way to line 668) ...
class MealDemandForecaster:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.llm_tokenizer = None
        self.llm_model = None

    def load_data(self, train_path, center_path, meal_path):
        """Load all datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv('train.csv')
        self.center_df = pd.read_csv('fulfilment_center_info.csv')
        self.meal_df = pd.read_csv('meal_info.csv')
        print(f"✓ Train data: {self.train_df.shape}")
        print(f"✓ Center data: {self.center_df.shape}")
        print(f"✓ Meal data: {self.meal_df.shape}")

    def engineer_features(self, df, is_train=True):
        """Create advanced features including NLP-based features"""
        if is_train:
            print("\nEngineering features...")

        # Merge with center and meal info
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
        """Prepare data for model training"""
        print("\nPreparing training data...")

        # Engineer features
        self.train_df = self.engineer_features(self.train_df)

        # Encode categorical variables
        self.train_df = self.encode_categorical(self.train_df, fit=True)

        # Define feature columns
        exclude_cols = ['id', 'num_orders']
        self.feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]

        # Prepare X and y
        X = self.train_df[self.feature_cols]
        y = self.train_df['num_orders']

        print(f"✓ Feature matrix: {X.shape}")
        print(f"✓ Target variable: {y.shape}")

        return X, y

    def train_model(self, X, y):
        """Train XGBoost model"""
        print("\nTraining XGBoost model...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # XGBoost parameters
        params = {
            'objective': 'reg:squaredlogerror',  # For RMSLE
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
            'early_stopping_rounds': 50  # Moved to params
        }

        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        # Calculate RMSLE on validation set
        y_pred = self.model.predict(X_val)
        y_pred = np.maximum(0, y_pred)  # Ensure non-negative predictions
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_val))**2))

        print(f"\n✓ Model trained successfully!")
        print(f"✓ Validation RMSLE: {rmsle:.4f}")
        print(f"✓ Competition Score: {100 * rmsle:.2f}")

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(self.feature_importance.head(10))

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
        """Make predictions on test data"""
        print("\nMaking predictions...")

        # Engineer features
        test_df = self.engineer_features(test_df)

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

        test_df['predicted_orders'] = predictions

        print(f"✓ Predictions completed: {len(predictions)} records")
        print(f"✓ Average predicted demand: {predictions.mean():.2f}")
        print(f"✓ Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

        return test_df

    def explain_prediction_with_llm(self, center_id, meal_id, week, predicted_orders):
        """Use FLAN-T5 to generate natural language explanation"""

        # Get meal and center info
        meal_info = None
        center_info = None

        if hasattr(self, 'meal_df'):
            meal_matches = self.meal_df[self.meal_df['meal_id'] == meal_id]
            if len(meal_matches) > 0:
                meal_info = meal_matches.iloc[0]

        if hasattr(self, 'center_df'):
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
        """Process natural language queries about forecasts"""
        user_input_lower = user_input.lower()

    def interactive_query(self, user_input):
        """Process natural language queries about forecasts"""
        user_input_lower = user_input.lower()

        # Parse intent - FORECAST
        if 'forecast' in user_input_lower or 'predict' in user_input_lower:
            import re
            numbers = re.findall(r'\d+', user_input)

            if len(numbers) >= 3:
                week, center, meal = int(numbers[0]), int(numbers[1]), int(numbers[2])

                # First, try to look up in existing predictions
                prediction_found = False
                pred_value = 0
                source = ""

                if hasattr(self, 'predictions_df'):
                    result = self.predictions_df[
                        (self.predictions_df['week'] == week) &
                        (self.predictions_df['center_id'] == center) &
                        (self.predictions_df['meal_id'] == meal)
                    ]

                    if len(result) > 0:
                        prediction_found = True
                        pred_value = result.iloc[0]['predicted_orders']
                        source = "test data"

                # If not found, make on-the-fly prediction
                if not prediction_found:
                    try:
                        pred_value = self.predict_single(week, center, meal)
                        source = "on-the-fly prediction"
                        prediction_found = True
                    except Exception as e:
                        return f"❌ Could not make prediction: {e}\n\nTry with valid center and meal IDs."

                # Build response
                response = f"📊 DEMAND FORECAST:\n\n"
                response += f"  Week: {week}\n"
                response += f"  Center ID: {center}\n"
                response += f"  Meal ID: {meal}\n"

                # Get meal info if available
                meal_info = self.meal_df[self.meal_df['meal_id'] == meal]
                if len(meal_info) > 0:
                    response += f"  Category: {meal_info.iloc[0]['category']}\n"
                    response += f"  Cuisine: {meal_info.iloc[0]['cuisine']}\n"

                # Get center info if available
                center_info = self.center_df[self.center_df['center_id'] == center]
                if len(center_info) > 0:
                    response += f"  City: {center_info.iloc[0]['city_code']}\n"
                    response += f"  Region: {center_info.iloc[0]['region_code']}\n"

                response += f"\n  🎯 PREDICTED ORDERS: {pred_value:.0f} orders/week\n"
                response += f"     (≈ {pred_value/7:.0f} orders/day)\n"
                response += f"     Source: {source}\n\n"

                # Add recommendation based on demand level
                if pred_value < 100:
                    response += "  📦 LOW Demand\n"
                    response += "  └─ Recommendation: Minimal stock, basic staffing\n"
                    response += "  └─ Risk: Low waste potential\n"
                elif pred_value < 200:
                    response += "  📦 MODERATE Demand\n"
                    response += "  └─ Recommendation: Standard inventory levels\n"
                    response += "  └─ Risk: Medium - monitor closely\n"
                elif pred_value < 400:
                    response += "  📦 HIGH Demand\n"
                    response += "  └─ Recommendation: Increase inventory by 30-40%\n"
                    response += "  └─ Risk: High - ensure adequate staffing\n"
                    response += "  └─ Action: Schedule 2-3 extra staff members\n"
                else:
                    response += "  📦 VERY HIGH Demand\n"
                    response += "  └─ Recommendation: Maximum inventory preparation\n"
                    response += "  └─ Risk: Critical - stock out potential\n"
                    response += "  └─ Action: All hands on deck, prepare 50% extra\n"

                # Add cost/revenue estimates
                avg_order_value = 250
                response += f"\n  💰 Business Impact:\n"
                response += f"     • Expected Revenue: ₹{pred_value * avg_order_value:,.0f}/week\n"
                response += f"     • Raw Material Budget: ₹{pred_value * avg_order_value * 0.35:,.0f}/week\n"

                return response

            # If not enough numbers provided
            response = "🤖 I can forecast demand for any combination!\n\n"
            response += "Please provide:\n"
            response += "  • Week number (1-155, or future weeks)\n"
            response += "  • Center ID (check your data for valid IDs)\n"
            response += "  • Meal ID (check your data for valid IDs)\n\n"
            response += "Example: 'Forecast for week 150, center 10, meal 1885'\n\n"
            response += "💡 Tip: Type 'sample' to see valid combinations from test data"
            return response

        # Parse intent - TOP/BEST
        if 'top' in user_input_lower or 'best' in user_input_lower or 'high' in user_input_lower:
            try:
                original_train = pd.read_csv('train.csv')
                original_train = original_train.merge(self.center_df, on='center_id', how='left')
                original_train = original_train.merge(self.meal_df, on='meal_id', how='left')

                top_predictions = original_train.nlargest(10, 'num_orders')[
                    ['week', 'center_id', 'meal_id', 'category', 'cuisine', 'num_orders']
                ].drop_duplicates(subset=['center_id', 'meal_id']).head(5)

                response = "🏆 Top 5 Historical High-Demand Combinations:\n\n"
                for idx, row in top_predictions.iterrows():
                    response += f"  {idx+1}. Center {int(row['center_id'])} - {row['category']} ({row['cuisine']})\n"
                    response += f"     → {int(row['num_orders'])} orders/week (Week {int(row['week'])})\n\n"
                return response
            except:
                return "Top performers data not available yet. Train the model first!"

        # Parse intent - FEATURES
        if 'feature' in user_input_lower or 'important' in user_input_lower:
            if hasattr(self, 'feature_importance'):
                response = "🔍 Most Important Features for Prediction:\n\n"
                for idx, row in self.feature_importance.head(8).iterrows():
                    feature = row['feature']
                    if 'demand_density' in feature:
                        explanation = "(Orders per square km)"
                    elif 'ewm' in feature:
                        explanation = "(Recent trend)"
                    elif 'rolling_mean' in feature:
                        explanation = "(Moving average)"
                    elif 'lag' in feature:
                        explanation = "(Past week's orders)"
                    elif 'price' in feature or 'discount' in feature:
                        explanation = "(Pricing impact)"
                    elif 'promotion' in feature or 'emailer' in feature:
                        explanation = "(Marketing effect)"
                    else:
                        explanation = ""

                    response += f"  {idx+1}. {row['feature']}: {row['importance']:.4f} {explanation}\n"
                response += "\n💡 These features have the biggest impact on demand forecasts!"
                return response
            return "Feature importance not available. Train the model first!"

        # Parse intent - SUMMARY
        if 'summary' in user_input_lower or 'overview' in user_input_lower:
            response = "📈 FORECAST SUMMARY:\n\n"

            try:
                original_train = pd.read_csv('train.csv')
                original_train = original_train.merge(self.center_df, on='center_id', how='left')
                original_train = original_train.merge(self.meal_df, on='meal_id', how='left')

                response += f"  📊 Dataset Statistics:\n"
                response += f"     • Total historical records: {len(original_train):,}\n"
                response += f"     • Average weekly demand: {original_train['num_orders'].mean():.2f} orders\n"
                response += f"     • Peak demand: {original_train['num_orders'].max():.0f} orders\n"
                response += f"     • Unique centers: {original_train['center_id'].nunique()}\n"
                response += f"     • Unique meals: {original_train['meal_id'].nunique()}\n"
                response += f"     • Date range: Week {original_train['week'].min()} to {original_train['week'].max()}\n\n"

                if hasattr(self, 'predictions_df'):
                    response += f"  🎯 Predictions Generated:\n"
                    response += f"     • Total predictions: {len(self.predictions_df):,}\n"
                # This was the line with the error, now corrected:
                    response += f"     • Forecast weeks: {self.predictions_df['week'].min()} to {self.predictions_df['week'].max()}\n"
                    response += f"     • Avg predicted demand: {self.predictions_df['predicted_orders'].mean():.2f} orders/week\n"
                    response += f"     • Peak predicted: {self.predictions_df['predicted_orders'].max():.0f} orders\n"
                return response
            except:
                return "Could not load summary data."
        # Parse intent - CATEGORY/CUISINE
        if 'category' in user_input_lower or 'cuisine' in user_input_lower:
            try:
                original_train = pd.read_csv('train.csv')
                original_train = original_train.merge(self.meal_df, on='meal_id', how='left')

                response = "🍽️ MEAL CATEGORIES & CUISINES:\n\n"

                cat_stats = original_train.groupby('category')['num_orders'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
                response += "  Top Categories by Total Demand:\n"
                for cat, row in cat_stats.head(5).iterrows():
                    response += f"    • {cat}: {row['sum']:,.0f} total orders (avg: {row['mean']:.1f}/week)\n"

                response += "\n  Top Cuisines by Total Demand:\n"
                cui_stats = original_train.groupby('cuisine')['num_orders'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
                for cui, row in cui_stats.head(5).iterrows():
                    response += f"    • {cui}: {row['sum']:,.0f} total orders (avg: {row['mean']:.1f}/week)\n"
                return response
            except:
                return "Could not load category/cuisine data."

        # Parse intent - SAMPLE
        if 'sample' in user_input_lower or 'show prediction' in user_input_lower:
            if hasattr(self, 'predictions_df') and len(self.predictions_df) > 0:
                response = "📋 SAMPLE PREDICTIONS:\n\n"
                samples = self.predictions_df.head(10)
                for idx, row in samples.iterrows():
                    response += f"  • Week {int(row['week'])}, Center {int(row['center_id'])}, Meal {int(row['meal_id'])}\n"
                    response += f"    → {row['predicted_orders']:.0f} orders/week\n"
                return response
            return "No predictions available. Run model with test data first."

        # Parse intent - HELP
        if 'help' in user_input_lower or '?' in user_input_lower:
            response = "🤖 AVAILABLE COMMANDS:\n\n"
            response += "  📊 'forecast for week X, center Y, meal Z' - Get specific prediction\n"
            response += "  📋 'show predictions' or 'sample' - See sample forecasts\n"
            response += "  🏆 'top' or 'best' - See high-performing meals\n"
            response += "  🔍 'features' or 'important' - View key prediction factors\n"
            response += "  📈 'summary' or 'overview' - Get data overview\n"
            response += "  🍽️ 'category' or 'cuisine' - Analyze meal types\n"
            response += "  💡 'example' - Get AI-generated explanation\n"
            response += "  ❌ 'quit' or 'exit' - End session\n"
            return response

        # Default response
        response = "I can help you with:\n\n"
        response += "  1. 📊 Specific demand forecasts (provide week, center, meal IDs)\n"
        response += "  2. 📋 Sample predictions from the test data\n"
        response += "  3. 🏆 Top performing meal-center combinations\n"
        response += "  4. 🔍 Important features affecting demand\n"
        response += "  5. 📈 Data summaries and overviews\n"
        response += "  6. 🍽️ Category and cuisine analysis\n"
        response += "\nType 'help' for detailed commands or ask any question!"

        return response
def main():
    print("=" * 70)
    print("STARTING MODEL TRAINING & ARTIFACT CREATION")
    print("=" * 70)

    # --- Create artifact directory ---
    ARTIFACTS_DIR = 'artifacts'
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    # --- Initialize and load data ---
    forecaster = MealDemandForecaster()
    try:
        forecaster.load_data(
            'train.csv',
            'fulfilment_center_info.csv',
            'meal_info.csv'
        )

        # --- Prepare and train ---
        X, y = forecaster.prepare_training_data()
        forecaster.train_model(X, y)
        print("\n✓ Model training complete.")

        # --- Save the artifacts ---
        
        # 1. Save the XGBoost model
        model_path = os.path.join(ARTIFACTS_DIR, 'xgb_model.json')
        forecaster.model.save_model(model_path)
        print(f"✓ XGBoost model saved to {model_path}")

        # 2. Save the Label Encoders
        encoder_path = os.path.join(ARTIFACTS_DIR, 'label_encoders.joblib')
        joblib.dump(forecaster.label_encoders, encoder_path)
        print(f"✓ Label encoders saved to {encoder_path}")

        # 3. Save the feature columns
        features_path = os.path.join(ARTIFACTS_DIR, 'feature_cols.json')
        with open(features_path, 'w') as f:
            json.dump(forecaster.feature_cols, f)
        print(f"✓ Feature columns saved to {features_path}")

        # 4. Save the historical stats (for filling test data)
        stats = {
            'center_stats': forecaster.center_stats,
            'meal_stats': forecaster.meal_stats,
            'category_stats': forecaster.category_stats,
            'cuisine_stats': forecaster.cuisine_stats,
            'center_type_stats': forecaster.center_type_stats,
            'region_stats': forecaster.region_stats,
        }
        stats_path = os.path.join(ARTIFACTS_DIR, 'feature_stats.joblib')
        joblib.dump(stats, stats_path)
        print(f"✓ Feature stats saved to {stats_path}")
        
        # 5. Save other info for the API
        joblib.dump(forecaster.meal_df, os.path.join(ARTIFACTS_DIR, 'meal_df.joblib'))
        joblib.dump(forecaster.center_df, os.path.join(ARTIFACTS_DIR, 'center_df.joblib'))
        joblib.dump(forecaster.feature_importance, os.path.join(ARTIFACTS_DIR, 'feature_importance.joblib'))
        print("✓ Helper dataframes saved.")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE. Artifacts are ready for the API.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()