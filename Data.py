import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Set page config
st.set_page_config(
    page_title="Car Type Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Function to ensure arrays are dense (not sparse)
def ensure_dense(X):
    if issparse(X):
        return X.toarray()
    return X

# Function to load or train models
@st.cache_resource
def load_or_train_models(retrain=False):
    # Try to load the models and preprocessor
    if not retrain and os.path.exists("models/models.pkl") and os.path.exists("models/preprocessor.pkl") and os.path.exists("models/label_encoder.pkl"):
        with open("models/models.pkl", "rb") as f:
            models_dict = pickle.load(f)
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Also load model accuracies if available
        if os.path.exists("models/accuracies.pkl"):
            with open("models/accuracies.pkl", "rb") as f:
                accuracies = pickle.load(f)
        else:
            accuracies = {model_name: 0.0 for model_name in models_dict.keys()}
            
        return models_dict, preprocessor, label_encoder, accuracies
    
    # If not found or retrain is True, train the models
    models_dict, preprocessor, label_encoder, accuracies = train_models()
    
    # Save the models for future use
    with open("models/models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/accuracies.pkl", "wb") as f:
        pickle.dump(accuracies, f)
        
    return models_dict, preprocessor, label_encoder, accuracies

# Function to train all models
def train_models():
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data
    status_text.text("Loading dataset...")
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv("Cars2_extended_features5.csv", encoding=encoding)
            break
        except Exception as e:
            continue
    
    if df is None:
        st.error("Could not load the dataset with any of the tried encodings")
        st.stop()
    
    # Prepare data
    status_text.text("Preparing data...")
    progress_bar.progress(10)
    
    X = df[[
        "Engine size (L)", "Cylinders", "Transmission", "Fuel type", 
        "City (L/100 km)", "Highway (L/100 km)", "Combined (L/100 km)",
        "Combined (mpg)", "CO2 emissions (g/km)", "Horsepower", "Torque", 
        "Fuel Tank (L)", "Passenger Capacity"
    ]]
    y = df["Type"]
    
    # Set up label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Setup preprocessor
    categorical_cols = ["Transmission", "Fuel type"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define models
    status_text.text("Setting up models...")
    progress_bar.progress(20)
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=10000, class_weight='balanced', solver='saga', random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights='distance', n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        "Support Vector Machine": SVC(
            class_weight='balanced', probability=True, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
        "Naive Bayes": Pipeline([
            ('ensure_dense', FunctionTransformer(ensure_dense)),
            ('model', GaussianNB())
        ])
    }
    
    # Train the models
    status_text.text("Training models... This might take a few minutes.")
    results = {}
    accuracies = {}
    
    # Train each model
    for i, (name, model) in enumerate(models.items()):
        progress = 20 + (i * 70 / len(models))
        progress_bar.progress(int(progress))
        status_text.text(f"Training {name}...")
        
        try:
            if name == "Naive Bayes":
                # For Naive Bayes, preprocess the data first
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)
                
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)
                
                # Store the model and accuracy
                results[name] = model
                accuracies[name] = accuracy_score(y_test, y_pred)
            else:
                # For other models, use pipeline with preprocessor
                clf = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                # Store the model and accuracy
                results[name] = clf
                accuracies[name] = accuracy_score(y_test, y_pred)
                
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            continue
    
    # Complete the progress
    progress_bar.progress(100)
    status_text.text("Training complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return results, preprocessor, label_encoder, accuracies

# Main app
def main():
    st.title("🚗 Car Type Classifier")
    st.write("Predict the type of car based on its features")
    
    # Sidebar
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox(
        "Choose mode", 
        ["Predict Car Type", "Model Performance", "Train Models"]
    )
    
    # Load or train models
    if app_mode == "Train Models":
        st.header("Train Models")
        st.write("This will train all classification models and save them to disk.")
        
        if st.button("Train Models", key="train_button"):
            models_dict, preprocessor, label_encoder, accuracies = load_or_train_models(retrain=True)
            st.success("Models trained and saved successfully!")
            
            # Display model accuracies
            st.subheader("Model Accuracies")
            accuracies_df = pd.DataFrame({
                'Model': list(accuracies.keys()),
                'Accuracy': list(accuracies.values())
            }).sort_values('Accuracy', ascending=False)
            
            st.dataframe(accuracies_df)
            
            # Plot accuracies using Streamlit
            st.bar_chart(accuracies_df.set_index('Model'))
            
    elif app_mode == "Model Performance":
        st.header("Model Performance")
        
        # Load models first
        try:
            models_dict, preprocessor, label_encoder, accuracies = load_or_train_models()
            
            # Display model accuracies
            st.subheader("Model Accuracies")
            accuracies_df = pd.DataFrame({
                'Model': list(accuracies.keys()),
                'Accuracy': list(accuracies.values())
            }).sort_values('Accuracy', ascending=False)
            
            st.dataframe(accuracies_df)
            
            # Plot accuracies using Streamlit
            st.bar_chart(accuracies_df.set_index('Model'))
            
            # Show feature importance for Random Forest
            if "Random Forest" in models_dict:
                st.subheader("Feature Importance (Random Forest)")
                
                try:
                    # Get Random Forest model
                    rf_model = models_dict["Random Forest"].named_steps['classifier']
                    
                    # Get preprocessor
                    if "Random Forest" in models_dict:
                        preprocessor = models_dict["Random Forest"].named_steps['preprocessor']
                        
                        # Get feature names
                        categorical_cols = ["Transmission", "Fuel type"]
                        numeric_cols = [
                            "Engine size (L)", "Cylinders", "City (L/100 km)", 
                            "Highway (L/100 km)", "Combined (L/100 km)",
                            "Combined (mpg)", "CO2 emissions (g/km)", "Horsepower", 
                            "Torque", "Fuel Tank (L)", "Passenger Capacity"
                        ]
                        
                        try:
                            categorical_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
                            all_features = list(numeric_cols) + list(categorical_features)
                            
                            # Get feature importances
                            importances = rf_model.feature_importances_
                            
                            # Check if lengths match
                            if len(importances) != len(all_features):
                                all_features = [f"Feature {i}" for i in range(len(importances))]
                                
                            # Create feature importance DataFrame
                            feature_importances = pd.DataFrame({
                                'Feature': all_features,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            # Display feature importances
                            st.dataframe(feature_importances)
                            
                            # Plot feature importances using Streamlit
                            st.bar_chart(feature_importances.set_index('Feature').head(15))
                        except Exception as e:
                            st.warning(f"Couldn't extract feature names: {str(e)}")
                    
                except Exception as e:
                    st.warning(f"Error computing feature importance: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please train the models first using the 'Train Models' option.")
    
    else:  # Predict Car Type
        st.header("Predict Car Type")
        
        try:
            # Load models
            models_dict, preprocessor, label_encoder, accuracies = load_or_train_models()
            
            # Get possible car types from the label encoder
            car_types = label_encoder.classes_
            
            # Sort models by accuracy
            sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            model_names = [model[0] for model in sorted_models]
            
            # Model selection
            selected_model = st.selectbox("Select Model", model_names)
            
            # Create two columns for input parameters
            col1, col2 = st.columns(2)
            
            with col1:
                engine_size = st.number_input("Engine size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
                cylinders = st.number_input("Cylinders", min_value=0, max_value=16, value=4, step=1)
                transmission = st.selectbox("Transmission", ["A", "AM", "AS", "AV", "M"])
                fuel_type = st.selectbox("Fuel type", ["D", "E", "N", "X", "Z"])
                city_fuel = st.number_input("City (L/100 km)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
                highway_fuel = st.number_input("Highway (L/100 km)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
            
            with col2:
                combined_fuel = st.number_input("Combined (L/100 km)", min_value=0.0, max_value=30.0, value=9.0, step=0.1)
                combined_mpg = st.number_input("Combined (mpg)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
                co2_emissions = st.number_input("CO2 emissions (g/km)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
                horsepower = st.number_input("Horsepower", min_value=0, max_value=1000, value=200, step=1)
                torque = st.number_input("Torque", min_value=0, max_value=1000, value=200, step=1)
                fuel_tank = st.number_input("Fuel Tank (L)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
                passenger_capacity = st.number_input("Passenger Capacity", min_value=1, max_value=10, value=5, step=1)
            
            # Create input data
            input_data = pd.DataFrame({
                "Engine size (L)": [engine_size],
                "Cylinders": [cylinders],
                "Transmission": [transmission],
                "Fuel type": [fuel_type],
                "City (L/100 km)": [city_fuel],
                "Highway (L/100 km)": [highway_fuel],
                "Combined (L/100 km)": [combined_fuel],
                "Combined (mpg)": [combined_mpg],
                "CO2 emissions (g/km)": [co2_emissions],
                "Horsepower": [horsepower],
                "Torque": [torque],
                "Fuel Tank (L)": [fuel_tank],
                "Passenger Capacity": [passenger_capacity]
            })
            
            # Make prediction
            if st.button("Predict Car Type"):
                if selected_model in models_dict:
                    model = models_dict[selected_model]
                    
                    # Make prediction
                    if selected_model == "Naive Bayes":
                        # For Naive Bayes, preprocess the data first
                        input_data_processed = preprocessor.transform(input_data)
                        prediction = model.predict(input_data_processed)
                    else:
                        # For other models, use the full pipeline
                        prediction = model.predict(input_data)
                    
                    # Display prediction
                    st.success(f"The predicted car type is: **{prediction[0]}**")
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba') or hasattr(model, 'named_steps'):
                        try:
                            if selected_model == "Naive Bayes":
                                probabilities = model.predict_proba(input_data_processed)[0]
                            else:
                                probabilities = model.predict_proba(input_data)[0]
                            
                            # Create DataFrame for probability display
                            prob_df = pd.DataFrame({
                                'Car Type': car_types,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            # Display probabilities
                            st.subheader("Prediction Probabilities")
                            
                            # Create a dataframe for probabilities
                            prob_df = pd.DataFrame({
                                'Car Type': car_types,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            # Display probabilities using Streamlit
                            st.dataframe(prob_df)
                            
                            # Plot probabilities using Streamlit
                            st.bar_chart(prob_df.set_index('Car Type'))
                        except Exception as e:
                            st.info("Probability information not available for this model.")
                else:
                    st.error(f"Model {selected_model} not found.")
        
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please train the models first using the 'Train Models' option.")

if __name__ == "__main__":
    main()
