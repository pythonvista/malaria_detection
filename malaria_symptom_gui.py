import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import os

class MalariaSymptomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Malaria Symptom-Based Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load models and data
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.load_models()
        
        # Current data variables
        self.current_features = None
        
        self.setup_ui()
        
    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Load feature names
            if os.path.exists("malaria_symptom_features.joblib"):
                self.feature_names = joblib.load("malaria_symptom_features.joblib")
                print("Feature names loaded successfully")
            else:
                print("Feature names file not found")
                return
            
            # Load scaler
            if os.path.exists("malaria_symptom_scaler.joblib"):
                self.scaler = joblib.load("malaria_symptom_scaler.joblib")
                print("Scaler loaded successfully")
            
            # Load models
            model_files = {
                'Decision Tree': 'malaria_symptom_decision_tree.joblib',
                'SVM': 'malaria_symptom_svm.joblib',
                'Logistic Regression': 'malaria_symptom_logistic_regression.joblib',
                'Random Forest': 'malaria_symptom_random_forest.joblib'
            }
            
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
                    print(f"{model_name} model loaded successfully")
                else:
                    print(f"{model_name} model file not found: {filename}")
            
            if not self.models:
                messagebox.showerror("Error", "No trained models found. Please run malaria_symptom_classification.py first.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="Malaria Symptom-Based Detection System", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Create main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left frame for input
        left_frame = tk.Frame(main_container, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Input section title
        input_label = tk.Label(
            left_frame,
            text="Patient Information & Symptoms",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_label.pack(pady=(0, 15))
        
        # Create input frame with scrollbar
        input_canvas = tk.Canvas(left_frame, bg='white', relief='sunken')
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=input_canvas.yview)
        scrollable_frame = tk.Frame(input_canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: input_canvas.configure(scrollregion=input_canvas.bbox("all"))
        )
        
        input_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        input_canvas.configure(yscrollcommand=scrollbar.set)
        
        input_canvas.pack(side="left", fill="both", expand=True, pady=(0, 10))
        scrollbar.pack(side="right", fill="y", pady=(0, 10))
        
        # Input fields
        self.input_vars = {}
        self.setup_input_fields(scrollable_frame)
        
        # Buttons frame
        buttons_frame = tk.Frame(left_frame, bg='#f0f0f0')
        buttons_frame.pack(fill='x', pady=10)
        
        # Load sample button
        self.load_sample_button = tk.Button(
            buttons_frame,
            text="Load Sample Case",
            command=self.load_sample_data,
            bg='#3498db',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        self.load_sample_button.pack(side='left', padx=(0, 10))
        
        # Clear button
        self.clear_button = tk.Button(
            buttons_frame,
            text="Clear All",
            command=self.clear_data,
            bg='#95a5a6',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        self.clear_button.pack(side='left', padx=(0, 10))
        
        # Model selection
        model_label = tk.Label(
            buttons_frame,
            text="Model:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        )
        model_label.pack(side='left', padx=(20, 5))
        
        self.model_var = tk.StringVar(value="Random Forest")
        self.model_combo = ttk.Combobox(
            buttons_frame,
            textvariable=self.model_var,
            values=list(self.models.keys()),
            state="readonly",
            width=15
        )
        self.model_combo.pack(side='left')
        
        # Right frame for results
        right_frame = tk.Frame(main_container, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))
        
        # Results section
        results_label = tk.Label(
            right_frame,
            text="Analysis Results",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_label.pack(pady=(0, 10))
        
        # Results display
        self.results_text = tk.Text(
            right_frame,
            height=20,
            width=40,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.results_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Prediction button
        self.predict_button = tk.Button(
            right_frame,
            text="Predict Malaria Risk",
            command=self.predict_malaria,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 14, "bold"),
            relief='flat',
            padx=30,
            pady=15
        )
        self.predict_button.pack(fill='x', pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Enter patient information and symptoms")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief='sunken',
            anchor='w',
            bg='#ecf0f1',
            fg='#2c3e50',
            font=("Arial", 10)
        )
        status_bar.pack(side='bottom', fill='x')
    
    def setup_input_fields(self, parent):
        """Setup input fields for all features"""
        # Feature descriptions for better user understanding
        feature_descriptions = {
            'age': 'Patient Age (years)',
            'sex': 'Gender',
            'fever': 'Fever',
            'cold': 'Cold/Cough',
            'rigor': 'Rigor/Chills',
            'fatigue': 'Fatigue',
            'headace': 'Headache',
            'bitter_tongue': 'Bitter Taste in Mouth',
            'vomitting': 'Vomiting',
            'diarrhea': 'Diarrhea',
            'Convulsion': 'Convulsions/Seizures',
            'Anemia': 'Anemia',
            'jundice': 'Jaundice',
            'cocacola_urine': 'Coca-Cola Colored Urine',
            'hypoglycemia': 'Low Blood Sugar',
            'prostraction': 'Prostration (Weakness)',
            'hyperpyrexia': 'Very High Fever'
        }
        
        row = 0
        for feature in self.feature_names:
            # Feature frame
            feature_frame = tk.Frame(parent, bg='white')
            feature_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
            parent.grid_columnconfigure(0, weight=1)
            
            # Feature label
            description = feature_descriptions.get(feature, feature.replace('_', ' ').title())
            label = tk.Label(
                feature_frame,
                text=f"{description}:",
                width=25,
                anchor='w',
                bg='white',
                font=("Arial", 10)
            )
            label.grid(row=0, column=0, padx=(0, 10))
            
            # Input widget based on feature type
            if feature == 'age':
                # Age input (numerical)
                var = tk.StringVar(value="25")
                entry = tk.Entry(
                    feature_frame,
                    textvariable=var,
                    width=15,
                    font=("Arial", 10)
                )
                entry.grid(row=0, column=1)
                self.input_vars[feature] = var
                
            elif feature == 'sex':
                # Gender selection
                var = tk.StringVar(value="0")
                gender_frame = tk.Frame(feature_frame, bg='white')
                gender_frame.grid(row=0, column=1)
                
                rb1 = tk.Radiobutton(
                    gender_frame,
                    text="Female",
                    variable=var,
                    value="0",
                    bg='white'
                )
                rb1.pack(side='left', padx=(0, 10))
                
                rb2 = tk.Radiobutton(
                    gender_frame,
                    text="Male",
                    variable=var,
                    value="1",
                    bg='white'
                )
                rb2.pack(side='left')
                self.input_vars[feature] = var
                
            else:
                # Binary symptom inputs (Yes/No)
                var = tk.StringVar(value="0")
                symptom_frame = tk.Frame(feature_frame, bg='white')
                symptom_frame.grid(row=0, column=1)
                
                rb_no = tk.Radiobutton(
                    symptom_frame,
                    text="No",
                    variable=var,
                    value="0",
                    bg='white'
                )
                rb_no.pack(side='left', padx=(0, 10))
                
                rb_yes = tk.Radiobutton(
                    symptom_frame,
                    text="Yes",
                    variable=var,
                    value="1",
                    bg='white'
                )
                rb_yes.pack(side='left')
                self.input_vars[feature] = var
            
            row += 1
    
    def load_sample_data(self):
        """Load sample patient data for testing"""
        # Sample data representing a patient with severe malaria symptoms
        sample_data = {
            'age': '25',
            'sex': '1',  # Male
            'fever': '1',  # Yes
            'cold': '1',   # Yes
            'rigor': '1',  # Yes
            'fatigue': '1', # Yes
            'headace': '1', # Yes
            'bitter_tongue': '1', # Yes
            'vomitting': '1', # Yes
            'diarrhea': '1', # Yes
            'Convulsion': '0', # No
            'Anemia': '1', # Yes
            'jundice': '1', # Yes
            'cocacola_urine': '1', # Yes
            'hypoglycemia': '1', # Yes
            'prostraction': '1', # Yes
            'hyperpyrexia': '1'  # Yes
        }
        
        # Set sample data in input fields
        for feature, value in sample_data.items():
            if feature in self.input_vars:
                self.input_vars[feature].set(value)
        
        self.status_var.set("Sample data loaded - Patient with multiple severe symptoms")
        
        # Show sample info
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Sample Patient Data Loaded\n")
        self.results_text.insert(tk.END, "=" * 30 + "\n\n")
        self.results_text.insert(tk.END, "This is a sample patient with multiple malaria symptoms:\n\n")
        self.results_text.insert(tk.END, "• 25-year-old male\n")
        self.results_text.insert(tk.END, "• Fever, chills, fatigue\n")
        self.results_text.insert(tk.END, "• Headache, vomiting, diarrhea\n")
        self.results_text.insert(tk.END, "• Jaundice, anemia\n")
        self.results_text.insert(tk.END, "• Coca-cola colored urine\n")
        self.results_text.insert(tk.END, "• Low blood sugar\n")
        self.results_text.insert(tk.END, "• Prostration, high fever\n\n")
        self.results_text.insert(tk.END, "Click 'Predict Malaria Risk' to analyze.\n")
        self.results_text.insert(tk.END, "Expected: High risk of severe malaria")
    
    def clear_data(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            if var.get() != "0":  # Don't reset age
                var.set("0")
        
        # Reset age to default
        if 'age' in self.input_vars:
            self.input_vars['age'].set("25")
        
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Data cleared - Enter new patient information")
    
    def predict_malaria(self):
        """Predict malaria risk using selected model"""
        if not self.models:
            messagebox.showerror("Error", "No trained models available")
            return
        
        selected_model = self.model_var.get()
        if selected_model not in self.models:
            messagebox.showerror("Error", f"Model '{selected_model}' not found")
            return
        
        try:
            self.status_var.set("Analyzing patient symptoms...")
            self.root.update()
            
            # Collect input data
            features = []
            for feature in self.feature_names:
                try:
                    value = float(self.input_vars[feature].get())
                    features.append(value)
                except ValueError:
                    messagebox.showerror("Error", f"Please enter valid data for all fields")
                    return
            
            # Prepare data for prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Get model and make prediction
            model = self.models[selected_model]
            
            # Scale data if needed
            if selected_model in ['SVM', 'Logistic Regression'] and self.scaler:
                features_scaled = self.scaler.transform(features_array)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0]
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "MALARIA RISK ASSESSMENT\n")
            self.results_text.insert(tk.END, "=" * 30 + "\n\n")
            self.results_text.insert(tk.END, f"Model Used: {selected_model}\n\n")
            
            if prediction == 1:
                self.results_text.insert(tk.END, "RISK LEVEL: HIGH\n", "high_risk")
                self.results_text.insert(tk.END, "🚨 SEVERE MALARIA RISK DETECTED\n\n")
                self.results_text.insert(tk.END, "The patient shows symptoms indicating a high risk of severe malaria.\n")
                self.results_text.insert(tk.END, "Immediate medical attention is recommended.\n\n")
            else:
                self.results_text.insert(tk.END, "RISK LEVEL: LOW\n", "low_risk")
                self.results_text.insert(tk.END, "✅ NO SEVERE MALARIA RISK\n\n")
                self.results_text.insert(tk.END, "The patient shows minimal symptoms of severe malaria.\n")
                self.results_text.insert(tk.END, "Continue monitoring for any symptom changes.\n\n")
            
            # Show confidence scores
            self.results_text.insert(tk.END, "CONFIDENCE SCORES:\n")
            self.results_text.insert(tk.END, f"No Severe Malaria: {probability[0]:.2%}\n")
            self.results_text.insert(tk.END, f"Severe Malaria Risk: {probability[1]:.2%}\n\n")
            
            # Show symptom summary
            self.results_text.insert(tk.END, "PATIENT SYMPTOM SUMMARY:\n")
            self.results_text.insert(tk.END, "-" * 25 + "\n")
            
            symptom_count = 0
            for i, feature in enumerate(self.feature_names):
                if feature not in ['age', 'sex']:
                    if features[i] == 1:
                        symptom_count += 1
                        self.results_text.insert(tk.END, f"✓ {feature.replace('_', ' ').title()}\n")
            
            self.results_text.insert(tk.END, f"\nTotal Symptoms Present: {symptom_count}\n")
            
            # Configure text colors
            self.results_text.tag_configure("high_risk", foreground="red", font=("Arial", 12, "bold"))
            self.results_text.tag_configure("low_risk", foreground="green", font=("Arial", 12, "bold"))
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")

def main():
    """Main function to run the symptom-based GUI"""
    root = tk.Tk()
    app = MalariaSymptomGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
