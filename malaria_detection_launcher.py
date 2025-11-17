#!/usr/bin/env python3
"""
Malaria Detection System - Main Launcher
Combined Image-Based and Symptom-Based Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import joblib
import cv2
import os
import sys
from pathlib import Path

class MalariaDetectionLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Malaria Detection System - Complete Suite")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Set working directory to deployment folder
        self.deployment_path = Path(__file__).parent / "deployment"
        os.chdir(self.deployment_path)
        
        # Load models
        self.image_model = None
        self.symptom_models = {}
        self.symptom_scaler = None
        self.symptom_features = []
        self.load_all_models()
        
        # Current predictions
        self.image_prediction = None
        self.symptom_prediction = None
        
        self.setup_ui()
        
    def load_all_models(self):
        """Load all trained models"""
        try:
            models_path = self.deployment_path / "models"
            
            # Load image-based model
            image_model_path = models_path / "rf_malaria_100_5"
            if image_model_path.exists():
                self.image_model = joblib.load(image_model_path)
                print("✅ Image-based model loaded successfully")
            else:
                print("⚠️ Image-based model not found")
            
            # Load symptom-based models
            symptom_features_path = models_path / "malaria_symptom_features.joblib"
            if symptom_features_path.exists():
                self.symptom_features = joblib.load(symptom_features_path)
                print("✅ Symptom features loaded successfully")
            
            symptom_scaler_path = models_path / "malaria_symptom_scaler.joblib"
            if symptom_scaler_path.exists():
                self.symptom_scaler = joblib.load(symptom_scaler_path)
                print("✅ Symptom scaler loaded successfully")
            
            # Load symptom models
            symptom_model_files = {
                'Decision Tree': 'malaria_symptom_decision_tree.joblib',
                'SVM': 'malaria_symptom_svm.joblib',
                'Logistic Regression': 'malaria_symptom_logistic_regression.joblib',
                'Random Forest': 'malaria_symptom_random_forest.joblib'
            }
            
            for model_name, filename in symptom_model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    self.symptom_models[model_name] = joblib.load(model_path)
                    print(f"✅ Symptom {model_name} model loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="🦠 Malaria Detection System", 
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Complete AI-Powered Malaria Detection Suite",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Create main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left side - Detection Methods
        left_frame = tk.Frame(main_container, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Detection methods title
        methods_label = tk.Label(
            left_frame,
            text="Detection Methods",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        methods_label.pack(pady=(0, 20))
        
        # Image-based detection button
        self.image_button = tk.Button(
            left_frame,
            text="🖼️ Image-Based Detection\n(Analyze Blood Cell Images)",
            command=self.open_image_detection,
            bg='#3498db',
            fg='white',
            font=("Arial", 14, "bold"),
            relief='flat',
            padx=30,
            pady=20,
            width=25
        )
        self.image_button.pack(pady=10)
        
        # Symptom-based detection button
        self.symptom_button = tk.Button(
            left_frame,
            text="🏥 Symptom-Based Detection\n(Analyze Clinical Symptoms)",
            command=self.open_symptom_detection,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 14, "bold"),
            relief='flat',
            padx=30,
            pady=20,
            width=25
        )
        self.symptom_button.pack(pady=10)
        
        # Integrated analysis button
        self.integrated_button = tk.Button(
            left_frame,
            text="🔄 Integrated Analysis\n(Combined Approach)",
            command=self.run_integrated_analysis,
            bg='#9b59b6',
            fg='white',
            font=("Arial", 14, "bold"),
            relief='flat',
            padx=30,
            pady=20,
            width=25
        )
        self.integrated_button.pack(pady=10)
        
        # Right side - System Status and Results
        right_frame = tk.Frame(main_container, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))
        
        # Status section
        status_label = tk.Label(
            right_frame,
            text="System Status",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        status_label.pack(pady=(0, 10))
        
        # Status display
        self.status_text = tk.Text(
            right_frame,
            height=15,
            width=40,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.status_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Quick analysis section
        quick_label = tk.Label(
            right_frame,
            text="Quick Analysis",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        quick_label.pack(pady=(10, 5))
        
        # Quick analysis buttons
        quick_frame = tk.Frame(right_frame, bg='#f0f0f0')
        quick_frame.pack(fill='x', pady=5)
        
        self.quick_image_button = tk.Button(
            quick_frame,
            text="Quick Image Test",
            command=self.quick_image_analysis,
            bg='#27ae60',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=15,
            pady=8
        )
        self.quick_image_button.pack(side='left', padx=(0, 5))
        
        self.quick_symptom_button = tk.Button(
            quick_frame,
            text="Quick Symptom Test",
            command=self.quick_symptom_analysis,
            bg='#f39c12',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat',
            padx=15,
            pady=8
        )
        self.quick_symptom_button.pack(side='left')
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select detection method")
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
        
        # Initialize status display
        self.update_system_status()
    
    def update_system_status(self):
        """Update system status display"""
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "SYSTEM STATUS REPORT\n")
        self.status_text.insert(tk.END, "=" * 25 + "\n\n")
        
        # Model status
        self.status_text.insert(tk.END, "📊 MODEL STATUS:\n")
        if self.image_model:
            self.status_text.insert(tk.END, "✅ Image-based model: LOADED\n")
        else:
            self.status_text.insert(tk.END, "❌ Image-based model: NOT FOUND\n")
        
        if self.symptom_models:
            self.status_text.insert(tk.END, f"✅ Symptom models: {len(self.symptom_models)} LOADED\n")
        else:
            self.status_text.insert(tk.END, "❌ Symptom models: NOT FOUND\n")
        
        # Features status
        self.status_text.insert(tk.END, f"\n📋 FEATURES:\n")
        if self.symptom_features:
            self.status_text.insert(tk.END, f"✅ Symptom features: {len(self.symptom_features)} loaded\n")
        else:
            self.status_text.insert(tk.END, "❌ Symptom features: NOT FOUND\n")
        
        # Data status
        data_path = self.deployment_path / "data"
        self.status_text.insert(tk.END, f"\n📁 DATA FILES:\n")
        if (data_path / "mmc1.csv").exists():
            self.status_text.insert(tk.END, "✅ Symptom dataset: AVAILABLE\n")
        else:
            self.status_text.insert(tk.END, "❌ Symptom dataset: NOT FOUND\n")
        
        # System ready status
        self.status_text.insert(tk.END, f"\n🚀 SYSTEM STATUS:\n")
        if self.image_model and self.symptom_models and self.symptom_features:
            self.status_text.insert(tk.END, "✅ FULLY OPERATIONAL\n")
            self.status_text.insert(tk.END, "Ready for malaria detection!\n")
        else:
            self.status_text.insert(tk.END, "⚠️ PARTIALLY OPERATIONAL\n")
            self.status_text.insert(tk.END, "Some features may not be available.\n")
    
    def open_image_detection(self):
        """Open image-based detection window"""
        if not self.image_model:
            messagebox.showerror("Error", "Image-based model not available")
            return
        
        # Create image detection window
        image_window = tk.Toplevel(self.root)
        image_window.title("Image-Based Malaria Detection")
        image_window.geometry("800x600")
        image_window.configure(bg='#f0f0f0')
        
        self.setup_image_detection_ui(image_window)
    
    def setup_image_detection_ui(self, window):
        """Setup image detection UI"""
        # Title
        title_label = tk.Label(
            window,
            text="🖼️ Image-Based Malaria Detection",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(window, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left side - Input
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Upload button
        upload_button = tk.Button(
            left_frame,
            text="Upload Blood Cell Image",
            command=lambda: self.upload_and_analyze_image(window),
            bg='#3498db',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        upload_button.pack(pady=20)
        
        # Contour input section
        contour_label = tk.Label(
            left_frame,
            text="Or Enter Contour Areas:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        )
        contour_label.pack(pady=(20, 10))
        
        # Contour input fields
        self.contour_vars = []
        contour_names = ["Contour Area 1", "Contour Area 2", "Contour Area 3", "Contour Area 4", "Contour Area 5"]
        
        for i, name in enumerate(contour_names):
            frame = tk.Frame(left_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=5)
            
            label = tk.Label(frame, text=f"{name}:", width=15, anchor='w', bg='#f0f0f0')
            label.pack(side='left')
            
            var = tk.StringVar(value="0")
            self.contour_vars.append(var)
            
            entry = tk.Entry(frame, textvariable=var, width=15, font=("Arial", 11))
            entry.pack(side='right')
        
        # Right side - Results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))
        
        # Results display
        results_label = tk.Label(
            right_frame,
            text="Analysis Results",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0'
        )
        results_label.pack(pady=(0, 10))
        
        self.image_results_text = tk.Text(
            right_frame,
            height=15,
            width=30,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.image_results_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Predict button
        predict_button = tk.Button(
            right_frame,
            text="Analyze Blood Cell",
            command=lambda: self.analyze_image(window),
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        predict_button.pack(fill='x')
    
    def upload_and_analyze_image(self, window):
        """Upload and automatically analyze image"""
        file_path = filedialog.askopenfilename(
            title="Select Blood Cell Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Process image to extract contours
                contours = self.process_image(file_path)
                if contours:
                    # Update contour input fields
                    for i, area in enumerate(contours):
                        if i < len(self.contour_vars):
                            self.contour_vars[i].set(str(area))
                    
                    # Automatically analyze
                    self.analyze_image(window)
                else:
                    messagebox.showerror("Error", "Could not extract contours from image")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def process_image(self, image_path):
        """Process blood cell image to extract contour areas"""
        try:
            # Load image
            im = cv2.imread(image_path)
            if im is None:
                return None
            
            # Apply preprocessing
            im = cv2.GaussianBlur(im, (5, 5), 2)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            
            # Extract contour areas
            contour_areas = []
            for i in range(5):
                try:
                    area = cv2.contourArea(contours[i])
                    contour_areas.append(area)
                except:
                    contour_areas.append(0)
            
            return contour_areas
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def analyze_image(self, window):
        """Analyze blood cell image"""
        try:
            # Get contour areas
            features = []
            for var in self.contour_vars:
                try:
                    value = float(var.get())
                    features.append(value)
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid contour areas")
                    return
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            prediction = self.image_model.predict(features_array)[0]
            probability = self.image_model.predict_proba(features_array)[0]
            
            # Store prediction for integrated analysis
            self.image_prediction = prediction
            
            # Display results
            self.image_results_text.delete(1.0, tk.END)
            self.image_results_text.insert(tk.END, "BLOOD CELL ANALYSIS\n")
            self.image_results_text.insert(tk.END, "=" * 20 + "\n\n")
            
            if prediction == "Parasitized":
                self.image_results_text.insert(tk.END, "RESULT: MALARIA DETECTED\n", "warning")
                self.image_results_text.insert(tk.END, "🦠 Blood cell appears infected\n\n")
            else:
                self.image_results_text.insert(tk.END, "RESULT: NO MALARIA\n", "success")
                self.image_results_text.insert(tk.END, "✅ Blood cell appears healthy\n\n")
            
            self.image_results_text.insert(tk.END, "CONFIDENCE SCORES:\n")
            self.image_results_text.insert(tk.END, f"Parasitized: {probability[0]:.2%}\n")
            self.image_results_text.insert(tk.END, f"Uninfected: {probability[1]:.2%}\n")
            
            self.image_results_text.tag_configure("warning", foreground="red", font=("Arial", 11, "bold"))
            self.image_results_text.tag_configure("success", foreground="green", font=("Arial", 11, "bold"))
            
            self.status_var.set("Image analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Image analysis failed: {str(e)}")
    
    def open_symptom_detection(self):
        """Open symptom-based detection window"""
        if not self.symptom_models:
            messagebox.showerror("Error", "Symptom-based models not available")
            return
        
        # Create symptom detection window
        symptom_window = tk.Toplevel(self.root)
        symptom_window.title("Symptom-Based Malaria Detection")
        symptom_window.geometry("900x700")
        symptom_window.configure(bg='#f0f0f0')
        
        self.setup_symptom_detection_ui(symptom_window)
    
    def setup_symptom_detection_ui(self, window):
        """Setup symptom detection UI"""
        # Title
        title_label = tk.Label(
            window,
            text="🏥 Symptom-Based Malaria Detection",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Create scrollable frame
        canvas = tk.Canvas(window, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main frame
        main_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left side - Input
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Input section
        input_label = tk.Label(
            left_frame,
            text="Patient Information & Symptoms",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_label.pack(pady=(0, 15))
        
        # Setup symptom input fields
        self.symptom_vars = {}
        self.setup_symptom_inputs(left_frame)
        
        # Right side - Results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))
        
        # Results section
        results_label = tk.Label(
            right_frame,
            text="Symptom Analysis Results",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_label.pack(pady=(0, 10))
        
        # Model selection
        model_label = tk.Label(
            right_frame,
            text="Select Model:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        )
        model_label.pack(pady=(0, 5))
        
        self.symptom_model_var = tk.StringVar(value="Random Forest")
        self.symptom_model_combo = ttk.Combobox(
            right_frame,
            textvariable=self.symptom_model_var,
            values=list(self.symptom_models.keys()),
            state="readonly",
            width=20
        )
        self.symptom_model_combo.pack(pady=(0, 10))
        
        self.symptom_results_text = tk.Text(
            right_frame,
            height=15,
            width=35,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.symptom_results_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Predict button
        predict_button = tk.Button(
            right_frame,
            text="Analyze Symptoms",
            command=lambda: self.analyze_symptoms(window),
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        predict_button.pack(fill='x')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_symptom_inputs(self, parent):
        """Setup symptom input fields"""
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
        for feature in self.symptom_features:
            feature_frame = tk.Frame(parent, bg='#f0f0f0')
            feature_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
            parent.grid_columnconfigure(0, weight=1)
            
            description = feature_descriptions.get(feature, feature.replace('_', ' ').title())
            label = tk.Label(
                feature_frame,
                text=f"{description}:",
                width=25,
                anchor='w',
                bg='#f0f0f0',
                font=("Arial", 10)
            )
            label.grid(row=0, column=0, padx=(0, 10))
            
            if feature == 'age':
                var = tk.StringVar(value="25")
                entry = tk.Entry(feature_frame, textvariable=var, width=15, font=("Arial", 10))
                entry.grid(row=0, column=1)
                self.symptom_vars[feature] = var
            elif feature == 'sex':
                var = tk.StringVar(value="0")
                gender_frame = tk.Frame(feature_frame, bg='#f0f0f0')
                gender_frame.grid(row=0, column=1)
                
                rb1 = tk.Radiobutton(gender_frame, text="Female", variable=var, value="0", bg='#f0f0f0')
                rb1.pack(side='left', padx=(0, 10))
                rb2 = tk.Radiobutton(gender_frame, text="Male", variable=var, value="1", bg='#f0f0f0')
                rb2.pack(side='left')
                self.symptom_vars[feature] = var
            else:
                var = tk.StringVar(value="0")
                symptom_frame = tk.Frame(feature_frame, bg='#f0f0f0')
                symptom_frame.grid(row=0, column=1)
                
                rb_no = tk.Radiobutton(symptom_frame, text="No", variable=var, value="0", bg='#f0f0f0')
                rb_no.pack(side='left', padx=(0, 10))
                rb_yes = tk.Radiobutton(symptom_frame, text="Yes", variable=var, value="1", bg='#f0f0f0')
                rb_yes.pack(side='left')
                self.symptom_vars[feature] = var
            
            row += 1
    
    def analyze_symptoms(self, window):
        """Analyze patient symptoms"""
        try:
            # Collect symptom data
            features = []
            for feature in self.symptom_features:
                try:
                    value = float(self.symptom_vars[feature].get())
                    features.append(value)
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid data for all fields")
                    return
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            selected_model = self.symptom_model_var.get()
            model = self.symptom_models[selected_model]
            
            if selected_model in ['SVM', 'Logistic Regression'] and self.symptom_scaler:
                features_scaled = self.symptom_scaler.transform(features_array)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0]
            
            # Store prediction for integrated analysis
            self.symptom_prediction = prediction
            
            # Display results
            self.symptom_results_text.delete(1.0, tk.END)
            self.symptom_results_text.insert(tk.END, "SYMPTOM ANALYSIS\n")
            self.symptom_results_text.insert(tk.END, "=" * 18 + "\n\n")
            self.symptom_results_text.insert(tk.END, f"Model: {selected_model}\n\n")
            
            if prediction == 1:
                self.symptom_results_text.insert(tk.END, "RISK: SEVERE MALARIA\n", "high_risk")
                self.symptom_results_text.insert(tk.END, "🚨 High risk detected\n\n")
            else:
                self.symptom_results_text.insert(tk.END, "RISK: LOW MALARIA\n", "low_risk")
                self.symptom_results_text.insert(tk.END, "✅ Low risk detected\n\n")
            
            self.symptom_results_text.insert(tk.END, "CONFIDENCE SCORES:\n")
            self.symptom_results_text.insert(tk.END, f"No Severe Malaria: {probability[0]:.2%}\n")
            self.symptom_results_text.insert(tk.END, f"Severe Malaria Risk: {probability[1]:.2%}\n")
            
            self.symptom_results_text.tag_configure("high_risk", foreground="red", font=("Arial", 11, "bold"))
            self.symptom_results_text.tag_configure("low_risk", foreground="green", font=("Arial", 11, "bold"))
            
            self.status_var.set("Symptom analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Symptom analysis failed: {str(e)}")
    
    def run_integrated_analysis(self):
        """Run integrated analysis combining both approaches"""
        if not self.image_prediction and not self.symptom_prediction:
            messagebox.showinfo("Info", "Please perform both image and symptom analyses first to get integrated results.")
            return
        
        # Create integrated analysis window
        integrated_window = tk.Toplevel(self.root)
        integrated_window.title("Integrated Malaria Analysis")
        integrated_window.geometry("800x600")
        integrated_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            integrated_window,
            text="🔄 Integrated Malaria Analysis",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Results display
        results_text = tk.Text(
            integrated_window,
            height=25,
            width=80,
            font=("Arial", 12),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        results_text.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Generate integrated analysis
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, "INTEGRATED MALARIA DETECTION ANALYSIS\n")
        results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        results_text.insert(tk.END, "🔍 INTEGRATED ANALYSIS SUMMARY\n")
        results_text.insert(tk.END, "-" * 30 + "\n\n")
        
        # Image analysis results
        results_text.insert(tk.END, "📊 IMAGE ANALYSIS:\n")
        if self.image_prediction:
            if self.image_prediction == "Parasitized":
                results_text.insert(tk.END, "   • Blood cell shows malaria parasites\n")
                image_risk = "HIGH"
            else:
                results_text.insert(tk.END, "   • Blood cell appears healthy\n")
                image_risk = "LOW"
        else:
            results_text.insert(tk.END, "   • No image analysis performed\n")
            image_risk = "UNKNOWN"
        
        # Symptom analysis results
        results_text.insert(tk.END, "\n🏥 SYMPTOM ANALYSIS:\n")
        if self.symptom_prediction is not None:
            if self.symptom_prediction == 1:
                results_text.insert(tk.END, "   • Patient shows severe malaria symptoms\n")
                symptom_risk = "HIGH"
            else:
                results_text.insert(tk.END, "   • Patient shows minimal malaria symptoms\n")
                symptom_risk = "LOW"
        else:
            results_text.insert(tk.END, "   • No symptom analysis performed\n")
            symptom_risk = "UNKNOWN"
        
        # Integrated conclusion
        results_text.insert(tk.END, "\n🎯 INTEGRATED CONCLUSION:\n")
        results_text.insert(tk.END, "-" * 25 + "\n\n")
        
        if image_risk == "HIGH" and symptom_risk == "HIGH":
            results_text.insert(tk.END, "🚨 CRITICAL: MALARIA CONFIRMED\n\n", "critical")
            results_text.insert(tk.END, "Both image and symptom analyses indicate malaria.\n")
            results_text.insert(tk.END, "Immediate medical treatment required.\n\n")
            recommendation = "URGENT MEDICAL ATTENTION"
            
        elif image_risk == "HIGH" or symptom_risk == "HIGH":
            results_text.insert(tk.END, "⚠️ SUSPECTED: MALARIA LIKELY\n\n", "warning")
            results_text.insert(tk.END, "One analysis suggests malaria presence.\n")
            results_text.insert(tk.END, "Further medical evaluation recommended.\n\n")
            recommendation = "MEDICAL EVALUATION"
            
        else:
            results_text.insert(tk.END, "✅ CLEAR: NO MALARIA DETECTED\n\n", "success")
            results_text.insert(tk.END, "Both analyses suggest no malaria.\n")
            results_text.insert(tk.END, "Continue monitoring for symptom changes.\n\n")
            recommendation = "ROUTINE MONITORING"
        
        # Recommendations
        results_text.insert(tk.END, "📋 RECOMMENDATIONS:\n")
        results_text.insert(tk.END, f"• {recommendation}\n")
        results_text.insert(tk.END, "• Follow up with healthcare provider\n")
        results_text.insert(tk.END, "• Monitor for symptom progression\n")
        results_text.insert(tk.END, "• Consider additional testing if needed\n\n")
        
        # Configure text colors
        results_text.tag_configure("critical", foreground="red", font=("Arial", 12, "bold"))
        results_text.tag_configure("warning", foreground="orange", font=("Arial", 12, "bold"))
        results_text.tag_configure("success", foreground="green", font=("Arial", 12, "bold"))
        
        self.status_var.set("Integrated analysis complete")
    
    def quick_image_analysis(self):
        """Quick image analysis with sample data"""
        if not self.image_model:
            messagebox.showerror("Error", "Image model not available")
            return
        
        # Use sample contour data
        sample_contours = [113, 15160, 0, 0, 0]  # Sample parasitized cell
        
        try:
            features_array = np.array(sample_contours).reshape(1, -1)
            prediction = self.image_model.predict(features_array)[0]
            probability = self.image_model.predict_proba(features_array)[0]
            
            result = "MALARIA DETECTED" if prediction == "Parasitized" else "NO MALARIA"
            confidence = max(probability) * 100
            
            messagebox.showinfo("Quick Image Test", 
                              f"Result: {result}\n"
                              f"Confidence: {confidence:.1f}%\n\n"
                              f"This is a sample test with parasitized cell data.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Quick test failed: {str(e)}")
    
    def quick_symptom_analysis(self):
        """Quick symptom analysis with sample data"""
        if not self.symptom_models:
            messagebox.showerror("Error", "Symptom models not available")
            return
        
        # Use sample symptom data (severe malaria case)
        sample_features = [25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        
        try:
            features_array = np.array(sample_features).reshape(1, -1)
            model = self.symptom_models['Random Forest']
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
            result = "SEVERE MALARIA RISK" if prediction == 1 else "LOW MALARIA RISK"
            confidence = max(probability) * 100
            
            messagebox.showinfo("Quick Symptom Test", 
                              f"Result: {result}\n"
                              f"Confidence: {confidence:.1f}%\n\n"
                              f"This is a sample test with severe malaria symptoms.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Quick test failed: {str(e)}")

def main():
    """Main function to run the malaria detection launcher"""
    root = tk.Tk()
    app = MalariaDetectionLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()
