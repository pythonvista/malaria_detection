import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import joblib
import cv2
import os

class IntegratedMalariaDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Malaria Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load models
        self.image_model = None
        self.symptom_models = {}
        self.symptom_scaler = None
        self.symptom_features = []
        self.load_all_models()
        
        self.setup_ui()
        
    def load_all_models(self):
        """Load both image-based and symptom-based models"""
        try:
            # Load image-based model
            if os.path.exists("rf_malaria_100_5"):
                self.image_model = joblib.load("rf_malaria_100_5")
                print("Image-based model loaded successfully")
            
            # Load symptom-based models
            if os.path.exists("malaria_symptom_features.joblib"):
                self.symptom_features = joblib.load("malaria_symptom_features.joblib")
                print("Symptom features loaded successfully")
            
            if os.path.exists("malaria_symptom_scaler.joblib"):
                self.symptom_scaler = joblib.load("malaria_symptom_scaler.joblib")
                print("Symptom scaler loaded successfully")
            
            # Load symptom models
            symptom_model_files = {
                'Decision Tree': 'malaria_symptom_decision_tree.joblib',
                'SVM': 'malaria_symptom_svm.joblib',
                'Logistic Regression': 'malaria_symptom_logistic_regression.joblib',
                'Random Forest': 'malaria_symptom_random_forest.joblib'
            }
            
            for model_name, filename in symptom_model_files.items():
                if os.path.exists(filename):
                    self.symptom_models[model_name] = joblib.load(filename)
                    print(f"Symptom {model_name} model loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def setup_ui(self):
        """Setup the integrated user interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="Integrated Malaria Detection System", 
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Combined Image-Based and Symptom-Based Analysis",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Image-based detection tab
        self.setup_image_tab(notebook)
        
        # Symptom-based detection tab
        self.setup_symptom_tab(notebook)
        
        # Integrated analysis tab
        self.setup_integrated_tab(notebook)
        
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
    
    def setup_image_tab(self, notebook):
        """Setup image-based detection tab"""
        image_frame = ttk.Frame(notebook)
        notebook.add(image_frame, text="🖼️ Image-Based Detection")
        
        # Left side - Image input
        left_frame = tk.Frame(image_frame, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=10, pady=10)
        
        # Image input section
        input_label = tk.Label(
            left_frame,
            text="Blood Cell Image Analysis",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_label.pack(pady=(0, 15))
        
        # Image upload button
        self.upload_button = tk.Button(
            left_frame,
            text="Upload Blood Cell Image",
            command=self.upload_image,
            bg='#3498db',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        self.upload_button.pack(pady=10)
        
        # Contour input section
        contour_label = tk.Label(
            left_frame,
            text="Or Enter Contour Areas Manually:",
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
        right_frame = tk.Frame(image_frame, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=10, pady=10)
        
        # Results display
        results_label = tk.Label(
            right_frame,
            text="Image Analysis Results",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_label.pack(pady=(0, 10))
        
        self.image_results_text = tk.Text(
            right_frame,
            height=20,
            width=40,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.image_results_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Predict button
        self.image_predict_button = tk.Button(
            right_frame,
            text="Analyze Blood Cell Image",
            command=self.predict_image,
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        self.image_predict_button.pack(fill='x')
    
    def setup_symptom_tab(self, notebook):
        """Setup symptom-based detection tab"""
        symptom_frame = ttk.Frame(notebook)
        notebook.add(symptom_frame, text="🏥 Symptom-Based Detection")
        
        # Create scrollable frame for symptoms
        canvas = tk.Canvas(symptom_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(symptom_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Left side - Symptom input
        left_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=10, pady=10)
        
        # Symptom input section
        input_label = tk.Label(
            left_frame,
            text="Patient Symptoms & Information",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_label.pack(pady=(0, 15))
        
        # Setup symptom input fields
        self.symptom_vars = {}
        self.setup_symptom_inputs(left_frame)
        
        # Right side - Results
        right_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        right_frame.pack(side='right', expand=True, fill='both', padx=10, pady=10)
        
        # Results display
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
            width=40,
            font=("Arial", 11),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.symptom_results_text.pack(expand=True, fill='both', pady=(0, 10))
        
        # Predict button
        self.symptom_predict_button = tk.Button(
            right_frame,
            text="Analyze Symptoms",
            command=self.predict_symptoms,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        self.symptom_predict_button.pack(fill='x')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_integrated_tab(self, notebook):
        """Setup integrated analysis tab"""
        integrated_frame = ttk.Frame(notebook)
        notebook.add(integrated_frame, text="🔄 Integrated Analysis")
        
        # Main content
        main_frame = tk.Frame(integrated_frame, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Combined Analysis Results",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Analysis results display
        self.integrated_results_text = tk.Text(
            main_frame,
            height=20,
            width=80,
            font=("Arial", 12),
            bg='white',
            relief='sunken',
            wrap='word'
        )
        self.integrated_results_text.pack(expand=True, fill='both', pady=(0, 20))
        
        # Integrated analysis button
        self.integrated_analyze_button = tk.Button(
            main_frame,
            text="🔍 Run Integrated Analysis",
            command=self.run_integrated_analysis,
            bg='#9b59b6',
            fg='white',
            font=("Arial", 14, "bold"),
            relief='flat',
            padx=30,
            pady=15
        )
        self.integrated_analyze_button.pack()
    
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
    
    def upload_image(self):
        """Upload and process blood cell image"""
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
                    
                    self.status_var.set(f"Image processed: {os.path.basename(file_path)}")
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
    
    def predict_image(self):
        """Predict malaria from image/contour data"""
        if not self.image_model:
            messagebox.showerror("Error", "Image model not loaded")
            return
        
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
            
            # Display results
            self.image_results_text.delete(1.0, tk.END)
            self.image_results_text.insert(tk.END, "BLOOD CELL IMAGE ANALYSIS\n")
            self.image_results_text.insert(tk.END, "=" * 30 + "\n\n")
            
            if prediction == "Parasitized":
                self.image_results_text.insert(tk.END, "RESULT: MALARIA DETECTED\n", "warning")
                self.image_results_text.insert(tk.END, "🦠 Blood cell appears infected with malaria parasites\n\n")
            else:
                self.image_results_text.insert(tk.END, "RESULT: NO MALARIA DETECTED\n", "success")
                self.image_results_text.insert(tk.END, "✅ Blood cell appears healthy\n\n")
            
            self.image_results_text.insert(tk.END, "CONFIDENCE SCORES:\n")
            self.image_results_text.insert(tk.END, f"Parasitized: {probability[0]:.2%}\n")
            self.image_results_text.insert(tk.END, f"Uninfected: {probability[1]:.2%}\n\n")
            
            self.image_results_text.insert(tk.END, "CONTOUR AREAS:\n")
            for i, area in enumerate(features):
                self.image_results_text.insert(tk.END, f"Contour {i+1}: {area:.1f}\n")
            
            self.image_results_text.tag_configure("warning", foreground="red", font=("Arial", 11, "bold"))
            self.image_results_text.tag_configure("success", foreground="green", font=("Arial", 11, "bold"))
            
            self.status_var.set("Image analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Image prediction failed: {str(e)}")
    
    def predict_symptoms(self):
        """Predict malaria from symptoms"""
        if not self.symptom_models:
            messagebox.showerror("Error", "Symptom models not loaded")
            return
        
        selected_model = self.symptom_model_var.get()
        if selected_model not in self.symptom_models:
            messagebox.showerror("Error", f"Model '{selected_model}' not found")
            return
        
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
            model = self.symptom_models[selected_model]
            
            if selected_model in ['SVM', 'Logistic Regression'] and self.symptom_scaler:
                features_scaled = self.symptom_scaler.transform(features_array)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0]
            
            # Display results
            self.symptom_results_text.delete(1.0, tk.END)
            self.symptom_results_text.insert(tk.END, "SYMPTOM ANALYSIS RESULTS\n")
            self.symptom_results_text.insert(tk.END, "=" * 30 + "\n\n")
            self.symptom_results_text.insert(tk.END, f"Model: {selected_model}\n\n")
            
            if prediction == 1:
                self.symptom_results_text.insert(tk.END, "RISK: SEVERE MALARIA\n", "high_risk")
                self.symptom_results_text.insert(tk.END, "🚨 High risk of severe malaria detected\n\n")
            else:
                self.symptom_results_text.insert(tk.END, "RISK: LOW MALARIA\n", "low_risk")
                self.symptom_results_text.insert(tk.END, "✅ Low risk of severe malaria\n\n")
            
            self.symptom_results_text.insert(tk.END, "CONFIDENCE SCORES:\n")
            self.symptom_results_text.insert(tk.END, f"No Severe Malaria: {probability[0]:.2%}\n")
            self.symptom_results_text.insert(tk.END, f"Severe Malaria Risk: {probability[1]:.2%}\n")
            
            self.symptom_results_text.tag_configure("high_risk", foreground="red", font=("Arial", 11, "bold"))
            self.symptom_results_text.tag_configure("low_risk", foreground="green", font=("Arial", 11, "bold"))
            
            self.status_var.set("Symptom analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Symptom prediction failed: {str(e)}")
    
    def run_integrated_analysis(self):
        """Run integrated analysis combining both approaches"""
        self.integrated_results_text.delete(1.0, tk.END)
        self.integrated_results_text.insert(tk.END, "INTEGRATED MALARIA DETECTION ANALYSIS\n")
        self.integrated_results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Check if both analyses have been performed
        if not hasattr(self, 'image_prediction') or not hasattr(self, 'symptom_prediction'):
            self.integrated_results_text.insert(tk.END, "⚠️ Please perform both image and symptom analyses first.\n")
            self.integrated_results_text.insert(tk.END, "1. Go to 'Image-Based Detection' tab and analyze blood cell\n")
            self.integrated_results_text.insert(tk.END, "2. Go to 'Symptom-Based Detection' tab and analyze symptoms\n")
            self.integrated_results_text.insert(tk.END, "3. Return here for integrated analysis\n\n")
            return
        
        # Perform integrated analysis
        self.integrated_results_text.insert(tk.END, "🔍 INTEGRATED ANALYSIS SUMMARY\n")
        self.integrated_results_text.insert(tk.END, "-" * 30 + "\n\n")
        
        # Image analysis results
        self.integrated_results_text.insert(tk.END, "📊 IMAGE ANALYSIS:\n")
        if self.image_prediction == "Parasitized":
            self.integrated_results_text.insert(tk.END, "   • Blood cell shows malaria parasites\n")
            image_risk = "HIGH"
        else:
            self.integrated_results_text.insert(tk.END, "   • Blood cell appears healthy\n")
            image_risk = "LOW"
        
        # Symptom analysis results
        self.integrated_results_text.insert(tk.END, "\n🏥 SYMPTOM ANALYSIS:\n")
        if self.symptom_prediction == 1:
            self.integrated_results_text.insert(tk.END, "   • Patient shows severe malaria symptoms\n")
            symptom_risk = "HIGH"
        else:
            self.integrated_results_text.insert(tk.END, "   • Patient shows minimal malaria symptoms\n")
            symptom_risk = "LOW"
        
        # Integrated conclusion
        self.integrated_results_text.insert(tk.END, "\n🎯 INTEGRATED CONCLUSION:\n")
        self.integrated_results_text.insert(tk.END, "-" * 25 + "\n\n")
        
        if image_risk == "HIGH" and symptom_risk == "HIGH":
            self.integrated_results_text.insert(tk.END, "🚨 CRITICAL: MALARIA CONFIRMED\n\n", "critical")
            self.integrated_results_text.insert(tk.END, "Both image and symptom analyses indicate malaria.\n")
            self.integrated_results_text.insert(tk.END, "Immediate medical treatment required.\n\n")
            recommendation = "URGENT MEDICAL ATTENTION"
            
        elif image_risk == "HIGH" or symptom_risk == "HIGH":
            self.integrated_results_text.insert(tk.END, "⚠️ SUSPECTED: MALARIA LIKELY\n\n", "warning")
            self.integrated_results_text.insert(tk.END, "One analysis suggests malaria presence.\n")
            self.integrated_results_text.insert(tk.END, "Further medical evaluation recommended.\n\n")
            recommendation = "MEDICAL EVALUATION"
            
        else:
            self.integrated_results_text.insert(tk.END, "✅ CLEAR: NO MALARIA DETECTED\n\n", "success")
            self.integrated_results_text.insert(tk.END, "Both analyses suggest no malaria.\n")
            self.integrated_results_text.insert(tk.END, "Continue monitoring for symptom changes.\n\n")
            recommendation = "ROUTINE MONITORING"
        
        # Recommendations
        self.integrated_results_text.insert(tk.END, "📋 RECOMMENDATIONS:\n")
        self.integrated_results_text.insert(tk.END, f"• {recommendation}\n")
        self.integrated_results_text.insert(tk.END, "• Follow up with healthcare provider\n")
        self.integrated_results_text.insert(tk.END, "• Monitor for symptom progression\n")
        self.integrated_results_text.insert(tk.END, "• Consider additional testing if needed\n\n")
        
        # Configure text colors
        self.integrated_results_text.tag_configure("critical", foreground="red", font=("Arial", 12, "bold"))
        self.integrated_results_text.tag_configure("warning", foreground="orange", font=("Arial", 12, "bold"))
        self.integrated_results_text.tag_configure("success", foreground="green", font=("Arial", 12, "bold"))
        
        self.status_var.set("Integrated analysis complete")

def main():
    """Main function to run the integrated system"""
    root = tk.Tk()
    app = IntegratedMalariaDetection(root)
    root.mainloop()

if __name__ == "__main__":
    main()
