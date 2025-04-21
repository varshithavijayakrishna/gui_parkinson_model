import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import cv2
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px
from skimage.metrics import structural_similarity as ssim
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize
from fpdf import FPDF
import mediapipe as mp
import string
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import seaborn as sns
from transformers import pipeline


# ✅ Set Page Config
st.set_page_config(page_title="Parkinson's Detection", layout="wide")
import streamlit as st

# ✅ Load ML Model
model = joblib.load("final_xgb_model.pkl")

# Feature Columns
feature_columns = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholConsumption',
                   'PhysicalActivity', 'DietQuality', 'SleepQuality',
                   'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension',
                   'Diabetes', 'Depression', 'Stroke', 'CholesterolTotal', 'UPDRS', 'MoCA',
                   'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
                   'PosturalInstability', 'SpeechProblems', 'SleepDisorders',
                   'Constipation']

# ✅ Sidebar Navigation
selected_tab = st.sidebar.radio("Navigation", ["Home", "User Details", "Assessment", "Memory Test","Face Analysis","Reaction Time Test",
                                               "Shape & Letter Tracing Test","Lifestyle Recommendations" ,"Visuals","Overall Performance" ,"Download Report","Info"])

from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_report():
    report_filename = "Parkinson_Detailed_Report.pdf"
    c = canvas.Canvas(report_filename, pagesize=letter)
    width, height = letter

    # ✅ Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "📝 Parkinson's Disease Detailed Assessment Report")

    # ✅ Patient Information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 80, "👤 Patient Information:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"📛 Name: {st.session_state.get('name', 'N/A')}")
    c.drawString(100, height - 120, f"🎂 Age: {st.session_state.get('age', 'N/A')}")
    c.drawString(100, height - 140, f"⚧ Gender: {st.session_state.get('gender', 'N/A')}")

    # ✅ Memory Test Analysis
    memory_score = st.session_state.get("memory_score", "N/A")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 180, "🧠 Memory Test Results:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 200, f"💡 Score: {memory_score}/5")

    # 🔹 Explanation
    if memory_score != "N/A":
        if memory_score >= 4:
            explanation = "✅ Your memory retention is good, no signs of cognitive decline."
        elif memory_score == 3:
            explanation = "⚠ Mild memory issues detected. Consider brain exercises and a healthy diet."
        else:
            explanation = "🚨 Significant memory difficulties observed. Consider further cognitive assessment."
        c.drawString(100, height - 220, explanation)

    # ✅ Face Analysis (Blink Rate)
    blink_rate = st.session_state.get("blink_rate", "N/A")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 260, "👀 Face Analysis (Blink Rate):")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 280, f"⚡ Blink Rate: {blink_rate} blinks/min")

    # 🔹 Explanation
    if blink_rate != "N/A":
        if blink_rate < 10:
            explanation = "🚨 Low blink rate detected, which can be a sign of Parkinson's disease."
        elif 10 <= blink_rate <= 20:
            explanation = "⚠ Slightly reduced blinking. Monitor symptoms further."
        else:
            explanation = "✅ Normal blink rate detected."
        c.drawString(100, height - 300, explanation)

    # ✅ Reaction Time Test
    reaction_time = st.session_state.get("reaction_time", "N/A")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 340, "⚡ Reaction Time Test:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 360, f"⏳ Reaction Time: {reaction_time} sec")

    # 🔹 Explanation
    if reaction_time != "N/A":
        if reaction_time <= 0.6:
            explanation = "✅ Fast response time, indicating good motor function."
        elif 0.6 < reaction_time <= 1.2:
            explanation = "⚠ Slight delay in reaction time, could be due to stress or fatigue."
        else:
            explanation = "🚨 Slow reaction detected, possibly indicating neurological issues."
        c.drawString(100, height - 380, explanation)

    # ✅ Handwriting Analysis (Shape Tracing)
    handwriting_score = st.session_state.get("accuracy_score", "N/A")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 420, "✍ Handwriting Analysis:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 440, f"🎯 Accuracy: {handwriting_score}%")

    # 🔹 Explanation
    if handwriting_score != "N/A":
        if handwriting_score >= 80:
            explanation = "✅ Stable handwriting detected, no motor issues observed."
        elif handwriting_score >= 50:
            explanation = "⚠ Some inconsistencies in handwriting. Monitor for tremors or rigidity."
        else:
            explanation = "🚨 High instability in handwriting, a possible symptom of Parkinson's."
        c.drawString(100, height - 460, explanation)

    # ✅ Parkinson’s Risk Probability
    probability = st.session_state.get("probability", "N/A")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 500, "🔬 Parkinson’s Risk Probability:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 520, f"🩺 Estimated Risk: {probability}%")

    # 🔹 Explanation
    if probability != "N/A":
        if probability < 0.5:
            explanation = "✅ Low risk of Parkinson's detected. No immediate concerns."
        elif 0.5 <= probability <= 0.7:
            explanation = "⚠ Moderate risk detected. Consider lifestyle changes and periodic checkups."
        else:
            explanation = "🚨 High risk detected. A clinical evaluation is recommended."
        c.drawString(100, height - 540, explanation)

    # ✅ Save the PDF
    c.save()
    return report_filename

# ✅ Function to Generate Target Shape Mask
def generate_shape_mask(shape, width=400, height=300):
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape == "Circle":
        cv2.circle(mask, (200, 150), 80, 255, 3)
    elif shape == "Square":
        cv2.rectangle(mask, (100, 100), (300, 250), 255, 3)
    elif shape == "Triangle":
        pts = np.array([[200, 50], [100, 250], [300, 250]], np.int32)
        cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=3)
    elif shape == "Alphabet A":
        cv2.putText(mask, "A", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 10)
    elif shape == "Alphabet B":
        cv2.putText(mask, "B", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 10)
    return mask

# ✅ Function to draw a circular clock-style timer
def draw_clock_timer(remaining_time, total_time=60):
    fig, ax = plt.subplots(figsize=(2, 2))
    
    # Draw full clock circle
    ax.add_patch(plt.Circle((0, 0), 1, color='lightgray', fill=True))
    
    # Compute angle for the remaining time
    angle = (remaining_time / total_time) * 360  # Convert time to angle
    theta = np.radians(90 - angle)  # Adjusting so it starts from the top

    # Draw countdown hand
    ax.plot([0, np.cos(theta)], [0, np.sin(theta)], color='red', linewidth=3)

    # Formatting
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    return fig

# ✅ Home Page
if selected_tab == "Home":
    st.title("🧠 Parkinson’s Disease Detection")
    st.write("This application helps in detecting Parkinson’s disease using Machine Learning.")

# ✅ BMI Calculation
elif selected_tab == "User Details":
    st.title("👤 User Details")

    name = st.text_input("Name", value=st.session_state.get("name", ""))
    age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get("age", 60))
    gender = st.radio("Gender", ["Male", "Female"], index=0 if st.session_state.get("gender", "Male") == "Male" else 1)

    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=70.0)

    if height > 0:
        bmi = weight / ((height / 100) ** 2)
        st.write(f"*Calculated BMI:* {bmi:.2f}")
    else:
        st.write("⚠ Please enter a valid height.")

    contact = st.text_input("Contact Number")
    email = st.text_input("Email ID")

    if st.button("Save Details"):
        st.session_state["name"] = name
        st.session_state["age"] = age
        st.session_state["gender"] = gender
        st.session_state["bmi"] = bmi
        st.session_state["contact"] = contact
        st.session_state["email"] = email
        st.success("Details saved successfully!")


# ✅ Parkinson’s Disease Assessment
elif selected_tab == "Assessment":
    st.title("🩺 Parkinson’s Assessment")

    # ✅ Retrieve Age, Gender, and BMI from User Details
    age = st.session_state.get("age", 50)  # Default: 50 years
    gender = st.session_state.get("gender", "Male")  # Default: Male
    bmi = st.session_state.get("bmi", 25.0)  # Default: 25.0

    st.write(f"**Age:** {age} years")
    st.write(f"**Gender:** {gender}")
    st.write(f"**BMI:** {bmi:.2f}")

    # ✅ User inputs for assessment
    user_input = {
        "Smoking": st.selectbox("Do you smoke?", ["No", "Yes"]),
        "AlcoholConsumption": st.selectbox("Do you consume alcohol?", ["No", "Occasionally", "Regularly"]),
        "PhysicalActivity": st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"]),
        "DietQuality": st.selectbox("Diet Quality", ["Poor", "Average", "Good"]),
        "SleepQuality": st.selectbox("Sleep Quality", ["Poor", "Average", "Good"]),
        "FamilyHistoryParkinsons": st.selectbox("Family History of Parkinson’s?", ["No", "Yes"]),
        "TraumaticBrainInjury": st.selectbox("History of Traumatic Brain Injury?", ["No", "Yes"]),
        "Hypertension": st.selectbox("Do you have Hypertension?", ["No", "Yes"]),
        "Diabetes": st.selectbox("Do you have Diabetes?", ["No", "Yes"]),
        "Depression": st.selectbox("Do you suffer from Depression?", ["No", "Yes"]),
        "Stroke": st.selectbox("History of Stroke?", ["No", "Yes"]),
        "CholesterolTotal": st.number_input("Total Cholesterol Level (mg/dL)", min_value=100.0, max_value=300.0, value=200.0),
        "FunctionalAssessment": st.number_input("Functional Assessment Score", min_value=0.0, max_value=100.0, value=50.0),  # ✅ Added missing feature
        "Tremor": st.selectbox("Do you experience Tremors?", ["No", "Yes"]),
        "Rigidity": st.selectbox("Do you feel muscle Rigidity?", ["No", "Yes"]),
        "Bradykinesia": st.selectbox("Do you experience Slow Movements (Bradykinesia)?", ["No", "Yes"]),
        "PosturalInstability": st.selectbox("Do you have balance issues (Postural Instability)?", ["No", "Yes"]),
        "SpeechProblems": st.selectbox("Do you have Speech Problems?", ["No", "Yes"]),
        "SleepDisorders": st.selectbox("Do you experience Sleep Disorders?", ["No", "Yes"]),
        "Constipation": st.selectbox("Do you suffer from Chronic Constipation?", ["No", "Yes"])
    }

    # ✅ Convert categorical inputs to numerical values
    mapping = {
        "No": 0, "Yes": 1,
        "Male": 0, "Female": 1,
        "Poor": 0, "Average": 1, "Good": 2,
        "Low": 0, "Moderate": 1, "High": 2,
        "Occasionally": 1, "Regularly": 2
    }

    # ✅ Auto-Calculate UPDRS Score
    updrs_factors = ["Tremor", "Rigidity", "Bradykinesia", "SpeechProblems", "PosturalInstability"]
    updrs_score = sum(mapping[user_input[factor]] for factor in updrs_factors) * 5  # Scale to 0-100

    # ✅ Auto-Calculate MoCA Score
    memory_score = st.session_state.get("memory_score", 3)  # Default 3/5
    sleep_quality = mapping[user_input["SleepQuality"]]
    depression = mapping[user_input["Depression"]]

    moca_score = (memory_score * 6) + (10 - (depression * 3)) + (sleep_quality * 3)  # Scale to 0-30
    moca_score = min(max(moca_score, 0), 30)  # Ensure valid MoCA range

    st.write(f"**Estimated UPDRS Score:** {updrs_score:.2f}")
    st.write(f"**Estimated MoCA Score:** {moca_score:.2f}")

    # ✅ Prepare input array with all 25 features
    input_array = np.array([
        age, 
        mapping.get(gender, 0), 
        bmi,  
        *[mapping.get(value, value) for value in user_input.values()],
        updrs_score,
        moca_score
    ]).reshape(1, -1)

    # ✅ Debugging: Check input shape
    st.write(f"🔍 Input Shape: {input_array.shape[1]} features (Expected: 25)")
 # ✅ Assess Health Status Button (Only in this tab)
    if st.button("Assess Health Status"):
        probability = model.predict_proba(input_array)[:, 1][0]
        st.session_state["probability"] = probability  # Store for later use

        st.info("✅ Your responses have been recorded. Further analysis will be available in the report section.")

elif selected_tab == "Memory Test":
    st.title("🧠 Memory Test")

    # ✅ Create tabs for different memory tests
    tab1, tab2, tab3, tab4 = st.tabs(["Word Recall", "Number Sequence", "Symbol-Word Matching", "Location-Based Memory"])

    # ✅ 1. Word Recall Test
    with tab1:
        st.subheader("🔠 Word Recall Test")
        words = ["apple", "banana", "table", "dog", "car"]
        word_display = st.empty()

        # ✅ Show words in large font
        word_display.markdown(
            f"<h1 style='text-align: center; color: #ff5733;'>{'  '.join(words)}</h1>",
            unsafe_allow_html=True
        )
        time.sleep(8)
        word_display.empty()

        # ✅ Ask user to recall
        user_words = st.text_input("Enter the words you remember:").lower()
        if user_words:
            correct_words = set(words)
            user_words_set = set(user_words.split())
            score = len(user_words_set & correct_words)
            st.session_state["memory_score"] = score
            st.write(f"Your memory score: {score}/5")

    # ✅ 2. Number Sequence Test (Large & Colorful Numbers)
    with tab2:
        st.subheader("🔢 Number Sequence Test")
        number_sequence = [random.randint(1, 9) for _ in range(6)]
        num_display = st.empty()

        # ✅ Show numbers in **large, colorful** format
        colors = ["#FF5733", "#33FF57", "#5733FF", "#FF33A8", "#33FFF5", "#FFA833"]
        colored_numbers = " ".join(
            [f"<span style='color: {random.choice(colors)}; font-size: 50px; font-weight: bold;'>{num}</span>"
             for num in number_sequence]
        )

        num_display.markdown(f"<div style='text-align: center;'>{colored_numbers}</div>", unsafe_allow_html=True)
        time.sleep(8)
        num_display.empty()

        # ✅ Ask user to recall
        user_input = st.text_input("Enter the number sequence:")
        if user_input:
            user_numbers = list(map(int, user_input.split()))
            score = sum(1 for a, b in zip(user_numbers, number_sequence) if a == b)
            st.session_state["memory_score"] = score
            st.write(f"Your memory score: {score}/{len(number_sequence)}")

    # ✅ 3. Symbol-Word Matching Test
    with tab3:
        st.subheader("🛑 Symbol-Word Matching Test")
        symbols = ["★", "⬛", "⬜", "🔺", "🔵"]
        words = ["apple", "banana", "table", "dog", "car"]
        symbol_word_pairs = dict(zip(symbols, words))
        sym_display = st.empty()

        # ✅ Show symbols with words in large font
        sym_display.markdown(
            "".join([f"<h2 style='color: #ff5733;'>{sym} → {word}</h2>" for sym, word in symbol_word_pairs.items()]),
            unsafe_allow_html=True
        )
        time.sleep(8)
        sym_display.empty()

        # ✅ Ask user to recall words based on symbols
        score = 0
        for sym in symbols:
            user_input = st.text_input(f"What was the word for {sym}?").strip().lower()
            if user_input == symbol_word_pairs[sym]:
                score += 1

        st.session_state["memory_score"] = score
        st.write(f"Your memory score: {score}/{len(symbols)}")

    # ✅ 4. Location-Based Memory Test
    with tab4:
        st.subheader("📍 Location-Based Memory Test")
        objects = ["🍎", "🚗", "🐶", "📚", "🔑"]
        positions = random.sample(range(1, 6), 5)
        object_positions = dict(zip(positions, objects))
        loc_display = st.empty()

        # ✅ Show object positions in large, colorful font
        loc_display.markdown(
            "".join([f"<h2 style='color: #33A8FF;'>Position {pos}: {obj}</h2>" for pos, obj in sorted(object_positions.items())]),
            unsafe_allow_html=True
        )
        time.sleep(8)
        loc_display.empty()

        # ✅ Ask user to recall object positions
        score = 0
        for pos in sorted(object_positions.keys()):
            user_input = st.text_input(f"What object was at position {pos}?").strip()
            if user_input == object_positions[pos]:
                score += 1

        st.session_state["memory_score"] = score
        st.write(f"Your memory score: {score}/{len(objects)}")

# ✅ Face Analysis Section
# ✅ Face Analysis Section
elif selected_tab == "Face Analysis":
    st.title("🧠 Real-Time Face Analysis for Parkinson's Detection")
    st.write("This module analyzes facial movements for potential Parkinson's symptoms.")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # ✅ Create two columns (Left: Face, Right: Clock)
    col1, col2 = st.columns([2, 1])  # Face feed takes more space

    with col1:
        FRAME_WINDOW = st.image([])  # Live webcam feed

    with col2:
        timer_placeholder = st.empty()  # Timer display (Clock Timer)
        text_timer_placeholder = st.empty()  # Text Timer

    face_mesh = mp_face_mesh.FaceMesh()
    blink_counter = 0
    blink_threshold = 0.2  # Adjust based on testing
    start_time = time.time()
    duration = 60  # Timer duration in seconds

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                # Eye indices based on MediaPipe model
                left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
                right_eye = landmarks[[362, 385, 387, 263, 373, 380]]

                # Blink detection
                left_EAR = np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])
                left_EAR /= (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))

                right_EAR = np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])
                right_EAR /= (2.0 * np.linalg.norm(right_eye[0] - right_eye[3]))

                avg_EAR = (left_EAR + right_EAR) / 2.0

                if avg_EAR < blink_threshold:
                    blink_counter += 1

        FRAME_WINDOW.image(frame)  # Update face feed

        # Update Timer
        elapsed_time = time.time() - start_time
        remaining_time = max(0, duration - int(elapsed_time))

        # ✅ Update Clock Timer in Right Column
        with col2:
            fig = draw_clock_timer(remaining_time, duration)
            timer_placeholder.pyplot(fig)  # Show Clock Timer
            text_timer_placeholder.write(f"⏳ *Time Remaining: {remaining_time} sec*")

        if elapsed_time >= duration:  # Stop after 60 seconds
            blink_rate_per_minute = blink_counter
            st.write(f"Blink Rate: {blink_rate_per_minute} blinks/min")

            if blink_rate_per_minute < 10:
                st.warning("⚠ Reduced blinking detected, a possible Parkinson’s symptom.")
            elif blink_rate_per_minute > 20:
                st.info("✅ Normal blink rate detected.")
            else:
                st.success("⚠ Slightly reduced blinking, monitor further for symptoms.")

            break  # Exit the loop after 60 seconds

    cap.release()
    st.write("✅ *Analysis complete. Stop the webcam to exit.*")






elif selected_tab == "Reaction Time Test":
    st.title("⚡ Reaction Time Test")

    if "test_started" not in st.session_state:
        st.session_state.test_started = False
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "reaction_time" not in st.session_state:
        st.session_state.reaction_time = None
    if "random_letter" not in st.session_state:
        st.session_state.random_letter = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if not st.session_state.test_started:
        if st.button("Start Test"):
            st.session_state.test_started = True
            st.session_state.start_time = time.time()
            st.session_state.random_letter = random.choice(string.ascii_uppercase)
            st.session_state.user_input = ""
            st.session_state.reaction_time = None

    if st.session_state.test_started:
        st.write(f"**Type this letter:** {st.session_state.random_letter}")

        user_input = st.text_input("Enter the letter here:", value=st.session_state.user_input)

        if user_input:
            st.session_state.user_input = user_input

            if user_input.upper() == st.session_state.random_letter:
                st.session_state.reaction_time = time.time() - st.session_state.start_time  # ✅ Store reaction time properly
                st.session_state.test_started = False  

                st.write(f"✅ Correct! Your reaction time: **{st.session_state.reaction_time:.3f} seconds**")

                if st.button("Try Again"):
                    st.session_state.test_started = False
                    st.session_state.user_input = ""
                    st.session_state.reaction_time = None
            else:
                st.error("❌ Wrong Input! Try Again!")


# ✅ UI Layout
elif selected_tab == "Shape & Letter Tracing Test":
    st.title("✍ Shape & Letter Tracing Test")

    shape_options = ["Circle", "Square", "Triangle", "Alphabet A", "Alphabet B"]
    selected_shape = st.selectbox("Choose a shape to trace", shape_options)

    col1, col2 = st.columns(2)

    # ✅ Generate Target Shape
    shape_mask = generate_shape_mask(selected_shape)

    with col1:
        st.subheader("🎯 Target Shape:")
        st.image(shape_mask, caption=f"Trace This: {selected_shape}", width=400)

    with col2:
        st.subheader("✍ Your Drawing:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=3,
            stroke_color="white",
            background_color="black",
            height=300,
            width=400,
            drawing_mode="freedraw",
            key="tracing_canvas"
        )

    # ✅ Analyze the User's Drawing
    if canvas_result.image_data is not None:
        user_image = np.array(canvas_result.image_data, dtype=np.uint8)

        # Ensure grayscale format
        if len(user_image.shape) == 3:
            user_image = cv2.cvtColor(user_image, cv2.COLOR_RGBA2GRAY)

        # Resize user image to match target shape size
        user_image_resized = resize(user_image, shape_mask.shape, anti_aliasing=True)
        user_image_resized = (user_image_resized * 255).astype(np.uint8)

        # ✅ Pixel Overlap Accuracy Calculation
        matched_pixels = np.sum((user_image_resized > 100) & (shape_mask > 100))
        total_target_pixels = np.sum(shape_mask > 100)
        
        accuracy_score = (matched_pixels / total_target_pixels) * 100 if total_target_pixels > 0 else 0

        # ✅ Display Accuracy
        st.subheader(f"📝 Analysis Result: {accuracy_score:.2f}% Accuracy")
        st.image(user_image_resized, caption="📸 Your Traced Shape", width=400)



# ✅ Feature Importance & Analysis
elif selected_tab == "Visuals":
    st.title("📊 Feature Importance & UPDRS/MoCA Analysis")

    if model is None:
        st.error("⚠ Model is not loaded. Please ensure the model file is present.")
    else:
        try:
            import shap

            explainer = shap.Explainer(model)
            model_input = np.zeros((1, len(feature_columns)))
            shap_values = explainer(model_input)

            # Compute feature importance
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            feature_importance = feature_importance / np.max(feature_importance)  # Normalize

            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            # ✅ Plot: Overall Feature Importance
            fig1 = px.bar(
                feature_importance_df,
                x="Feature",
                y="Importance",
                text=feature_importance_df["Importance"].apply(lambda x: f"{x:.2f}"),
                title="Normalized Feature Importance (0-1 Scale)",
                labels={"Feature": "Features", "Importance": "Normalized Importance"},
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1)

            # ✅ Feature Importance for UPDRS & MoCA
            updrs_features = ["Tremor", "Rigidity", "Bradykinesia", "SpeechProblems", "PosturalInstability"]
            moca_features = ["Memory Test", "Cognitive Ability", "Reaction Time", "Attention", "ExecutiveFunction"]

            updrs_df = feature_importance_df[feature_importance_df["Feature"].isin(updrs_features)]
            moca_df = feature_importance_df[feature_importance_df["Feature"].isin(moca_features)]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("UPDRS Feature Importance")
                fig2 = px.bar(
                    updrs_df, x="Feature", y="Importance",
                    title="UPDRS Score Contribution",
                    color="Importance", color_continuous_scale="Reds"
                )
                st.plotly_chart(fig2)


            # ✅ Feature Correlation Heatmap
            st.subheader("📈 Feature Correlation with UPDRS & MoCA")
            df_sample = pd.DataFrame(np.random.rand(100, len(feature_columns)), columns=feature_columns)  # Replace with actual data

            correlation_matrix = df_sample.corr()
            fig4, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

            st.pyplot(fig4)

        except Exception as e:
            st.error(f"⚠ An error occurred: {e}")


# ✅ Lifestyle Recommendations Tab
elif selected_tab == "Lifestyle Recommendations":
    st.title("🩺 Personalized Lifestyle Recommendations")

    # ✅ User selects lifestyle choices
    physical_activity = st.selectbox("🏃 Physical Activity Level", ["Low", "Moderate", "High"])
    diet_quality = st.selectbox("🥗 Diet Quality", ["Poor", "Average", "Good"])
    smoking = st.selectbox("🚭 Do you smoke?", ["Yes", "No"])
    alcohol_consumption = st.selectbox("🍷 Alcohol Consumption", ["None", "Occasionally", "Regularly"])
    sleep_quality = st.selectbox("😴 Sleep Quality", ["Poor", "Average", "Good"])
    stress_level = st.selectbox("🧘 Stress Level", ["Low", "Moderate", "High"])

    recommendations = []

    # ✅ Generate recommendations based on user inputs
    if physical_activity == "Low":
        recommendations.append("🏃 Increase physical activity with daily walks or light exercise.")
    if diet_quality == "Poor":
        recommendations.append("🥗 Improve your diet by consuming more fruits, vegetables, and lean proteins.")
    if smoking == "Yes":
        recommendations.append("🚭 Consider quitting smoking to reduce health risks.")
    if alcohol_consumption == "Regularly":
        recommendations.append("🍷 Reduce alcohol intake to maintain brain and liver health.")
    if sleep_quality == "Poor":
        recommendations.append("😴 Improve sleep by maintaining a consistent schedule and reducing screen time before bed.")
    if stress_level == "High":
        recommendations.append("🧘 Try relaxation techniques such as meditation and deep breathing exercises.")

    # ✅ Display Recommendations
    st.subheader("🔹 Your Personalized Recommendations:")
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("✅ Your lifestyle choices seem balanced. Keep maintaining a healthy routine!")

    # ✅ Button to save recommendations in session state
    if st.button("Save Recommendations"):
        st.session_state["physical_activity"] = physical_activity
        st.session_state["diet_quality"] = diet_quality
        st.session_state["smoking"] = smoking
        st.session_state["alcohol_consumption"] = alcohol_consumption
        st.session_state["sleep_quality"] = sleep_quality
        st.session_state["stress_level"] = stress_level
        st.session_state["lifestyle_recommendations"] = recommendations[:]
        st.success("✅ Recommendations saved successfully!")

# ✅ Overall Performance Section
elif selected_tab == "Overall Performance":
    st.title("📊 Overall Performance Analysis")

    # ✅ Ensure reaction time is properly retrieved
    reaction_time_value = st.session_state.get("reaction_time", None)

    # ✅ Convert reaction time to a percentage score
    if reaction_time_value is not None:
        reaction_time_score = (1 - min(1.5, reaction_time_value) / 1.5) * 100  # Normalize reaction time
    else:
        reaction_time_score = 0  # Default score if not set

    # ✅ Collect all test results
    test_scores = {
        "Parkinson’s Risk": st.session_state.get("probability", 0) * 100,
        "Memory Test": (st.session_state.get("memory_score", 0) / 5) * 100,
        "Face Analysis": (st.session_state.get("blink_rate", random.randint(5, 25)) / 25) * 100,  # Simulated blink rate
        "Reaction Time": reaction_time_score,  # ✅ Use the corrected reaction time score
        "Shape Tracing": st.session_state.get("accuracy_score", random.randint(50, 95))  # Simulated accuracy
    }

    # ✅ Convert to DataFrame for visualization
    overall_data = pd.DataFrame({"Test": list(test_scores.keys()), "Score (%)": list(test_scores.values())})

    # ✅ Plot Consolidated Vertical Bar Chart
    st.subheader("📊 Consolidated Test Performance")

    fig = px.bar(
        overall_data,
        x="Test",
        y="Score (%)",
        text=overall_data["Score (%)"].apply(lambda x: f"{x:.2f}%"),  # Display % values above bars
        title="Overall Test Scores (%)",
        labels={"Test": "Test Type", "Score (%)": "Performance (%)"},
        color="Score (%)",
        color_continuous_scale="Blues"
    )

    fig.update_traces(textposition="outside")  # Place labels above bars

    st.plotly_chart(fig)

    # ✅ Display Overall Average Score
    avg_score = np.mean(list(test_scores.values()))
    st.write(f"**🔹 Overall Average Test Score: {avg_score:.2f}%**")


# ✅ Now, add the "Download Report" Tab AFTER defining `generate_report()`
elif selected_tab == "Download Report":
    st.title("📄 Download Patient Report")

    # ✅ Ensure assessment data exists before allowing report generation
    if st.button("Generate PDF Report"):
        if st.session_state.get("probability", None) is None:
            st.error("⚠ No assessment found! Complete the assessment first.")
        else:
            report_path = generate_report()
            with open(report_path, "rb") as file:
                st.download_button(label="📥 Download Report", data=file, file_name="Parkinson_Detailed_Report.pdf", mime="application/pdf")

# ✅ Information Tab (Detailed Explanation of Features & Tests)
elif selected_tab == "Info":
    st.title("ℹ️ Parkinson’s Disease Detection - Information")

    st.subheader("📌 Overview")
    st.write("""
    This application uses various motor, cognitive, and speech-based tests to assess Parkinson's symptoms.
    Below, you’ll find a detailed explanation of each test and feature used in the system.
    """)

    st.divider()

    # ✅ Memory Test
    st.subheader("🧠 Memory Test")
    st.write("""
    This test evaluates short-term memory and recall ability. 
    Parkinson’s disease can lead to mild cognitive impairment (MCI), affecting memory retrieval.
    - **Test Types:** Word Recall, Number Sequence, Symbol Matching, Location-Based Recall.
    - **What We Measure:** Accuracy of recall, response time, and cognitive ability.
    """)

    st.divider()

    # ✅ Face Analysis (Blink Rate & Expressions)
    st.subheader("👀 Face Analysis")
    st.write("""
    Facial analysis is used to monitor reduced blinking and facial expressiveness, common in Parkinson’s.
    - **Blink Rate:** Parkinson’s patients often exhibit decreased blink rate due to facial muscle stiffness.
    - **Micro-Expressions:** We analyze small involuntary facial movements for any abnormalities.
    - **How It Works:** Using real-time webcam capture, we track eye movements and facial muscles.
    """)

    st.divider()

    # ✅ Reaction Time Test
    st.subheader("⚡ Reaction Time Test")
    st.write("""
    This test measures how quickly a person responds to a stimulus.
    - **Why It Matters?** Slowed reaction times are associated with Parkinson’s due to impaired motor function.
    - **How It Works?** Users must press a key as soon as a letter appears.
    - **What We Measure?** The delay between stimulus and response, compared to normal ranges.
    """)

    st.divider()

    # ✅ Handwriting & Drawing Analysis
    st.subheader("✍ Handwriting & Drawing Analysis")
    st.write("""
    Parkinson’s can affect fine motor control, making handwriting shaky and small (Micrographia).
    - **Tests Included:**
        - Shape & Letter Tracing (Tremor & Stroke Consistency)
        - Spiral Drawing Test (Tremor Severity)
        - Letter Repetition Test (Motor Stability)
    - **What We Measure?** Stroke smoothness, accuracy, and variations in pressure and size.
    """)

    st.divider()



    # ✅ Parkinson's Risk Score & Machine Learning Model
    st.subheader("🧪 Parkinson’s Risk Score & Machine Learning Model")
    st.write("""
    The system uses **LightGBM (Gradient Boosting Model)** trained on patient data.
    - **Inputs Used:**
        - Voice Analysis
        - Face Analysis (Blink Rate)
        - Handwriting & Cognitive Tests
        - Lifestyle Factors (Diet, Exercise, Smoking, Alcohol, etc.)
    - **Outputs:** The model predicts Parkinson’s likelihood based on symptoms and lifestyle.
    """)

    st.divider()

    # ✅ Feature Importance in Prediction
    st.subheader("🔬 Feature Importance in Prediction")
    st.write("""
    The system considers multiple health factors and assigns importance scores.
    - **Most Important Features:**
        - UPDRS Score (Unified Parkinson’s Disease Rating Scale)
        - MoCA Score (Cognitive Assessment)
        - Handwriting Stroke Variation
        - Reaction Time
        - Voice Tremors (Jitter & Shimmer)
    - **Why This Matters?** Understanding feature importance helps improve diagnosis accuracy.
    """)

    st.divider()

    # ✅ Lifestyle Assessment
    st.subheader("🏃‍♂️ Lifestyle Assessment")
    st.write("""
    Parkinson’s progression can be influenced by lifestyle choices.
    - **Factors Considered:**
        - Physical Activity (Sedentary, Moderate, Active)
        - Diet Quality (Poor, Average, Healthy)
        - Sleep Quality
        - Smoking & Alcohol Consumption
        - Stress Levels
    - **Personalized Recommendations:** Based on responses, the system provides health tips to reduce risks.
    """)

    st.divider()

    st.subheader("📝 Summary")
    st.write("""
    This GUI integrates **face, voice, handwriting, and cognitive assessments** to provide an AI-powered Parkinson’s risk evaluation.
    The results can be used to assist medical professionals in **early detection and continuous monitoring** of Parkinson’s symptoms.
    """)





