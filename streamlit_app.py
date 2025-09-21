# ===============================
# streamlit_app.py
# Automated OMR Evaluation System (MVP)
# ===============================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Automated OMR Evaluation", layout="wide")
st.title("🎯 Automated OMR Evaluation & Scoring System")
st.markdown("Upload OMR sheets and get per-subject scores instantly!")

# -----------------
# Create output folder
# -----------------
os.makedirs("omr_output", exist_ok=True)

# -----------------
# Upload OMR Image
# -----------------
uploaded_file = st.file_uploader("📂 Upload your OMR sheet image (jpg/png)", type=["jpg","png"])
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded OMR Sheet", use_column_width=True)

    # -----------------
    # Preprocessing
    # -----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # -----------------
    # Detect Bubble Contours
    # -----------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect = w / float(h)
        area = cv2.contourArea(cnt)
        if 20 < w < 50 and 20 < h < 50 and 0.8 <= aspect <= 1.2 and 400 < area < 2000:
            bubble_contours.append((x, y, w, h))
    st.write(f"🔵 Total bubble-like contours found: {len(bubble_contours)}")

    # Sort by Y then X
    bubble_contours = sorted(bubble_contours, key=lambda c: (c[1], c[0]))

    # -----------------
    # Define Answer Key
    # -----------------
    answer_key = {
        1:'a',2:'c',3:'c',4:'c',5:'c',6:'a',7:'c',8:'c',9:'b',10:'c',
        11:'a',12:'a',13:'d',14:'a',15:'b',16:['a','b','c','d'],17:'c',18:'d',19:'a',20:'b',
        21:'a',22:'d',23:'b',24:'a',25:'c',26:'b',27:'a',28:'b',29:'d',30:'c',
        31:'c',32:'a',33:'b',34:'c',35:'a',36:'a',37:'d',38:'b',39:'a',40:'b',
        41:'c',42:'c',43:'c',44:'b',45:'b',46:'a',47:'c',48:'b',49:'d',50:'a',
        51:'c',52:'b',53:'c',54:'c',55:'a',56:'b',57:'b',58:'a',59:['a','b'],60:'b',
        61:'b',62:'c',63:'a',64:'b',65:'c',66:'b',67:'b',68:'c',69:'c',70:'b',
        71:'b',72:'b',73:'d',74:'b',75:'a',76:'b',77:'b',78:'b',79:'b',80:'b',
        81:'a',82:'b',83:'c',84:'b',85:'c',86:'b',87:'b',88:'b',89:'a',90:'b',
        91:'c',92:'b',93:'c',94:'b',95:'b',96:'c',97:'c',98:'a',99:'b',100:'c'
    }

    # -----------------
    # Map bubbles to answers
    # -----------------
    questions = {}
    question_num = 1
    rows = [bubble_contours[i:i+4] for i in range(0, len(bubble_contours), 4)]

    for row in rows:
        row = sorted(row, key=lambda c: c[0])
        answers = ['a','b','c','d']
        marked = []
        for ans, (x,y,w,h) in zip(answers, row):
            roi = thresh[y:y+h, x:x+w]
            filled_ratio = cv2.countNonZero(roi) / float(w*h)
            if filled_ratio > 0.5:
                marked.append(ans)
        questions[question_num] = marked
        question_num += 1

    # -----------------
    # Evaluate answers
    # -----------------
    results = []
    for q, marked in questions.items():
        correct = answer_key.get(q)
        if isinstance(correct, list):
            is_correct = sorted(marked) == sorted(correct)
        else:
            is_correct = (len(marked)==1 and marked[0]==correct)
        subject = (
            "Python" if q<=20 else
            "EDA" if q<=40 else
            "SQL" if q<=60 else
            "POWER BI" if q<=80 else
            "Statistics"
        )
        results.append({
            "Qno": q,
            "Marked": ",".join(marked) if marked else "-",
            "Correct": correct if isinstance(correct,str) else ",".join(correct),
            "IsCorrect": int(is_correct),
            "Subject": subject
        })

    df = pd.DataFrame(results)
    df.to_csv("omr_output/per_question_results.csv", index=False)

    # Subject-wise scores
    subject_scores = df.groupby("Subject")["IsCorrect"].sum()
    total_score = subject_scores.sum()
    st.subheader("📊 Per-subject scores")
    st.bar_chart(subject_scores)
    st.write(f"✅ Total Score (out of 100): {total_score}")

    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Per-question Results CSV",
        data=csv,
        file_name='per_question_results.csv',
        mime='text/csv'
    )

    # Debug overlay
    overlay = img.copy()
    for q, row in zip(range(1,len(rows)+1), rows):
        row = sorted(row, key=lambda c: c[0])
        answers = ['a','b','c','d']
        for ans, (x,y,w,h) in zip(answers, row):
            color = (0,255,0) if ans in questions[q] else (0,0,255)
            cv2.rectangle(overlay, (x,y), (x+w,y+h), color, 2)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Detected Answers Overlay", use_column_width=True)
