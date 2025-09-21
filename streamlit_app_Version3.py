#!/usr/bin/env python3
"""
Streamlit app: Automated OMR Evaluation with optional deskew/perspective alignment
- Upload OMR image
- Optional auto-alignment (deskew + perspective)
- Bubble detection and scoring (answer key editable/uploadable)
- Downloads: per-question CSV, overlay PNG
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import json

st.set_page_config(page_title="Automated OMR Evaluation", layout="wide")

st.title("📄 Automated OMR Evaluation")

st.sidebar.header("Detection settings")
enable_align = st.sidebar.checkbox("Enable auto-align (deskew + perspective)", value=True)
w_min = st.sidebar.number_input("Min bubble width (px)", min_value=5, max_value=200, value=20)
w_max = st.sidebar.number_input("Max bubble width (px)", min_value=10, max_value=500, value=50)
h_min = st.sidebar.number_input("Min bubble height (px)", min_value=5, max_value=200, value=20)
h_max = st.sidebar.number_input("Max bubble height (px)", min_value=10, max_value=500, value=50)
area_min = st.sidebar.number_input("Min contour area", min_value=0, max_value=100000, value=400)
area_max = st.sidebar.number_input("Max contour area", min_value=0, max_value=100000, value=2000)
aspect_lo = st.sidebar.slider("Min aspect ratio (w/h)", 0.1, 3.0, 0.8)
aspect_hi = st.sidebar.slider("Max aspect ratio (w/h)", 0.1, 3.0, 1.2)
fill_threshold = st.sidebar.slider("Fill threshold (fraction)", 0.0, 1.0, 0.5)
adaptive_block = st.sidebar.number_input("Adaptive Threshold Block Size (odd)", min_value=3, step=2, value=11)
adaptive_C = st.sidebar.number_input("Adaptive Threshold C", value=2)

st.sidebar.header("Answer key")
answer_key_upload = st.sidebar.file_uploader("Upload answer key (json or csv)", type=["json","csv"])
default_answer_key = {
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

def load_answer_key(upload):
    if upload is None:
        return default_answer_key
    name = upload.name.lower()
    content = upload.read()
    try:
        if name.endswith(".json"):
            return json.loads(content.decode("utf-8"))
        else:
            df = pd.read_csv(io.BytesIO(content))
            # Expect columns Qno,Answer or similar
            if "Qno" in df.columns and "Answer" in df.columns:
                return {int(r["Qno"]): (r["Answer"] if not pd.isna(r["Answer"]) else "") for _, r in df.iterrows()}
            elif df.shape[1] >= 2:
                return {int(r[0]): r[1] for _, r in df.iterrows()}
    except Exception as e:
        st.sidebar.error(f"Failed to parse answer key: {e}")
    st.sidebar.warning("Using default answer key.")
    return default_answer_key

answer_key = load_answer_key(answer_key_upload)

st.sidebar.markdown("Edit answer key (JSON) — optional")
answer_key_text = st.sidebar.text_area("Answer key JSON", value=json.dumps(answer_key, indent=2), height=250)
try:
    parsed = json.loads(answer_key_text)
    answer_key = parsed
except Exception:
    st.sidebar.warning("Answer key JSON invalid; using uploaded/default key.")

uploaded = st.file_uploader("Upload OMR sheet image (png/jpg)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an OMR image to start.")
    st.stop()

# Read image
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read image. Try another file.")
    st.stop()

st.subheader("Original Image")
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

# --- alignment helpers ---
def order_points(pts):
    # pts: 4x2
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_document_contour(image_gray):
    # try to find the biggest 4-point contour
    blurred = cv2.GaussianBlur(image_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

aligned_img = img.copy()
aligned_preview = None
if enable_align:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    doc_pts = find_document_contour(gray)
    if doc_pts is not None:
        try:
            aligned_img = four_point_transform(img, doc_pts)
            aligned_preview = aligned_img
            st.subheader("Auto-aligned Image (preview)")
            st.image(cv2.cvtColor(aligned_preview, cv2.COLOR_BGR2RGB), use_column_width=True)
            save_aligned = st.checkbox("Save aligned image for download", value=True)
        except Exception as e:
            st.warning(f"Auto-align failed: {e}")
            aligned_img = img.copy()
    else:
        st.warning("Could not find document contour for alignment. Using original image.")
        aligned_img = img.copy()

# Preprocessing on aligned image
gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, int(adaptive_block), int(adaptive_C))

st.subheader("Threshold preview")
st.image(thresh, channels="GRAY", use_column_width=True)

# Detect contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bubble_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / float(h) if h != 0 else 0
    area = cv2.contourArea(cnt)
    if w_min < w < w_max and h_min < h < h_max and aspect_lo <= aspect <= aspect_hi and area_min < area < area_max:
        bubble_contours.append((x, y, w, h))

st.write(f"🔵 Bubble-like contours found: {len(bubble_contours)}")

# Sort contours top->bottom then left->right
bubble_contours = sorted(bubble_contours, key=lambda c: (c[1], c[0]))

# Group into rows of 4 (best-effort)
rows = [bubble_contours[i:i+4] for i in range(0, len(bubble_contours), 4)]

if any(len(r) != 4 for r in rows):
    st.warning("Some detected rows don't have exactly 4 bubbles. Try adjusting detection parameters.")

# Map bubbles to answers
questions = {}
question_num = 1
for row in rows:
    if len(row) < 4:
        # skip incomplete rows (or you could attempt smarter grouping)
        continue
    row = sorted(row, key=lambda c: c[0])  # left→right
    answers = ['a','b','c','d']
    marked = []
    for ans, (x,y,w,h) in zip(answers, row):
        roi = thresh[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        filled_ratio = cv2.countNonZero(roi) / float(w*h)
        if filled_ratio > float(fill_threshold):
            marked.append(ans)
    questions[question_num] = marked
    question_num += 1

# Evaluate
results = []
for q, marked in questions.items():
    correct = answer_key.get(str(q)) or answer_key.get(q)
    if isinstance(correct, list):
        is_correct = sorted(marked) == sorted(correct)
        correct_str = ",".join(correct)
    else:
        is_correct = (len(marked) == 1 and marked[0] == correct)
        correct_str = correct if correct is not None else "-"
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
        "Correct": correct_str,
        "IsCorrect": int(is_correct),
        "Subject": subject
    })

df = pd.DataFrame(results)
st.subheader("Per-question results")
st.dataframe(df, height=300)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download per-question CSV", data=csv_bytes, file_name="per_question_results.csv", mime="text/csv")

# Subject-wise scores
subject_scores = df.groupby("Subject")["IsCorrect"].sum().reindex(["Python","EDA","SQL","POWER BI","Statistics"]).fillna(0).astype(int)
st.subheader("Subject-wise scores (out of 20)")
st.table(subject_scores)

# Bar chart
fig, ax = plt.subplots(figsize=(7,3))
subject_scores.plot(kind="bar", color="orange", ax=ax)
ax.set_ylim(0, 20)
ax.set_ylabel("Correct answers (out of 20)")
ax.set_title("Subject-wise Scores")
plt.tight_layout()
st.pyplot(fig)

# Overlay debug image with rectangles
overlay = aligned_img.copy()
for q, row in zip(range(1, len(rows)+1), rows):
    if len(row) < 4:
        continue
    row = sorted(row, key=lambda c: c[0])
    answers = ['a','b','c','d']
    for ans, (x,y,w,h) in zip(answers, row):
        color = (0,255,0) if ans in questions.get(q, []) else (0,0,255)
        cv2.rectangle(overlay, (x,y), (x+w,y+h), color, 2)

st.subheader("Detected Answers Overlay")
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
st.image(overlay_rgb, use_column_width=True)

# Downloads
success, png = cv2.imencode('.png', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
if success:
    st.download_button("Download overlay PNG", data=png.tobytes(), file_name="detected_answers.png", mime="image/png")

if enable_align and aligned_preview is not None:
    success2, png2 = cv2.imencode('.png', cv2.cvtColor(aligned_preview, cv2.COLOR_BGR2RGB))
    if success2:
        st.download_button("Download aligned image PNG", data=png2.tobytes(), file_name="aligned_image.png", mime="image/png")

st.success("Processing complete. Tweak sidebar controls if detection is noisy or misses bubbles.")