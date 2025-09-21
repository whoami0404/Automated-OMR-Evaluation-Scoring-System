```markdown
# Automated OMR Evaluation — Streamlit

This repository contains a Streamlit app implementing an Automated OMR Evaluation system.

Features
- Upload OMR sheet image (photo or scan)
- Optional automatic alignment (deskew + perspective transform)
- Bubble detection and scoring using an answer key
- Edit or upload answer key (JSON/CSV)
- Per-question results table, subject-wise scores, and overlay image
- Downloadable CSV and overlay PNG

Quick start (local)
1. Create and activate a Python environment (recommended).
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   streamlit run streamlit_app.py
4. In the sidebar optionally enable alignment and tweak detection thresholds.

Deploy to Streamlit Community Cloud
1. Push the repository to GitHub.
2. Go to https://share.streamlit.io and create a new app pointing to this repo and `streamlit_app.py`.
3. Streamlit will install dependencies from `requirements.txt` and deploy.

Notes & tips
- If detection fails or finds extra noise: tweak the bubble width/height/area and adaptive threshold settings.
- The app groups detected bubble-like contours into rows of 4 (best-effort). If your template differs, preprocessing or template-specific mapping may be needed.
- The answer key can be edited directly in the sidebar as JSON or uploaded as CSV/JSON.

License
- Add your preferred license when creating the GitHub repo.
```