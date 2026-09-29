# App Analytics Decision Dashboard

Streamlit deployment of the AADM Advanced Analytics for Decision-Making project.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
Push this folder to GitHub, then create a Streamlit Community Cloud app using `app.py` as the main file.

The application uses the supplied 2,500-app dataset and retrains the deterministic project models on startup. Retention prediction excludes the actual retention percentage from its predictors to avoid target leakage.
