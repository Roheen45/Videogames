# Video game sales ML demo

Small two-model example built from `vgsales.csv.zip`.

## Setup and run

1. Put `vgsales.csv.zip` in this folder.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Train and save both models: `python train_models.py`
4. Start the web app: `uvicorn main:app --reload`
5. Open <http://127.0.0.1:8000>.

The API also documents itself at <http://127.0.0.1:8000/docs>. The app trains linear regression to estimate global sales from regional sales (a baseline, since global sales is the sum of regional sales) and a random forest classifier to predict genre from platform and release year. Training prints test-set R² and accuracy. Saved model files go in `models/`.
