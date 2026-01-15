First You Should have installed Python in Your System.
Then Run CMD on Root Folder
Install required packages
**python -m pip install flask flask-cors torch numpy pandas scikit-learn**
Run Prepare dataset CMD
**python training/prepare_data.py**
Then Train Your Model
**python training/train.py**
Start Flask API Backend on Port Run This CMD
**python backend/app.py**
