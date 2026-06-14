# Predictive Maintenance System

A machine health monitoring and predictive maintenance dashboard for mining equipment.

## Project Structure
- `data/machine_data.xlsx`: The dataset containing machine statistics.
- `models/maintenance_model.pkl`: The trained Random Forest model.
- `requirements.txt`: Python dependencies.
- `risk_engine.py`: Logic for computing risk scores.
- `train_model.py`: Script to train the Machine Learning model.
- `app.py`: Streamlit dashboard application.

## 1. How to Install Requirements

1. Open a terminal or Command Prompt.
2. Navigate to the project directory:
   ```bash
   cd "c:\Users\KIIT0001\Desktop\CCL FINAL"
   ```
3. Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## 2. How to Train the Model

The dashboard will work fine without the model using just the risk engine, but training it unlocks predictive breakdown probabilities.

1. Ensure your dataset is present at `data/machine_data.xlsx`.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. You will see an output confirming the model was saved to `models/maintenance_model.pkl`.

## 3. How to Run the Dashboard

1. Run the Streamlit application using the following command:
   ```bash
   streamlit run app.py
   ```
2. A new tab will automatically open in your web browser showing the professional dashboard.

## 4. How to Troubleshoot Common Errors

- **`Dataset not found` Error**: Make sure you have placed the Excel file exactly at `data/machine_data.xlsx`.
- **`ModuleNotFoundError`**: This means a required package isn't installed. Run `pip install -r requirements.txt` again.
- **Model Training Fails due to missing columns**: Make sure your dataset contains the exact columns needed (e.g., `Progressive Work Hours`, `Total Running Hrs.`, etc.). The dashboard will still run gracefully without the model.
- **Port already in use**: If Streamlit says the port is in use, you can specify a different port: `streamlit run app.py --server.port 8502`.
