import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the dataframe by handling missing values and ensuring proper numeric types.
    Safely injects missing required columns to prevent crashes.
    """
    # Columns required for risk calculation
    numeric_cols = [
        'Life Completed (in Yrs)',
        'Progressive Breakdown Hrs',
        'Actual Progressive % Availability',
        'Actual Progressive % Utilization',
        'Total Running Hrs.'
    ]
    
    # Safely ensure all numeric columns exist
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0 # Default if entirely missing

    # Fill missing values with appropriate defaults
    for col in numeric_cols:
        # Convert to numeric, forcing errors to NaN, then fill NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Availability' in col or 'Utilization' in col:
            df[col] = df[col].fillna(100) # Default to 100% if missing
        else:
            df[col] = df[col].fillna(0)   # Default to 0 if missing
                
    return df

def calculate_risk_score(row):
    """
    Calculates a risk score (0-100) based on machine metrics.
    """
    try:
        # Extract features safely
        life_completed = row.get('Life Completed (in Yrs)', 0)
        breakdown_hrs = row.get('Progressive Breakdown Hrs', 0)
        availability = row.get('Actual Progressive % Availability', 100)
        utilization = row.get('Actual Progressive % Utilization', 100)
        total_hrs = row.get('Total Running Hrs.', 0)

        # 1. Life Completed Risk (Max 30 points)
        life_risk = min((life_completed / 20.0) * 30, 30)

        # 2. Breakdown Risk (Max 30 points)
        if total_hrs > 0:
            breakdown_ratio = breakdown_hrs / total_hrs
            breakdown_risk = min((breakdown_ratio * 10) * 30, 30)
        else:
            breakdown_risk = 0

        # 3. Availability Risk (Max 20 points)
        avail_risk = min(max(100 - availability, 0), 20)

        # 4. Utilization Risk (Max 20 points)
        util_risk = min(max(100 - utilization, 0), 20)

        # Total Score
        total_score = life_risk + breakdown_risk + avail_risk + util_risk
        return round(min(total_score, 100), 2)

    except Exception:
        # Fallback
        return 0.0

def get_risk_category(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"

def get_priority(category):
    if category == "HIGH":
        return "Priority 1"
    elif category == "MEDIUM":
        return "Priority 2"
    else:
        return "Priority 3"

def get_maintenance_recommendation(category):
    if category == "HIGH":
        return "Immediate Inspection"
    elif category == "MEDIUM":
        return "Service Within 7 Days"
    else:
        return "Healthy"

def get_next_action(category):
    if category == "HIGH":
        return "Schedule Immediate Inspection"
    elif category == "MEDIUM":
        return "Plan Routine Service"
    else:
        return "Continue Normal Operation"

def process_machine_data(df):
    """
    End-to-end processing of the machine dataframe:
    Cleans data, calculates risk scores, assigns categories, priorities, and recommendations.
    """
    df = clean_data(df.copy())
    
    # Calculate Risk Metrics
    df['Risk Score'] = df.apply(calculate_risk_score, axis=1)
    df['Risk Category'] = df['Risk Score'].apply(get_risk_category)
    
    # Priority & Recommendations
    df['Priority'] = df['Risk Category'].apply(get_priority)
    df['Maintenance Recommendation'] = df['Risk Category'].apply(get_maintenance_recommendation)
    df['Next Action'] = df['Risk Category'].apply(get_next_action)
    
    return df
