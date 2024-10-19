# Predictive Maintenance for IoT Devices using Azure ML

## Overview

This project demonstrates a predictive maintenance solution for IoT devices using machine learning techniques. The goal is to anticipate device failures before they occur, reducing unplanned downtime and optimizing maintenance schedules. The solution uses **Azure Machine Learning** for model training and deployment, along with a **Random Forest Classifier** for predicting device failures.

## Project Structure

```bash
├── iot_data.csv               # Sample IoT sensor data
├── train_model.ipynb           # Jupyter notebook for model training and evaluation
├── score.py                    # Scoring script for model inference (used during deployment)
├── environment.yml             # Conda environment file
├── README.md                   # Project documentation
└── rf_model.pkl                # Saved Random Forest model (after training)
```

## Features

- **Predictive Maintenance**: Predict when an IoT device is likely to fail using historical sensor data.
- **Machine Learning**: Uses a Random Forest Classifier to train and evaluate the model.
- **Azure Integration**: Utilizes Azure Machine Learning services for model training, deployment, and monitoring.
- **Real-Time Scoring**: Deploys the model as a web service using Azure Container Instances (ACI) for real-time inference.

## Dataset

The dataset used for this project consists of simulated IoT sensor readings, with features such as `sensor1`, `sensor2`, `sensor3`, and a target variable indicating whether a device failed (`failure_event`).

- `sensor1`, `sensor2`, `sensor3`: Sensor readings for each IoT device.
- `device_age`: The age of the device in days.
- `failure_event`: Binary (0 or 1) indicating if the device failed.

### Sample Data Structure

| sensor1 | sensor2 | sensor3 | device_age | failure_event |
|---------|---------|---------|------------|---------------|
| 23.5    | 50.1    | 89.2    | 365        | 0             |
| 30.2    | 54.5    | 92.0    | 400        | 1             |

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Azure Subscription**
- **Azure Machine Learning SDK**
- **Jupyter Notebook**
- **Azure ML Workspace** (for managing experiments and deployments)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/predictive-maintenance-iot.git
   cd predictive-maintenance-iot
   ```

2. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate predictive-maintenance
   ```

3. Set up your **Azure ML Workspace** by updating the configuration in `config.json`.

### Running the Project

1. **Train the Model**: Open the `train_model.ipynb` notebook and run all cells to train the Random Forest model on the IoT dataset. This notebook will:
   - Load the dataset
   - Perform feature engineering
   - Train the model using a Random Forest Classifier
   - Evaluate the model’s performance
   - Log metrics to Azure ML

2. **Deploy the Model**: Once the model is trained, deploy it as an Azure web service by running the deployment section in the notebook. The `score.py` script is used for model inference during deployment.

3. **Test the Deployed Model**: Use the provided code to send sample data to the deployed web service and retrieve predictions.

### Example Code

#### Model Training (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('iot_data.csv')

# Split features and target
X = data[['sensor1', 'sensor2', 'sensor3', 'device_age']]
y = data['failure_event']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy}")
```

#### Model Deployment

```bash
az ml model deploy --name predictive-maintenance --model-id rf_model.pkl --inference-config score.py --compute-target aci
```

## Azure ML Pipeline

1. **Data Loading**: Load the IoT sensor data from Azure Blob or CSV.
2. **Feature Engineering**: Create features like device age and aggregate sensor readings.
3. **Model Training**: Train the Random Forest Classifier on the training data.
4. **Deployment**: Deploy the model as a web service on **Azure Container Instances (ACI)**.
5. **Real-Time Predictions**: Use the web service for real-time device failure predictions.

## Results

The model achieves an accuracy of **~90%**, effectively predicting device failures based on sensor readings and device age.

## Future Work

- **Improve Feature Engineering**: Explore additional features such as rolling averages of sensor readings.
- **Advanced Models**: Experiment with more complex models like XGBoost or Deep Learning.
- **Edge Deployment**: Consider deploying the model on IoT edge devices for faster predictions.

## Contributing

Feel free to submit pull requests or raise issues if you have suggestions for improvement.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This `README.md` file includes all the necessary details to help others understand, install, and run the predictive maintenance project. It highlights the integration with Azure ML, how the model works, and includes instructions for deploying and testing the model.
