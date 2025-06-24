# Weather Prediction

## Project Overview

The Weather Prediction project leverages machine learning algorithms to forecast weather conditions based on historical data. By analyzing patterns in past weather data, the model predicts future weather scenarios, aiding in various applications such as agriculture planning, event management, and more.

## Features

- **Data Collection**: Aggregates historical weather data from multiple sources.
- **Data Preprocessing**: Cleans and prepares data for modeling.
- **Model Training**: Utilizes machine learning algorithms to train predictive models.
- **Prediction**: Provides forecasts for future weather conditions.
- **Visualization**: Offers graphical representations of data and predictions.


## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Preksha-0211/Weather_Prediction.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd Weather_Prediction
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Launch the Jupyter interface** (choose one):

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Run the command-line interface**:

   ```bash
   # Train the model
   python weather_cli.py train <train_csv> <test_csv> <model_out>

   # Generate predictions
   python weather_cli.py predict <model_out> <test_csv> <predictions_csv>
   ```
