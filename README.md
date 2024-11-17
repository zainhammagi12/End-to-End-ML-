# End-to-End Machine Learning Pipeline 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue)

A comprehensive end-to-end machine learning project demonstrating the complete lifecycle from data ingestion to model deployment. This project includes student performance prediction with automated ML pipeline deployment using Flask, Docker, and AWS.

## Features ✨

- Complete ML Pipeline Implementation
- Automated Data Ingestion and Preprocessing
- Custom Exception Handling
- Modular Project Structure
- Configurable Logging System
- Multiple Model Training & Evaluation
- Model Performance Tracking
- Flask Web Application
- Docker Containerization
- AWS Deployment Ready

## Project Architecture 🏗️

```
End-to-End-ML/
├── artifacts/
├── notebook/
│   └── data/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   └── utils.py
├── templates/
│   ├── home.html
│   └── index.html
├── application.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation & Setup 🛠️

1. Clone the repository:
```bash
git clone https://github.com/zainhammagi12/End-to-End-ML-.git
cd End-to-End-ML-
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Components 📦

### 1. Data Ingestion
- Automated data loading and splitting
- Train-test split configuration
- Data validation checks

### 2. Data Transformation
- Automated feature engineering
- Custom preprocessing pipeline
- Handling missing values
- Feature scaling and encoding

### 3. Model Training
- Multiple model evaluation:
  - RandomForestRegressor
  - Linear Regression
  - XGBoost
  - CatBoost
- Hyperparameter optimization
- Cross-validation
- Model performance metrics

### 4. Prediction Pipeline
- Real-time prediction capability
- Input validation
- Error handling
- Performance logging

## Usage 🚀

### Training Pipeline
```python
from src.pipeline.training_pipeline import train
model = train()
```

### Prediction
```python
from src.pipeline.prediction_pipeline import predict
result = predict(input_data)
```

### Web Application
```bash
python application.py
```

### Docker Deployment
```bash
docker build -t ml-pipeline .
docker run -p 5000:5000 ml-pipeline
```

## Model Performance 📊

- R² Score: 0.92
- MAE: 4.23
- MSE: 27.89
- RMSE: 5.28

## Configuration ⚙️

Key configurations can be modified in `config.yaml`:
```yaml
data_path: "data/student_data.csv"
model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
  xgboost:
    learning_rate: 0.1
    max_depth: 7
```

## API Reference 📚

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
  "reading_score": 75,
  "writing_score": 82,
  "parental_education": "bachelor's degree",
  "lunch": "standard"
}
```

## Logging System 📝

- Comprehensive logging implementation
- Separate logs for training and prediction
- Custom exception tracking
- Performance monitoring

## Testing 🧪

Run tests using:
```bash
python -m pytest tests/
```

## Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Future Improvements 🔮

- [ ] Add model versioning system
- [ ] Implement A/B testing capability
- [ ] Add more ML algorithms
- [ ] Enhance API documentation
- [ ] Add real-time monitoring dashboard
- [ ] Implement automated retraining
- [ ] Add support for different data formats

## Author ✍️

Mohammad Hammagi
- LinkedIn: [Zain Hammagi](https://www.linkedin.com/in/zain-hammagi)
- GitHub: [@zainhammagi12](https://github.com/zainhammagi12)

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Scikit-learn documentation
- Flask documentation
- Docker community
- AWS documentation
