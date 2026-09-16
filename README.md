# FareSense

FareSense is a beginner-friendly machine-learning web application that estimates the fare of an Indian domestic flight from its airline, route, schedule, cabin class, journey duration, stops, and booking lead time.

The interface uses a minimal black-and-green terminal theme and explains every input. The prediction is an educational estimate based on historical data, not a live airline quotation.

This serves no purpose and only exists because i wanted to learn stuff
## What this project demonstrates

- Supervised machine-learning regression
- Data ingestion and train/test splitting
- Numerical and categorical preprocessing
- Training and comparison of regression models
- Experiment tracking with MLflow
- Dataset and pipeline tracking with DVC
- Saving and loading trained Python objects
- Serving predictions through Flask
- A responsive HTML, CSS, and JavaScript interface

## Machine-learning approach

This is a **supervised regression** project:

- **Supervised** means the training examples contain flight details and their known historical prices.
- **Regression** means the output is a continuous number, such as ₹8,750, rather than a category.

The training pipeline compares Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, and Random Forest. The model with the highest test R² score is saved.

The current saved model is a **Random Forest Regressor with 100 decision trees**. Random Forest is useful because fare relationships are nonlinear: the effect of duration or booking lead time can depend on the route, airline, cabin class, and number of stops.

## Model inputs

| Field | Meaning |
| --- | --- |
| airline | Airline operating the flight |
| source_city | City where the journey begins |
| destination_city | City where the journey ends |
| departure_time | General time period when the flight departs |
| arrival_time | General time period when the flight arrives |
| stops | Non-stop, one stop, or two or more stops |
| classs | Economy or Business cabin |
| duration | Total journey duration in decimal hours, including layovers |
| days_left | Calendar days between checking the fare and departure |

The value being predicted is **price**, measured in Indian rupees.

## Dataset

The local dataset contains 300,153 historical flight listings with these columns:

    airline, source_city, departure_time, stops, arrival_time,
    destination_city, classs, duration, days_left, price

It covers:

- Six airlines
- Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, and Chennai
- Economy and Business class
- Journey durations from 0.83 to 49.83 hours
- Booking lead times from 1 to 49 days
- Recorded prices from ₹1,105 to ₹123,071

The pipeline creates an 80/20 random split:

- 240,122 training rows
- 60,031 testing rows

### Dataset limitation

The dataset does not contain collection dates, and its original source, license, and collection period are not documented. It is suitable for learning and portfolio demonstration, but its output should not be treated as a reliable estimate of current airfare.

## Preprocessing

Scikit-learn pipelines prepare the form data:

- **SimpleImputer** fills missing numerical values with the median and categorical values with the most frequent value.
- **OrdinalEncoder** converts airline, city, schedule, stops, and class labels into numbers.
- **MinMaxScaler** scales features into comparable numerical ranges.
- **ColumnTransformer** applies the appropriate operations to numerical and categorical columns.

The fitted preprocessing pipeline is saved as **Artifacts/Preprocessor.pkl**.

## Evaluation

The latest local MLflow evaluation recorded approximately:

| Metric | Result |
| --- | ---: |
| R² | 0.9849 |
| MAE | ₹1,084.85 |
| RMSE | ₹2,792.41 |

These results measure performance on a random test sample from the same historical dataset. They do not measure performance on current or future fares.

## Application flow

    Web form
        ↓
    Flask receives nine input values
        ↓
    Preprocessor.pkl transforms the values
        ↓
    Model.pkl predicts the fare
        ↓
    Flask renders the estimated price

## Project structure

    FareSense/
    ├── app.py                              # Flask application
    ├── templates/                          # Form and result pages
    ├── static/                             # Terminal theme and browser interactions
    ├── src/FlightPricePrediction/
    │   ├── components/
    │   │   ├── Data_ingestion.py
    │   │   ├── Data_transformation.py
    │   │   ├── Model_trainer.py
    │   │   └── Model_evaluation.py
    │   └── pipeline/
    │       ├── Training_pipeline.py
    │       └── Prediction_pipeline.py
    ├── Notebook_Experiments/               # Exploration and training notebooks
    ├── Artifacts/                           # Local data/model outputs
    ├── mlruns/                              # MLflow experiment metadata
    ├── dvc.yaml                             # DVC training stages
    ├── requirements.txt
    └── PROJECT_GUIDE.md                     # Detailed beginner explanation

For a deeper walkthrough, read [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Important artifact notice

The trained model, fitted preprocessor, and full CSV files are excluded from Git because they are generated or too large for standard GitHub storage.

The web application requires:

    Artifacts/Model.pkl
    Artifacts/Preprocessor.pkl

Training additionally requires:

    Notebook_Experiments/Data/data.csv

The repository includes a DVC pointer for the dataset, but no DVC remote storage is configured. Therefore, **dvc pull cannot restore the data on a fresh clone** until a remote is added.

## Local setup

### 1. Clone the repository

    git clone https://github.com/WaqassKhn/FareSense.git
    cd FareSense

### 2. Create and activate a virtual environment

Windows PowerShell:

    py -3.12 -m venv venv
    .\venv\Scripts\Activate.ps1

macOS or Linux:

    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

### 4. Add or generate the model artifacts

Place the trained model files under **Artifacts/**, or place **data.csv** under **Notebook_Experiments/Data/** and run:

    python src/FlightPricePrediction/pipeline/Training_pipeline.py

Training the Random Forest can take significant time and memory.

### 5. Start the Flask application

    python app.py

Open:

    http://127.0.0.1:8080

If port 8080 is occupied, stop the older Flask process before starting the application again.

## Docker

After the model artifacts are available locally:

    docker build -t faresense .
    docker run --rm -p 8080:8080 faresense

Then open **http://127.0.0.1:8080**.

## Technology stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Pandas | Tabular data loading and manipulation |
| NumPy | Numerical arrays |
| Scikit-learn | Preprocessing, training, and evaluation |
| Flask | Web server and prediction endpoint |
| HTML/CSS/JavaScript | User interface |
| MLflow | Experiment and metric tracking |
| DVC | Dataset and pipeline tracking |
| Pickle | Saving and loading the model and preprocessor |
| GitHub | Source-code version control |

## Recommended improvements

- Collect recent, licensed airfare data with observation and departure dates.
- Replace the random split with a chronological train/validation/test split.
- Add exact time, weekday, season, holiday, airport, and flight-number features.
- Compare Random Forest with XGBoost, LightGBM, and CatBoost.
- Replace arbitrary ordinal category relationships with a more suitable encoding strategy.
- Add automated tests and a non-empty CI workflow.
- Store model and dataset artifacts in configured DVC remote storage.
- Disable Flask debug mode for production deployment.
- Monitor prediction errors and data drift after deployment.

## License

See [LICENSE](LICENSE).
