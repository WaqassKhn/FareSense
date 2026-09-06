# Flight Fare Prediction: a beginner's complete guide

This guide explains the repository from first principles. You do not need to know Python, machine learning, Flask, DVC, MLflow, or Docker before starting.

> Important: this describes the project **as it currently exists**, including its strengths and problems. A fresh clone is not immediately runnable because the dataset and trained model files are absent and no DVC remote is configured. Read [What will and will not run](#what-will-and-will-not-run) before trying commands.

## 1. The project in one sentence

The project learns patterns from historical Indian flight records and uses them to estimate the price of a new flight entered through a web form.

The entire idea is:

1. Historical examples contain flight details and known prices.
2. Code converts those details into numbers.
3. Several algorithms learn relationships between the details and prices.
4. The algorithm with the best test score is saved.
5. A Flask website collects a new flight's details.
6. The saved preprocessing logic and model estimate its price.

This is **supervised machine learning** because training examples include the correct answer (`price`). It is **regression** because that answer is a continuous number, not a category.

## 2. AI, machine learning, and this project

Artificial intelligence (AI) is the broad idea of making computers perform tasks that appear intelligent. Machine learning (ML) is one part of AI. Instead of writing every pricing rule by hand, we show an algorithm many examples so it can estimate useful rules from data.

This is traditional tabular ML—not a chatbot, large language model, neural network, or generative-AI system. Its input is one structured row and its output is one number:

```text
predicted_price = f(airline, cities, times, stops, class, duration, days_left)
```

The learned function `f` is the trained model.

## 3. Essential vocabulary

| Term | Plain-English meaning | Example here |
|---|---|---|
| Dataset | A table of examples | Historical flight records |
| Row / sample | One example | One flight |
| Feature | Information used to predict | Airline or duration |
| Target / label | The answer to learn | `price` |
| Categorical feature | A named group | `Vistara`, `Delhi`, `Economy` |
| Numerical feature | A measured number | Duration or days left |
| Training | Learning from examples | `model.fit(...)` |
| Inference | Using a trained model | `model.predict(...)` |
| Preprocessing | Creating model-ready numbers | Encoding text and scaling numbers |
| Regression | Predicting a number | Predicting a fare |
| Artifact | A file created by a pipeline | `Model.pkl` |
| Pipeline | Ordered processing steps | Read → transform → train |
| Overfitting | Memorizing training rows instead of learning general patterns | A deep decision tree can do this |

## 4. The data

The checked-in notebook output describes 300,153 rows with these original columns:

| Column | Meaning | Used by production? |
|---|---|---|
| `Unnamed: 0` | An old saved row index, not a real feature | No |
| `airline` | Airline company | Yes |
| `flight` | Identifier such as `SG-8709` | No |
| `source_city` | Departure city | Yes |
| `departure_time` | Time-of-day group | Yes |
| `stops` | Zero, one, or two-or-more stops | Yes |
| `arrival_time` | Arrival time group | Yes |
| `destination_city` | Arrival city | Yes |
| `class` / `classs` | Economy or Business | Yes; spelled `classs` in production |
| `duration` | Journey duration in hours | Yes |
| `days_left` | Days between booking and departure | Yes |
| `price` | Known fare | Target only |

The notebook reports no missing values, a price range of 1,105–123,071, a median of 7,425, and a mean near 20,890. The higher mean suggests expensive fares pull the distribution upward.

The actual CSV is **not in Git**. `Notebook_Experiments/Data/data.csv.dvc` is only a DVC pointer with a hash and expected size (about 22.8 MB). DVC normally downloads the content from separate storage, but `.dvc/config` is empty, so no download location is provided.

Logs show the later production dataset had removed `flight` and renamed `class` to `classs`. Notebook experiments and the production pipeline therefore do not use exactly the same schema.

## 5. The two execution flows

Training is expensive and occasional; prediction is lightweight and happens for each form submission.

### Training

```text
data.csv
   ↓
Data_ingestion.py — copy and train/test split
   ↓
Data_transformation.py — encode text and scale numbers
   ↓
Model_trainer.py — train and compare regressors
   ↓
Model.pkl + Preprocessor.pkl
   ↓
Model_evaluation.py — calculate and log metrics
```

### Prediction

```text
Browser form
   ↓
Flask route in app.py
   ↓
CustomData → one-row pandas DataFrame
   ↓
Preprocessor.pkl.transform(...)
   ↓
Model.pkl.predict(...)
   ↓
result.html displays the estimate
```

Prediction must reuse the **same fitted preprocessor** used during training. Otherwise numeric values could mean something different to the model.

## 6. Training pipeline explained

The entry point is `src/FlightPricePrediction/pipeline/Training_pipeline.py`.

### Step 1: ingestion

`DataIngestion.initiate_data_ingestion()`:

1. Reads `Notebook_Experiments/Data/data.csv` with pandas.
2. Creates `Artifacts/`.
3. Saves `Raw_data.csv`.
4. Uses `train_test_split` with 80% for training and 20% for testing.
5. Uses `random_state=42` so the split repeats exactly.
6. Saves both portions.

A held-out test set estimates performance on unseen examples. Testing on training rows would reward memorization.

There is a code bug: ingestion returns `(test_path, train_path)`, but the training entry point assigns that to `(train_data_path, test_data_path)`. As a result, the transformation stage trains on the 20% file and evaluates on the 80% file. The sets remain separate, but their names and intended sizes are reversed.

### Step 2: transformation

`DataTransformation.get_data_transformation()` builds scikit-learn pipelines.

For numbers (`duration`, `days_left`):

1. `SimpleImputer(strategy='median')` replaces missing values with the training median.
2. `MinMaxScaler()` maps observed training values roughly to 0–1.

For categories:

1. `SimpleImputer(strategy='most_frequent')` fills a missing category with the most common one.
2. `OrdinalEncoder` replaces hard-coded names with numbers.
3. `MinMaxScaler` scales those numbers.

Encoding is needed because the models accept numbers. But airlines and cities are **nominal**: one is not naturally greater than another. Ordinal encoding invents an order. `OneHotEncoder` is often a safer default; ordinal encoding can still work with trees, but affects linear and distance-based models.

`initialize_data_transformation(...)` reads the splits, separates `price`, drops `price` and `Unnamed: 0`, fits preprocessing on training data, transforms test data, appends the targets to NumPy arrays, and saves the fitted preprocessor as `Artifacts/Preprocessor.pkl`.

`fit` means learn medians, category mappings, minimums, and maximums. `transform` means apply those learned values. Fitting on test data would cause **data leakage**.

### Step 3: train and choose a model

`ModelTrainer.initate_model_training(...)` separates each array:

```text
X = all columns except the last = features
y = last column = actual price
```

It compares:

- `LinearRegression`: one weighted linear formula; fast but limited.
- `Lasso`: linear regression that can shrink weights to zero.
- `Ridge`: linear regression that keeps weights small.
- `ElasticNet`: combines Lasso and Ridge penalties.
- `DecisionTreeRegressor`: learns if/then splits; flexible but prone to overfitting.
- `RandomForestRegressor`: averages many varied trees; usually more robust than one tree.

SVR and KNN are imported but commented out. Gradient Boosting is imported but absent from the candidate dictionary.

For each candidate, `utils.evaluate_model(...)` calls `fit`, predicts the test rows, and calculates R². The candidate with the highest R² is pickled to `Artifacts/Model.pkl`.

Tracked logs repeatedly select Random Forest. Local MLflow records contain roughly:

| Metric | Recorded range | Meaning |
|---|---:|---|
| R² | 0.9816–0.9817 | About 98.2% of target variance explained on that split |
| MAE | 1,413–1,416 | Average miss of about 1,415 price units |
| RMSE | 3,070–3,077 | A measure that punishes large errors more |

These historical scores do not guarantee future accuracy. Random Forest has no fixed `random_state`, so small run-to-run differences are expected.

### Step 4: evaluate and track

`ModelEvaluation` loads `Model.pkl` and calculates:

- **MAE:** average absolute error, in price units.
- **RMSE:** also in price units, but large misses count more heavily.
- **R²:** comparison with always predicting the mean. `1` is perfect, `0` is no better than that baseline, and negative values are possible.

MLflow records experiment metrics and artifacts so runs can be compared. The repository checks in local `mlruns` history.

The code sets a DagsHub **registry** URI but not the MLflow **tracking** URI. `get_tracking_uri()` can therefore remain the local file store. A working remote setup would also need authentication stored in environment variables or secrets, never committed credentials.

## 7. One web prediction, step by step

1. The browser sends an HTTP `POST` to `/`.
2. `app.py` reads fields from `request.form`.
3. It converts `duration` to `float` and `days_left` to `int`.
4. `CustomData` stores nine values.
5. `get_data_as_dataframe()` creates one pandas row with expected column names.
6. `PredictPipeline` loads the preprocessor and model pickles.
7. The preprocessor converts the row into training-compatible numbers.
8. The Random Forest averages predictions from its decision trees.
9. Flask rounds the first prediction to two decimals.
10. Jinja places it into `result.html`.

Flask is the bridge between browser and Python. Jinja expressions such as `{{ final_result }}` insert Python values into HTML.

## 8. Repository map

| Path | Purpose |
|---|---|
| `app.py` | Flask app and form-to-prediction connection |
| `requirements.txt` | pip packages; `-e .` installs this project in editable mode |
| `setup.py` | Local package metadata and dependency reader |
| `Dockerfile` | Python 3.8 container that installs packages and runs the app |
| `dvc.yaml` | Training command, dependencies, and expected outputs |
| `dvc.lock` | Hashes/sizes from a prior successful DVC run |
| `template.py` | One-time project scaffolding script; not used at runtime |
| `README.md` | Original brief overview |
| `LICENSE` | Source-project license terms |
| `components/Data_ingestion.py` | Reads and splits data |
| `components/Data_transformation.py` | Fits and saves preprocessing |
| `components/Model_trainer.py` | Compares models and saves the winner |
| `components/Model_evaluation.py` | Calculates and logs metrics |
| `pipeline/Training_pipeline.py` | Runs all training components |
| `pipeline/Prediction_pipeline.py` | Builds one-row input and predicts |
| `utils/utils.py` | Saves/loads pickles and compares candidates |
| `logger.py` | Writes timestamped files under `logs/` |
| `exception.py` | Adds filename and line number to errors |
| `templates/form.html` | Browser input form |
| `templates/result.html` | Prediction-result page |
| `static/styles.css` | Page styling |
| `Model_Training.ipynb` | Earlier interactive model experiments |
| `Exploratory_Data_Analysis.ipynb` | Empty (zero bytes) in this commit |
| `data.csv.dvc` | Pointer to the missing CSV |
| `Artifacts/` | Intended generated CSV, preprocessor, and model files |
| `mlruns/` | Historical local MLflow records |
| `logs/` | Historical execution logs |
| `.github/workflows/main.yaml` | Empty; no CI workflow is defined |

The component paths in the table are under `src/FlightPricePrediction/`.

## 9. Why each technology is used

| Technology | Job |
|---|---|
| Python | Main programming language |
| pandas | Reads CSV and handles DataFrames |
| NumPy | Numeric arrays passed to models |
| scikit-learn | Split, impute, encode, scale, model, and measure |
| Flask | Web server and form handling |
| HTML/CSS/Jinja | Browser UI, style, and dynamic values |
| pickle | Saves Python model objects |
| Jupyter | Interactive experimentation |
| Matplotlib/Seaborn | Experimental charts |
| MLflow | Experiment tracking |
| DVC | Data/artifact versioning and pipeline reproduction |
| Docker | Portable application environment |
| Git | Source-code history |

Git is ideal for code and text. Large datasets and 300 MB models make Git slow. DVC keeps small pointer/lock files in Git while large content lives in separate storage.

## 10. Notebook versus production

The notebook is historical exploration, not the exact application pipeline:

- It uses `flight`; production removes it.
- It uses `class`; production expects `classs`.
- It compares more models and reports Extra Trees as best (R² about 0.9848); production selects Random Forest.
- It incorrectly calls `fit_transform` on both training and test features. Test data should only call `transform`.
- Visible code refers to `Results` and `df_bk` without defining them. Stored output suggests hidden or later-deleted notebook state.
- It labels `log(RMSE)` as RMSLE, which is not the standard RMSLE formula.

This is why production logic should be repeatable, testable, and independent of notebook state.

## 11. What will and will not run

Present: source code, UI, DVC definitions, pointer file, historical notebook output, logs, and MLflow metrics.

Missing:

- `Notebook_Experiments/Data/data.csv`
- `Artifacts/Model.pkl`
- `Artifacts/Preprocessor.pkl`
- a configured DVC remote

Consequences:

- Flask can start, but submitting the form fails while opening missing pickles.
- Training fails while opening the missing CSV.
- `dvc pull` has nowhere configured to download from.
- `dvc repro` cannot rebuild outputs until data is supplied.

Hashes in `dvc.lock` are records, not the files themselves.

## 12. Intended Windows setup

The Dockerfile uses Python 3.8, so it is the closest match to the original 2023 environment.

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A virtual environment keeps this project's packages separate. Most requirements are unpinned, while MLflow is pinned to `2.2.2`, so installations can differ from the author's environment.

Supply a licensed, compatible CSV at:

```text
Notebook_Experiments/Data/data.csv
```

Current production code expects:

```text
Unnamed: 0, airline, source_city, departure_time, stops,
arrival_time, destination_city, classs, duration, days_left, price
```

The notebook's original data also has `flight` and spells `class` normally. A cleaning step must drop `flight` and rename `class` before current training code can use it.

Fix the path-order bug so ingestion and the entry point consistently exchange `(train_path, test_path)`. Then train with either:

```powershell
python -m src.FlightPricePrediction.pipeline.Training_pipeline
```

or:

```powershell
dvc repro
```

After all artifacts exist, run:

```powershell
python app.py
```

Open `http://127.0.0.1:8080/`.

Docker can be used after artifacts exist:

```powershell
docker build -t flight-fare-prediction .
docker run --rm -p 8080:8080 flight-fare-prediction
```

The original README maps port 5000, but `app.py` listens on 8080; use `8080:8080` unless the code changes.

## 13. Limitations and risks

1. Fresh-clone reproduction is broken: no data/artifacts or DVC remote.
2. Train and test paths are reversed.
3. Notebook and production schemas differ.
4. There are no automated tests.
5. Dependencies are largely unpinned.
6. Ordinal encoding invents order among names.
7. Unknown categories can make `OrdinalEncoder` fail.
8. Model selection repeatedly uses the test set; validation or cross-validation should select models, leaving final test data untouched.
9. No hyperparameter tuning or cross-validation exists.
10. Random Forest is unseeded.
11. Negative duration or invalid day counts are not rejected server-side.
12. Pickle files can execute code while loading; use only trusted artifacts.
13. `debug=True` is useful locally but unsafe on a public server.
14. The result page shows `$`; routes and values suggest rupees, but currency is undocumented.
15. Historical predictions are not live quotes; demand, inventory, taxes, and market changes may be missing.
16. A single answer gives no uncertainty range.
17. Remote MLflow tracking is incomplete.
18. The CI workflow and EDA notebook are empty.

## 14. A stronger ML workflow

1. Define currency, market, prediction date, and exact question.
2. Record data source, license, collection time, and meanings.
3. Perform reproducible exploratory analysis.
4. Create train/validation/test sets or use cross-validation.
5. Put schema cleaning and preprocessing in one fitted pipeline.
6. Establish a simple baseline, such as the training median.
7. Compare models with validation data and suitable metrics.
8. Tune only on training/validation data.
9. Evaluate once on untouched test data.
10. Inspect errors by airline, class, route, and price band.
11. Version code, data, schema, model, metrics, and environment.
12. Add unit, pipeline, and HTTP tests.
13. Deploy with validation, monitoring, and retraining rules.

## 15. Learning path and exercises

### Start with Python and tables

Read `app.py`, `Prediction_pipeline.py`, then `Data_ingestion.py`. Practice `df.head()`, `df.shape`, `df.dtypes`, and `df.describe()` on a small CSV.

### Learn preprocessing

Read `Data_transformation.py`. On a tiny table, compare missing-value imputation, ordinal encoding, one-hot encoding, scaling, and `fit` versus `transform`.

### Learn models and metrics

Read `Model_trainer.py` and `utils.py`. Compare Linear Regression, one Decision Tree, and Random Forest using MAE, RMSE, R², training time, and file size.

### Learn ML engineering

Trace `form.html` → `app.py` → `Prediction_pipeline.py` → `result.html`, then inspect logging, exceptions, DVC, and MLflow. An ML product needs reliable data, repeatable preprocessing, artifacts, tests, deployment, and monitoring—not just an algorithm.

### Improvement exercises, easiest first

1. Fix return order and test the 80/20 counts.
2. Add HTML and server-side value validation.
3. Document and display the right currency.
4. Seed random models.
5. Handle unknown categories.
6. Compare one-hot and ordinal encoding with cross-validation.
7. Add a baseline model.
8. Save preprocessing and model as one scikit-learn `Pipeline`.
9. Test GET, valid POST, and invalid POST.
10. Configure DVC storage and reproduce from a clean clone.
11. Pin dependencies and add CI.
12. Add honest uncertainty information.

## 16. Check your understanding

1. Why is this regression rather than classification?
2. Why must test rows stay outside training?
3. What does preprocessing learn during `fit`?
4. Why must prediction reuse that preprocessor?
5. Why can ordinal encoding misrepresent airlines?
6. Why can a forest outperform one tree?
7. How do MAE and RMSE differ?
8. Why does high historical R² not guarantee a good future quote?
9. How do Git, DVC, and MLflow differ?
10. Which two pickle files must exist before form submission works?
11. What creates the 20%/80% reversal?
12. Why is hidden notebook state hard to reproduce?

## 17. Final mental model

```text
DATA supplies examples
  → PREPROCESSING creates numeric representations
  → TRAINING estimates patterns
  → EVALUATION checks unseen examples
  → SERIALIZATION saves fitted behavior
  → FLASK collects new input
  → INFERENCE reuses preprocessing and model
  → MONITORING should reveal when reality changes
```

The model is only one piece. Reliable ML depends on keeping the meaning of the data consistent across every arrow.

## 18. Evidence and confidence

This guide was derived from the checked-out source at commit `e571ca3` (2023-12-04): Python modules, HTML/CSS, Dockerfile, DVC files, stored notebook code/output, logs, and local MLflow records.

- **High confidence:** current code flow, paths, absent tracked data/artifacts, path-order bug, feature lists, and historical metrics; these are directly visible in repository files.
- **Medium confidence:** intended currency is Indian rupees; routes and values strongly suggest it, but the repository never states it and HTML displays `$`.
- **Unknown:** original data source/license, DVC storage location, public deployment status, and accuracy on newer fares.
