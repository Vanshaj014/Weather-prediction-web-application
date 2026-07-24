<div align="center">

# 🌦️ Weather Prediction Web Application

**Real-time weather insights and machine-learning forecasts for any city on Earth.**

A Django web app that combines live [OpenWeatherMap](https://openweathermap.org/) data with
scikit-learn models to predict the chance of rain tomorrow and forecast the hours ahead —
all behind a clean, weather-aware interface.

<br />

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.1-092E20?style=for-the-badge&logo=django&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Models](#-machine-learning-models)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running Tests](#-running-tests)
- [Evaluating Model Accuracy](#-evaluating-model-accuracy)
- [Configuration Reference](#-configuration-reference)
- [License](#-license)

---

## 🔍 Overview

This application lets a user search for any city and instantly see:

1. **Current conditions** pulled live from the OpenWeatherMap API.
2. **A rain forecast for tomorrow**, produced by a tuned Random Forest Classifier.
3. **A short-term outlook** of upcoming temperature and humidity, visualized in an interactive chart.

The interface adapts to the weather — the background changes to match the current sky
(clear, rain, snow, fog, thunderstorm, and more), giving each result a sense of place.

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🌡️ | **Current Weather** | Temperature, feels-like, min/max, humidity, cloud cover, wind speed & direction, pressure, and visibility. |
| 🌧️ | **Rain Prediction** | Probability that it rains tomorrow, from a `GridSearchCV`-tuned Random Forest Classifier. |
| 📈 | **Short-Term Forecast** | Upcoming temperature & humidity slots from the OpenWeatherMap 3-hourly forecast endpoint. |
| 📊 | **Interactive Chart** | Chart.js visualization of the forecast trend. |
| 🎨 | **Dynamic Background** | The page background reacts to real-time conditions. |
| ⚡ | **Response Caching** | API responses are cached per city (5 min) to cut latency and stay within rate limits. |
| 🛡️ | **Robust Error Handling** | Graceful handling of timeouts, network errors, unknown cities, and API rate limits. |

---

## 🧰 Tech Stack

- **Backend:** Python, Django 5.1
- **Machine Learning:** scikit-learn, pandas, NumPy, joblib
- **Data Source:** OpenWeatherMap API (`/weather` and `/forecast`)
- **Frontend:** HTML, CSS, Chart.js
- **Config:** `python-dotenv` for environment-based secrets

---

## 🤖 Machine Learning Models

Models are trained offline and loaded **once at server startup** (`forecast/apps.py`) to avoid
per-request disk I/O.

| Model | Algorithm | Key Metric |
|---|---|---|
| **Rain Prediction** | `RandomForestClassifier` + `GridSearchCV` + `class_weight='balanced'` | 93%+ accuracy |
| **Temperature Forecast** | `RandomForestRegressor` (autoregressive, window = 3) | ~1 °C MAE |
| **Humidity Forecast** | `RandomForestRegressor` (autoregressive, window = 3) | ~4.6% MAE |

> Training data lives in `weatherProject/forecast/data/weather.csv`. Trained model files are
> **not** committed — they are regenerated locally via `train_models.py` (see [Getting Started](#-getting-started)).

---

## 📂 Project Structure

```
Weather prediction/
├── analyze_project.py          # Project analysis / reporting script
├── evaluate_models.py          # Standalone model-accuracy evaluation
├── requirements.txt
├── .env                        # NOT in repo — create manually (see Getting Started)
├── .gitignore
└── weatherProject/
    ├── manage.py
    ├── forecast/               # Main Django app
    │   ├── data/
    │   │   └── weather.csv          # Historical training data
    │   ├── models/                  # Saved ML models (generated locally, not in repo)
    │   ├── static/                  # CSS, JS, images
    │   ├── templates/
    │   │   └── weather.html
    │   ├── apps.py                  # Loads ML models once at startup
    │   ├── train_models.py          # Offline training script
    │   ├── views.py                 # API calls, prediction, request handling
    │   └── tests.py                 # Unit & integration tests
    └── weatherProject/              # Django project settings
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- A free **API key** from [OpenWeatherMap](https://openweathermap.org/api)

### 1. Clone the repository

```bash
git clone https://github.com/Vanshaj014/Weather-prediction-web-application.git
cd "Weather-prediction-web-application"
```

### 2. Create and activate a virtual environment

```powershell
# Windows (PowerShell)
python -m venv myenv
myenv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

Create a file named `.env` in the project root:

```env
API_KEY="YOUR_OPENWEATHERMAP_API_KEY"
SECRET_KEY="YOUR_DJANGO_SECRET_KEY"
DEBUG=True
ALLOWED_HOSTS=*
```

Generate a Django secret key with:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 5. Train the ML models

The trained model files are not included in the repo. Run this once before starting the server:

```bash
python weatherProject/forecast/train_models.py
```

This saves the models into `weatherProject/forecast/models/`.

### 6. Run the development server

```bash
cd weatherProject
python manage.py runserver
```

Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser and search for a city. 🎉

---

## 🧪 Running Tests

The `forecast` app ships with unit and integration tests (API calls are mocked, so no
network or API key is required):

```bash
cd weatherProject
python manage.py test forecast
```

---

## 📏 Evaluating Model Accuracy

```bash
python evaluate_models.py
```

This runs all three models against a held-out 20% test set and reports **accuracy, precision,
recall, and F1** for the rain classifier, plus **MAE and R²** for the regression models.

---

## ⚙️ Configuration Reference

| Variable | Purpose | Example |
|---|---|---|
| `API_KEY` | OpenWeatherMap API key | `abc123...` |
| `SECRET_KEY` | Django cryptographic secret | *(generated, see above)* |
| `DEBUG` | Enable debug mode | `True` / `False` |
| `ALLOWED_HOSTS` | Comma-separated allowed hosts | `*` |

---

## 📄 License

Released under the **MIT License** — free to use, modify, and distribute.

<div align="center">
<sub>Built with ☕ and Django by <a href="https://github.com/Vanshaj014">Vanshaj</a></sub>
</div>
