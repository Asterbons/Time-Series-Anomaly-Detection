# Time Series Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic project implementing **multiple anomaly detection techniques** for time series data. The goal is to identify anomalous segments in 30 different time series datasets using various machine learning approaches.

---

## Sample Result

<p align="center">
  <img src="assets/detection_example.png" alt="Anomaly Detection Example" width="900"/>
</p>

*Example showing perfect detection on `003_Anomaly_4375` - predicted location matches the actual anomaly exactly.*

---

## Problem Overview

Each time series contains:
- A **training segment** (normal data)
- A **test segment** containing exactly **one anomaly** of unknown size and shape

**Challenge**: Accurately locate the anomaly boundaries (start and end positions) in the test segment.

---

## Detection Methods & Benchmark

| Method | Description | Score | Best For |
|--------|-------------|-------|----------|
| **Clustering** | LOF, KMeans, kNN ensemble | **3.5** | General use |
| **Nearest Neighbor** | kNN distance-based | 5.5 | Local anomalies |
| **Forecasting** | Ridge regression + LOF + kNN | 7.4 | Periodic data |
| **Classification** | Isolation Forest, SVM, LOF | 63.3 | Outlier detection |
| **Statistics** | KDE, Mahalanobis, Z-score, IQR | 13.2 | Simple patterns |
| **Regression** | Gradient Boosting residuals | 11.9 | Trend anomalies |

*Lower score = better. Score is mean relative error (%) across 30 datasets.*

<p align="center">
  <img src="assets/methods_comparison.png" alt="Methods Comparison Chart" width="700"/>
</p>

---

## Project Structure

```
├── src/
│   ├── detectors/          # 6 detection method implementations
│   │   ├── clustering.py   # Best performing method
│   │   ├── regression.py
│   │   ├── forecasting.py
│   │   ├── statistics.py
│   │   ├── classification.py
│   │   └── nearest_neighbor.py
│   └── utils/              # Data loading & visualization
├── notebooks/
│   ├── primer_detector.ipynb
│   └── primer_visualization.ipynb
├── main.py                 # CLI runner
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/time-series-anomaly-detection.git
cd time-series-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Run detection (outputs to results/)
python main.py --method clustering --data-dir notebooks
```

### CLI Options

```bash
python main.py --method clustering    # Best method
python main.py --method all           # Run all methods
python main.py --show-plots           # Display visualizations
python main.py --help                 # See all options
```

---

## Key Techniques

- **Feature Engineering**: Rolling statistics, derivatives, lag features
- **Ensemble Methods**: Multiple algorithms combined with score normalization
- **Robust Preprocessing**: RobustScaler, window-based extraction, Gaussian smoothing

---

## Technologies

- **NumPy** & **Pandas** - Data manipulation
- **Scikit-learn** - ML algorithms (KMeans, IsolationForest, OneClassSVM, LOF)
- **Matplotlib** & **Seaborn** - Visualizations
- **aeon** - Time series analysis toolkit

---

## License

MIT License - see [LICENSE](LICENSE) for details.
