"""
Streamlit App for Time Series Anomaly Detection

Features:
- Interactive visualization with Plotly
- Multiple detection methods from src/detectors
- Anomaly score visualization
- Upload custom files or use local dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, '.')

from aeon.segmentation import find_dominant_window_sizes
from src.utils.data_loader import load_dataset, sliding_window
from src.detectors import (
    ClusteringDetector,
    ClassificationDetector,
    RegressionDetector,
    ForecastingDetector,
    StatisticsDetector,
    NearestNeighborDetector
)

DETECTORS = {
    'Clustering (Best)': ClusteringDetector,
    'Nearest Neighbor': NearestNeighborDetector,
    'Forecasting': ForecastingDetector,
    'Classification': ClassificationDetector,
    'Statistics': StatisticsDetector,
    'Regression': RegressionDetector,
}

st.set_page_config(
    page_title="Time Series Anomaly Detection", 
    layout="wide",
    page_icon="AD"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #ced4da;
        color: #495057;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #0d6efd;
        color: #0d6efd;
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #6c757d;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #d63384 !important;
        border-bottom: 2px solid #d63384 !important;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .success-text { color: #198754; font-weight: 600; }
    .error-text { color: #dc3545; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_local_data():
    """Load dataset from notebooks folder."""
    zip_path = "notebooks/phase_1.zip"
    labels_path = "notebooks/labels.csv"
    
    if not os.path.exists(zip_path):
        return None, None, None, None
    
    return load_dataset(zip_path, labels_path)


def read_series_from_zip(file_name, file_list, locations, zf, folder_in_zip):
    """Read a specific series from the ZIP file."""
    internal_name = folder_in_zip + file_name
    
    with zf.open(internal_name) as f:
        data = pd.read_csv(f, header=None)
    data = np.array(data).flatten()
    
    # Extract metadata
    name = file_name.split('.')[0]
    splits = name.split('_')
    test_start = int(splits[-1])
    
    # Get anomaly location
    anomaly = (-1, -1)
    if name in locations.index:
        row = locations.loc[name]
        anomaly = (int(row["Start"]), int(row["End"]))
    
    return name, test_start, data, anomaly


def run_detection(data, test_start, detector_name):
    """Run anomaly detection using the selected method."""
    try:
        period = find_dominant_window_sizes(data[:test_start])
        window_size = int(period) if not isinstance(period, (list, np.ndarray)) else int(period[0])
    except:
        window_size = 50
    
    if window_size < 5:
        window_size = 50
    
    # Create sliding windows
    X = sliding_window(data, window_size)
    
    # Get detector
    DetectorClass = DETECTORS[detector_name]
    detector = DetectorClass(window_size=window_size)
    
    # Run detection
    if DetectorClass == StatisticsDetector:
        score, predicted = detector.detect(X, test_start, data)
    else:
        score, predicted = detector.detect(X, test_start)
    
    return score, predicted, window_size




st.title("Time Series Anomaly Detection")

# Sidebar
st.sidebar.title("Settings")

# Data source
data_mode = st.sidebar.radio("Data Source", ["Local Dataset", "Upload File"])

selected_file = None
is_upload = False
data = None
test_start = None
anomaly = None
file_id = None

if data_mode == "Local Dataset":
    result = load_local_data()
    if result[0] is not None:
        file_list, locations, zf, folder_in_zip = result
        
        # Filter to only Anomaly files
        anomaly_files = [f for f in file_list if "Anomaly" in f]
        selected_file = st.sidebar.selectbox("Select Time Series", anomaly_files)
        
        if selected_file:
            file_id, test_start, data, anomaly = read_series_from_zip(
                selected_file, file_list, locations, zf, folder_in_zip
            )
    else:
        st.sidebar.error("Dataset not found in notebooks/")
        st.sidebar.info("Expected: notebooks/phase_1.zip")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        data = df.iloc[:, 0].values.flatten()
        file_id = uploaded_file.name.split('.')[0]
        test_start = st.sidebar.slider(
            "Train/Test Split Point", 
            int(len(data) * 0.1), 
            int(len(data) * 0.9), 
            int(len(data) * 0.5)
        )
        anomaly = (-1, -1)
        is_upload = True

# Detection method selector
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Method")
detector_name = st.sidebar.selectbox(
    "Select Method",
    list(DETECTORS.keys()),
    label_visibility="collapsed"
)

# Run detection button
run_detection_btn = st.sidebar.button("Run Detection", type="primary", use_container_width=True)



if data is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Series", file_id)
    with col2:
        st.metric("Length", f"{len(data):,}")
    with col3:
        st.metric("Train/Test Split", test_start)
    with col4:
        if anomaly[0] > 0:
            st.metric("Known Anomaly", f"{anomaly[0]} - {anomaly[1]}")
        else:
            st.metric("Known Anomaly", "Unknown")
    
    # Store detection results in session state
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    
    if run_detection_btn:
        with st.spinner(f"Running {detector_name} detection..."):
            score, predicted, window_size = run_detection(data, test_start, detector_name)
            st.session_state.detection_results = {
                'score': score,
                'predicted': predicted,
                'window_size': window_size,
                'method': detector_name
            }
    
    # Create visualization
    st.markdown("### Time Series Visualization")
    
    results = st.session_state.detection_results
    
    if results:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Train segment
        fig.add_trace(go.Scatter(
            x=np.arange(test_start),
            y=data[:test_start],
            mode='lines',
            line=dict(color='#3498db', width=1),
            name='Train'
        ), row=1, col=1)
        
        # Test segment
        fig.add_trace(go.Scatter(
            x=np.arange(test_start, len(data)),
            y=data[test_start:],
            mode='lines',
            line=dict(color='#e67e22', width=1),
            name='Test'
        ), row=1, col=1)
        
        # Actual anomaly 
        if anomaly[0] > 0:
            fig.add_trace(go.Scatter(
                x=np.arange(anomaly[0], anomaly[1]),
                y=data[anomaly[0]:anomaly[1]],
                mode='lines',
                line=dict(color='#e74c3c', width=2),
                name='Actual Anomaly'
            ), row=1, col=1)
        
        # Predicted region
        pred = results['predicted']
        pred_start = max(0, pred - 30)
        pred_end = min(len(data), pred + 30)
        fig.add_vrect(
            x0=pred_start, x1=pred_end,
            fillcolor='#2ecc71', opacity=0.3,
            layer='below', line_width=0,
            row=1, col=1
        )
        
        # Anomaly scores - align x-axis with data
        score = results['score']
        window_size = results['window_size']
        score_viz = score.copy()
        
        # Hide training region scores
        score_viz[:test_start] = np.nan
        
        # X-axis: score indices shifted by window offset to align with data
        score_x = np.arange(len(score_viz)) + (window_size - 1)
        
        fig.add_trace(go.Scatter(
            x=score_x,
            y=score_viz,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#e67e22', width=1),
            fillcolor='rgba(230, 126, 34, 0.3)',
            name='Anomaly Score'
        ), row=2, col=1)
        
        # Predicted line
        fig.add_vline(x=pred, line_dash="solid", line_color="#2ecc71", line_width=2, row=2, col=1)
        
        # Actual center line
        if anomaly[0] > 0:
            actual_center = (anomaly[0] + anomaly[1]) // 2
            fig.add_vline(x=actual_center, line_dash="dash", line_color="#e74c3c", line_width=2, row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)', font=dict(color='black')),
            plot_bgcolor='#f5f5f5',
            paper_bgcolor='#e8e8e8',
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(color='black'),
            xaxis=dict(showgrid=True, gridcolor='#d0d0d0', linecolor='#aaa', mirror=True, tickfont=dict(color='black'), title_font=dict(color='black')),
            xaxis2=dict(showgrid=True, gridcolor='#d0d0d0', title="Time Index", linecolor='#aaa', mirror=True, tickfont=dict(color='black'), title_font=dict(color='black')),
            yaxis=dict(showgrid=True, gridcolor='#d0d0d0', title="Time Series", linecolor='#aaa', mirror=True, tickfont=dict(color='black'), title_font=dict(color='black')),
            yaxis2=dict(showgrid=True, gridcolor='#d0d0d0', title="Anomaly Score", linecolor='#aaa', mirror=True, tickfont=dict(color='black'), title_font=dict(color='black')),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results summary
        st.markdown("### Detection Results")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Method", results['method'].split()[0])
        with res_col2:
            st.metric("Window Size", results['window_size'])
        with res_col3:
            st.metric("Predicted Location", results['predicted'])
        with res_col4:
            if anomaly[0] > 0:
                actual_center = (anomaly[0] + anomaly[1]) // 2
                error = abs(results['predicted'] - actual_center)
                st.metric("Error", error)
            else:
                st.metric("Error", "N/A")
        
    else:
        # Simple single chart without detection
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=np.arange(test_start),
            y=data[:test_start],
            mode='lines',
            line=dict(color='#3498db', width=1),
            name='Train'
        ))
        
        fig.add_trace(go.Scatter(
            x=np.arange(test_start, len(data)),
            y=data[test_start:],
            mode='lines',
            line=dict(color='#e67e22', width=1),
            name='Test'
        ))
        
        if anomaly[0] > 0:
            fig.add_trace(go.Scatter(
                x=np.arange(anomaly[0], anomaly[1]),
                y=data[anomaly[0]:anomaly[1]],
                mode='lines',
                line=dict(color='#e74c3c', width=2),
                name='Known Anomaly'
            ))
        
        fig.update_layout(
            height=500,
            xaxis=dict(rangeslider=dict(visible=True), showgrid=True, gridcolor='#d0d0d0', linecolor='#aaa', mirror=True, tickfont=dict(color='black')),
            yaxis=dict(showgrid=True, gridcolor='#d0d0d0', linecolor='#aaa', mirror=True, tickfont=dict(color='black')),
            plot_bgcolor='#f5f5f5',
            paper_bgcolor='#e8e8e8',
            font=dict(color='black'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)', font=dict(color='black')),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("Click Run Detection in the sidebar to analyze this time series")

else:
    st.info("Select a time series from the sidebar to get started")
    
    # Show available methods
    st.markdown("### Available Detection Methods")
    
    methods_data = {
        'Method': ['Clustering', 'Nearest Neighbor', 'Forecasting', 'Classification', 'Statistics', 'Regression'],
        'Score': [2.27, 4.0, 5.5, 8.0, 9.7, 10.0],
        'Description': [
            'LOF, KMeans, kNN ensemble',
            'kNN distance-based',
            'Ridge regression + LOF + kNN',
            'Isolation Forest, SVM, LOF',
            'KDE, Mahalanobis, Z-score, IQR',
            'Gradient Boosting residuals'
        ]
    }
    st.table(pd.DataFrame(methods_data))
