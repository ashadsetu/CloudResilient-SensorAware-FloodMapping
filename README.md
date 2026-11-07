# 🌊 FloodSense — Cloud-Resilient Flood Mapping

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning research demo for flood mapping using **Sentinel-1 SAR radar** and **Sentinel-2 optical imagery**. The sensor-aware fusion model handles missing sensors and cloudy conditions while targeting a controlled missed-flood rate through conformal risk control.

>  Research demonstration only — not an operational flood-warning system.

** Live Demo:** https://cloudresilient-sensoraware-floodmapping-d.streamlit.app/

##  Key Features

- **Sensor-Aware Fusion:** Combines Sentinel-1 and Sentinel-2 imagery.
- **Flexible Sensor Modes:** Radar-only, optical-only, or both sensors.
- **Cloud Simulation:** Evaluates performance under different cloud levels.
- **Missed-Flood Control:** Configurable 5%, 10%, and 20% risk targets.
- **Visual Explanations:** Flood maps, probability maps, gate weights, and error maps.
- **Results Dashboard:** Model evaluation, cloud robustness, and ablation results.
- **Custom Uploads:** Supports GeoTIFF and NPZ inputs.

## 🛠️ Technologies

Python · PyTorch · Streamlit · NumPy · Pandas · Matplotlib · Plotly

**Dataset:** Sen1Floods11 (128 × 128 resized image chips).

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/CloudResilient-FloodMapping.git
cd CloudResilient-FloodMapping

python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## ☁️ Deployment

Hosted using [Streamlit Community Cloud](https://share.streamlit.io/).

- Branch: `main`
- Main file: `app.py`

## ⚠️ Limitations

- Cloud conditions are simulated, not real atmospheric observations.
- The missed-flood target is an expected chip-level risk control, not a guarantee for individual images.
- Performance may vary under regional and dataset shifts.
- No Bangladesh-specific validation has been performed.
- Uploaded imagery must follow the documented Sen1Floods11 conventions.

## 📄 License

MIT License. Sen1Floods11 data remain subject to their original license.
