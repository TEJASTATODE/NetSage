<img width="1536" height="1024" alt="ChatGPT Image May 9, 2026, 03_19_20 PM" src="https://github.com/user-attachments/assets/48d9450e-6b68-4fbd-8331-c90e1e48ad47" />
# NetSage IDS — Realtime ML-based Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming-black?style=flat-square&logo=apachekafka)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Autoencoder-FF6F00?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A production-style, end-to-end distributed system that detects network intrusions in realtime using a hybrid Autoencoder + XGBoost ML pipeline, with SHAP explainability, dynamic thresholding, and a live React dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Why NetSage?](#why-netsage)
- [System Architecture](#system-architecture)
- [ML Inference Pipeline](#ml-inference-pipeline)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Autoencoder — Anomaly Detection](#2-autoencoder--anomaly-detection)
  - [Dynamic Threshold](#3-dynamic-threshold)
  - [XGBoost — Attack Classification](#4-xgboost--attack-classification)
  - [SHAP — Explainability](#5-shap--explainability)
- [FastAPI Backend Architecture](#fastapi-backend-architecture)
- [Frontend Dashboard](#frontend-dashboard)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Future Improvements](#future-improvements)

---

## Overview

NetSage IDS is a realtime, ML-powered Network Intrusion Detection System built as a full production-style distributed application.

It captures live network traffic, streams it through Apache Kafka, runs a three-stage hybrid ML inference pipeline (Autoencoder + XGBoost + SHAP), stores results in Redis and PostgreSQL, and delivers live threat alerts to a React dashboard via FastAPI WebSockets — all containerized with Docker.

This is not a notebook experiment. Every component is designed the way it would be in a real production security system.

---

## Why NetSage?

Traditional Intrusion Detection Systems rely on fixed, signature-based rules. They can only detect threats they have already seen. Novel attack patterns, zero-day exploits, and subtle behavioral anomalies slip past them entirely.

NetSage takes a fundamentally different approach:

- Instead of matching known signatures, it **learns what normal network behavior looks like** using an Autoencoder trained on legitimate traffic
- Anything that deviates meaningfully from that learned baseline is flagged as anomalous — even if that attack pattern has never been seen before
- A **dynamic threshold** adapts continuously to production traffic drift, avoiding alert fatigue from benign changes while staying sensitive to genuine threats
- XGBoost then **classifies the exact attack type** with calibrated confidence
- SHAP makes every decision **auditable and explainable** — no black boxes

---

## System Architecture

```
Live Network / PCAP
        │
        ▼
┌─────────────────────┐
│  Data Collection    │  Scapy — Packet Capture, Feature Extraction,
│  (Producer)         │  Serialize & Publish to Kafka Topic
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Apache Kafka      │  Topic: network_packets
│   (Broker)          │  Decouples ingestion from inference
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Consumer & ML      │  Consume → Preprocess → Autoencoder
│  Inference          │  → XGBoost → SHAP → Store Results
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           Storage Layer             │
│  Redis (Latest Results / Cache)     │
│  PostgreSQL (Historical Data)       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  REST API + WebSocket + JWT Auth
│                     │  Data Aggregation + Alert Triggering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  React Dashboard    │  Realtime Alerts, Traffic Analytics,
│                     │  Attack Distribution, SHAP Explanations
└─────────────────────┘
```

---

## ML Inference Pipeline

The core of NetSage is a sequential four-stage ML pipeline that runs on every consumed packet.

### 1. Data Preprocessing

Before any model sees the data, raw packet features go through:

- Handle missing values
- Encode categorical features (protocol_type, service, flag)
- Feature scaling and normalization
- Convert to numerical input vector of shape `(n_features,)`

### 2. Autoencoder — Anomaly Detection

The Autoencoder is a neural network trained **exclusively on normal (benign) network traffic**.

**Architecture:**

```
Input Layer → Encoder → Bottleneck (Latent Space) → Decoder → Reconstructed Output
```

**How it detects anomalies:**

During training, the model learns to compress and reconstruct normal traffic with minimal error. At inference time:

- Normal traffic → reconstruction is accurate → **low MSE**
- Anomalous / attack traffic → model cannot reconstruct it faithfully → **MSE spikes**

```
Reconstruction Error = MSE = mean((Input − Reconstructed)²)
```

This spike in reconstruction error is the **anomaly signal** that feeds into the next stage.

The Autoencoder never needs to have seen a specific attack pattern before. Any input that does not match its learned definition of "normal" is flagged automatically.

### 3. Dynamic Threshold

Rather than using a fixed MSE cutoff to decide whether a packet is anomalous, NetSage implements a **dynamic thresholding engine**.

The threshold is computed as a rolling statistic over recent reconstruction errors observed in production:

```
threshold = mean(recent_errors) + k * std(recent_errors)
```

Where `k` is a sensitivity multiplier tuned to the acceptable false positive rate.

**Why this matters:**

Network behavior changes naturally over time — new services, updated protocols, seasonal traffic patterns. A fixed threshold would either generate constant false alarms as normal traffic drifts, or miss real attacks as the baseline shifts. The dynamic threshold adapts continuously, keeping the system calibrated without manual retuning.

### 4. XGBoost — Attack Classification

Once a packet is flagged as anomalous, XGBoost takes over to classify **what kind of threat it is**.

**Features fed to XGBoost:**

- Original packet features
- Reconstruction error from the Autoencoder
- Statistical features derived from the traffic window
- Packet metadata

**Output classes:**

| Label | Attack Type |
|---|---|
| 0 | Normal |
| 1 | DDoS |
| 2 | Port Scan |
| 3 | Brute Force |
| 4 | SQL Injection |

XGBoost builds an ensemble of decision trees sequentially — each tree correcting the residual errors of the previous one. This gradient boosting mechanism produces **highly calibrated confidence scores**, not just binary labels.

The system outputs: `"Attack: DDoS — Confidence: 97%"`

### 5. SHAP — Explainability

Every single prediction is accompanied by a SHAP (SHapley Additive exPlanations) explanation.

SHAP assigns a contribution value to every input feature for every prediction, showing:

- Which features pushed the prediction toward **Attack**
- Which features pushed toward **Normal**
- The magnitude and direction of each feature's influence

**Top contributing features (example):**

```
src_bytes      ████████████  +0.38  (toward Attack)
dst_bytes      ██████        +0.21  (toward Attack)
duration       ███           +0.11  (toward Attack)
protocol_type  ██            -0.09  (toward Normal)
service        █             -0.04  (toward Normal)
```

This makes every alert **auditable**. Security analysts receive a reasoned explanation they can act on — not just an alarm they have to trust blindly.

---

## FastAPI Backend Architecture

```
API Gateway (FastAPI)
        │
        ├── JWT Auth — Secure all endpoints
        │
        ├── REST Endpoints
        │     GET /api/alerts
        │     GET /api/summary
        │     GET /api/packets
        │     GET /api/stats
        │     GET /api/shap/{id}
        │
        ├── WebSocket
        │     /ws/alerts  — Realtime alert stream
        │
        └── Business Logic / Services
              ├── Fetch from Redis (latest results)
              ├── Fetch from PostgreSQL (historical)
              ├── Data Aggregation
              └── Alert Triggering Logic
```

**Storage Layer:**

| Store | Role |
|---|---|
| Redis | Realtime cache — sub-millisecond reads for latest inference results |
| PostgreSQL | Persistent storage — full threat history, audit trails, analytics |

---

## Frontend Dashboard

Built with **React + Tailwind CSS**, the dashboard streams live data over WebSocket with no page refresh required.

| Panel | Description |
|---|---|
| Overview | Total packets, attacks detected, attack rate, active alerts |
| Traffic Over Time | Live chart of Normal vs Attack traffic |
| Live Alerts | Realtime feed with Source IP, Attack Type, Severity |
| Attack Distribution | Donut chart breaking down attack categories |
| SHAP Explanation | Bar chart of top contributing features per prediction |
| Packet Details | Raw packet metadata with model prediction and confidence |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Packet Capture | Python, Scapy, Pandas, NumPy |
| Streaming | Apache Kafka |
| ML — Anomaly Detection | TensorFlow / Keras (Autoencoder) |
| ML — Classification | XGBoost |
| Explainability | SHAP |
| Cache | Redis |
| Database | PostgreSQL |
| Backend | FastAPI |
| Frontend | React, Tailwind CSS, Axios, Recharts |
| Containerization | Docker, Docker Compose |

---

## Dataset

NetSage is trained and evaluated on the **UNSW-NB15** benchmark dataset — a widely used network intrusion dataset containing both normal traffic and nine categories of modern attack types including Fuzzers, DoS, Exploits, Reconnaissance, Backdoors, and more.

The dataset provides realistic, labeled network flow records suitable for training both the Autoencoder (on normal flows only) and the XGBoost classifier (on labeled flows).

---

## Results

| Metric | Value |
|---|---|
| Detection Accuracy | 98%+ |
| ROC-AUC Score | ~0.99 |
| Inference Latency | Sub-second per packet |
| Threshold Type | Dynamic (rolling statistical window) |
| Explainability | Per-prediction SHAP values |
| Deployment | Fully containerized via Docker |

---

## Project Structure

```
NetSage/
├── producer/
│   └── packet_producer.py        # Scapy capture + Kafka publisher
├── consumer/
│   └── ml_consumer.py            # Kafka consumer + ML inference pipeline
├── ml/
│   ├── autoencoder.py            # Autoencoder model definition + training
│   ├── xgboost_classifier.py     # XGBoost training + inference
│   ├── dynamic_threshold.py      # Rolling threshold computation
│   ├── shap_explainer.py         # SHAP explanation generation
│   └── preprocessing.py          # Feature engineering pipeline
├── backend/
│   ├── main.py                   # FastAPI app + routers
│   ├── routes/
│   │   ├── alerts.py
│   │   ├── stats.py
│   │   └── shap.py
│   ├── websocket/
│   │   └── alert_stream.py       # WebSocket live alert delivery
│   └── services/
│       ├── redis_service.py
│       └── postgres_service.py
├── frontend/
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx
│       │   └── components/       # Dashboard panels
│       └── package.json
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Python 3.10+
- Node.js 18+ (for local frontend development)

### 1. Clone the repository

```bash
git clone https://github.com/TEJASTATODE/NetSage.git
cd NetSage
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start all services

```bash
docker-compose up --build
```

This starts Kafka, Redis, PostgreSQL, the FastAPI backend, and the React frontend together.

### 4. Run the producer

```bash
cd producer
python packet_producer.py
```

### 5. Run the ML consumer

```bash
cd consumer
python ml_consumer.py
```

### 6. Access the dashboard

Open your browser at `http://localhost:5173`

---

## Environment Variables

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=network_packets

REDIS_HOST=localhost
REDIS_PORT=6379

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=netsage
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

JWT_SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256

DYNAMIC_THRESHOLD_SENSITIVITY=2.5
```

---

## Future Improvements

- [ ] Integrate GeoIP lookup to map source IPs to geographic locations on the dashboard
- [ ] Add model retraining pipeline triggered automatically on detection drift
- [ ] Implement alert severity scoring beyond binary Attack / Normal
- [ ] Add support for live PCAP file replay for offline testing
- [ ] Extend attack classification to cover all UNSW-NB15 categories
- [ ] Add Prometheus + Grafana monitoring for pipeline health metrics
- [ ] Role-based access control for the dashboard (Admin / Analyst views)

---

## License

This project is licensed under the MIT License.

---

> Built by [Tejas Tatode](https://github.com/TEJASTATODE) — designed end-to-end as a production-style distributed ML system at the intersection of AI, cybersecurity, and backend engineering.
