# ArgosCam

### Intelligent Fall Detection and Monitoring System  
### Sistema Inteligente de Detecção e Monitoramento de Quedas

---

## 📖 Overview | Visão Geral

### English

**ArgosCam** is an artificial intelligence training and real-time monitoring system designed to detect human postures and identify critical events such as falls.

The project combines **computer vision**, **edge computing**, and **machine learning** to process live video streams, classify body positions, and automatically generate alerts when abnormal situations are detected.

ArgosCam operates locally for fast response and can communicate with an external backend API for alert persistence and mobile notification delivery.

---

### Português

O **ArgosCam** é um sistema de treinamento de inteligência artificial e monitoramento em tempo real projetado para detectar posturas humanas e identificar eventos críticos, como quedas.

O projeto combina **visão computacional**, **computação em borda (edge computing)** e **aprendizado de máquina** para processar transmissões de vídeo, classificar posições corporais e gerar alertas automáticos quando situações anormais são detectadas.

O ArgosCam opera localmente para resposta rápida e pode se comunicar com uma API externa para persistência de alertas e envio de notificações para dispositivos móveis.

---

## 🚀 Main Features | Funcionalidades Principais

### English

- Real-time human posture detection
- Fall event identification
- AI model training and validation
- Live camera or video file monitoring
- Circular frame buffering
- Automatic evidence video generation
- Local administrative visualization panel
- Edge processing for low-latency response
- Integration with external REST API

---

### Português

- Detecção de posturas humanas em tempo real
- Identificação automática de quedas
- Treinamento e validação do modelo de IA
- Monitoramento via câmera ao vivo ou arquivo de vídeo
- Buffer circular de frames
- Geração automática de vídeo de evidência
- Painel administrativo visual local
- Processamento em borda para baixa latência
- Integração com API REST externa

---

# 🛠️ Tech Stack

## Core Technologies

- **Python 3.10+**
- **YOLOv8 / YOLOv11 (Ultralytics)**
- **OpenCV**
- **NumPy**
- **FFmpeg**
- **Requests**

---

## 🔍 Component Responsibilities

### `argos_gate.py`

Main system engine responsible for:

- Loading the trained AI model
- Capturing video streams
- Performing real-time inference
- Stabilizing object IDs
- Detecting critical postures
- Managing frame buffering
- Rendering local monitoring interface

---

### `argos_api.py`

Integration layer responsible for:

- API authentication
- JWT token management
- Automatic re-authentication
- Alert metadata submission
- Evidence video upload

---

# 🧠 System Workflow

```text
Video Source
    ↓
Frame Capture (OpenCV)
    ↓
Pose Detection (YOLO)
    ↓
Fall Classification
    ↓
Circular Buffer Recovery
    ↓
Video Compression (FFmpeg)
    ↓
Alert Transmission (REST API)
```

---

# 🔧 Installation & Setup

## 1. Prerequisites

Install **FFmpeg** and ensure it is available in your system PATH.

Check installation:

```bash
ffmpeg -version
```

---

## 2. Install Dependencies

```bash
pip install numpy opencv-python ultralytics requests
```

---

## 3. Configuration

Edit the paths inside `argos_gate.py`:

```python
FONTE_ARQUIVO_LOCAL = r"C:\path\to\your_video.mp4"
PATH_MODELO = r"C:\path\to\your_model.pt"
```

Leave `FONTE_ARQUIVO_LOCAL` empty to use live camera/mobile streaming.

---

## 4. Run the System

```bash
python argos_gate.py
```

---


# 🔗 Related Projects

## API-ArgosCam

Backend REST API responsible for connecting ArgosCam to the mobile application.

Repository:

**https://github.com/juanpmf13/API-ArgosCam**

---

# 🚧 Project Status

Active development.

Academic and research project focused on intelligent fall detection and residential safety monitoring.

---

# 📌 Future Improvements

- Multi-person tracking
- Improved fall classification accuracy
- Mobile app integration
- Real-time push notifications
- Cloud event storage
- Performance optimization for embedded devices

---

# 📄 License

This project is currently developed for academic and research purposes.

Este projeto está sendo desenvolvido para fins acadêmicos e de pesquisa.