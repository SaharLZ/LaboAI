# 🧬 LaboAI

**LaboAI** is an intelligent digital laboratory platform designed for AI-assisted detection of multiple types of cancer. This integrated solution combines several specialized models, including:

* 🩺 Image classification for dermatological cases (skin cancer)
* 📊 Analysis of tabular data from medical exams (breast cancer and lung cancer)

The system is built on a robust modular architecture, including:

* 🖥️ A **Flask**-based backend for fast and secure AI model inference
* 🌐 An ergonomic web interface built with **Next.js**, offering healthcare professionals and researchers a smooth experience to interact with the prediction tools

---

## 🚀 Quick Installation & Startup

1. 📥 Clone this repository to your machine.

2. 🖥️ Open two separate terminals.

3. In the first terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

4. In the second terminal, start the backend:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

---

This setup allows you to easily develop, test, and use LaboAI while harnessing the power of AI models for early cancer detection. 🎯
