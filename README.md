# 📡 AI Drift Radar  
*A full-stack data drift, model degradation & MLOps assistant powered by Streamlit + Groq Llama 3.1.*

---

## 🚀 Overview  
AI Drift Radar helps ML teams detect **data drift**, diagnose **model degradation**, evaluate **metrics**, compute **embedding shift**, trigger **retraining decisions**, and generate detailed **reports**.  
It also includes a **multi-agent AI assistant** for contextual explanations.

---

## 🧩 Features  

### 🔍 Drift Detection  
- Numeric drift (**KS test**)  
- Categorical drift (**PSI**)  
- Timestamp decomposition  
- Automatic type correction (numeric string → number)  
- Drift visualization with smoothed curves  

### 📊 Model Monitoring  
- Upload `metrics.json`  
- Upload `.npy` embeddings  
- Embedding drift (mean cosine shift)  
- Auto-retrain evaluator  
- Webhook payload + cURL  

### 🧠 AI Drift Assistant  
- Memory aware  
- Domain inference  
- Multi-agent analysis  
  - Drift Analyst  
  - Data Quality Agent  
  - Business Impact Agent  
  - Retrain Advisor  
  - Ops / Integration Agent  
- Layman and Technical explanation modes  
- Blocks non-ML questions  

### 🧪 Sample Data Generator  
- 10 domains (ecommerce, finance, healthcare, etc.)  
- Seasonal pattern simulation  
- Built-in metrics.json generator  
- Embedding generator (.npy)

### 📥 Export Reports  
- TXT  
- DOCX  
- PDF  

---

## 📁 Project Structure  
```

.
├── app.py
├── requirements.txt
├── README.md

````

---

## ⚙️ Installation  

### 1. Install requirements  
```bash
pip install -r requirements.txt
````

### 2. Add your Groq API Key

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🧭 Usage Guide

### **1️⃣ Upload Data**

Go to **Upload & Analyze**
Upload:

* `reference_data.csv`
* `current_data.csv`

Outputs:

* Drift metrics
* Auto domain detection
* Drift visualizations

---

### **2️⃣ Model Monitor**

Upload:

* `metrics.json`
* `ref_embeddings.npy`
* `cur_embeddings.npy`

You will get:

* Performance degradation insights
* Embedding drift
* Auto-retrain decision
* Webhook JSON payload

---

### **3️⃣ AI Assistant**

Ask questions such as:

* “Explain drift in simple terms”
* “Why is my model degrading?”
* “What should I fix?”
* “Give retraining steps”

The assistant automatically runs multiple agents and produces:

* A summary
* Recommended actions
* A developer checklist

---

### **4️⃣ Export Reports**

Download your analysis as:

* TXT
* DOCX
* PDF

---

## 📬 Support

If you need help, improvements, deployment support, or customization — feel free to ask! 😊

```


If you want a **shorter version**, **more professional tone**, **project badges**, or **screenshots section**, I can generate that too.
```
