# 👬 MLATE V3: Multi-Tissue Scaffold Prediction Platform

**MLATE V3** is a fully integrated, machine learning-powered platform for predicting, optimizing, and generating detailed fabrication procedures for 3D-(bio)printed scaffolds in tissue engineering. This app enables researchers to input a wide range of biomaterials, cell lines, and printing parameters, and receive optimized scaffold compositions along with step-by-step printing instructions generated via Google Gemini.

> 📄 *Rafieyan et al. (preprint, 2025). MLATE V3: A fully integrated Multi-Tissue, machine learning platform for prediction, optimization and generating procedures for fabricating 3D-(bio)printing scaffolds for tissue engineering*

---

## 🚀 Features

- 🔬 Predict scaffold quality based on printability and cell response
- 🧪 Optimize biomaterial concentrations, cell densities, and printing parameters using Optuna
- 🧠 Powered by two fine-tuned **CatBoostClassifier** models
- 📋 Automatically generates fabrication protocols with Gemini API
- 🔐 Enforces safe defaults and intelligent UI input validation
- 🧱 Uses a real-world, curated dataset of **2847 samples** across **multiple tissues and cell lines**

---

## 📂 Dataset

This project includes a publicly available dataset (`Dataset.xlsx`) containing:
- 123 biomaterials
- 175 cell lines
- 7 printing parameters
- Scaffold performance labels

The dataset is available in the [Files and Versions](https://huggingface.co/spaces/your-username/your-space-name/blob/main/Dataset.xlsx) tab of this Space.

You may also optionally add this to Hugging Face Datasets for broader access.

---

## ⚙️ How It Works

1. **Input**: User selects biomaterials, cell line, and printing parameters with min/max/step values
2. **Optimization**: Optuna runs 50 trials to maximize predicted scaffold quality (WSSQ)
3. **Prediction**:
    - Two CatBoost models are used to predict:
        - `Printability` (3-class)
        - `Cell Response` (5-class)
    - Probabilistic predictions are mapped to expected scores
4. **Scaffold Quality**: A weighted combination of printability and cell response
5. **Procedure Generation**: A Gemini API prompt generates custom step-by-step fabrication instructions

---

## 💻 Running Locally

Clone the repo and install dependencies:

```bash
git clone https://huggingface.co/spaces/your-username/MLATE-V3
cd MLATE-V3

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY=your_key_here  # or set in .env

# Run the app
streamlit run app.py
```

---

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

You are free to:
- Share and adapt the code
- Use the dataset for academic research

But:
- **Commercial use is prohibited**
- **Citation is required** (see below)

---

## 📚 Citation

If you use MLATE V3 or its dataset in your research, please cite:

> Rafieyan et al. (preprint, 2025).  
> *MLATE V3: A fully integrated Multi-Tissue, machine learning platform for prediction, optimization and generating procedures for fabricating 3D-(bio)printing scaffolds for tissue engineering*  
> *(Preprint link to be added after publication)*

BibTeX:
```bibtex
@article{rafieyan2025mlate,
  author  = {Rafieyan, Saeed and others},
  title   = {MLATE V3: A fully integrated Multi-Tissue, machine learning platform for prediction, optimization and generating procedures for fabricating 3D-(bio)printing scaffolds for tissue engineering},
  journal = {Preprint},
  year    = {2025}
}
```

---

## ⚠️ Disclaimer

This tool is intended for **research and academic use only**. While we strive for accuracy, the predictions and fabrication procedures are generated using machine learning and language models and may contain errors or inconsistencies. The authors are **not responsible for any unintended consequences** arising from use of this tool in experimental or clinical settings.

---

Developed by [Saeed Rafieyan](https://sraf.ir)
