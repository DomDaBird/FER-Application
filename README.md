# Facial Emotion Recognition (FER) -- EfficientNetB0

A student project for IU course **Artificial Intelligence
(DLBDSEAIS02)** (Task 3: Emotion Detection in Images).

The system detects facial emotions in images using transfer learning
with **EfficientNetB0** and classifies images into:

-   Angry
-   Fear
-   Happy
-   Sad
-   Surprise

This repository is written so that non-developers (e.g., a marketing
team) can run a quick demo and understand what the system does.

------------------------------------------------------------------------

## 1) Quick Demo (Streamlit)

### 1. Install

``` bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Python version

This project is tested with **Python 3.11** on Windows.
Using Python 3.12/3.13 may cause TensorFlow installation issues on some systems.

### 2. Run the app

``` bash
streamlit run app_streamlit.py
```

### 3. Test with demo images

A small set of demo images is provided in:

`demo_data/`

Use the Streamlit uploader and try a few images from each emotion
folder.

------------------------------------------------------------------------

## 2) Project Structure (Short)

Core scripts:

-   `train.py` -- model training pipeline
-   `evaluate.py` -- evaluation on the test set (metrics + confusion
    matrix)
-   `infer_single.py` -- inference for a single image via command line
-   `app_streamlit.py` -- Streamlit web demo
-   `data.py` -- tf.data input pipeline (train/val/test)
-   `model.py` -- EfficientNetB0 transfer learning model
-   `config.py` -- configuration (paths, hyperparameters)

Outputs:

-   `reports/` -- evaluation results (e.g., confusion matrix,
    classification report)

Documentation:

-   `docs/` -- installation and usage guides

------------------------------------------------------------------------

## 3) Training & Evaluation (Developer Workflow)

### Dataset folder structure

dataset/ train/ angry/ fear/ happy/ sad/ surprise/ val/ angry/ fear/
happy/ sad/ surprise/ test/ angry/ fear/ happy/ sad/ surprise/

Note: The full Kaggle dataset is not included in this repository.

### Train

``` bash
python train.py
```

### Evaluate

``` bash
python evaluate.py
```

### Inference (single image)

``` bash
python infer_single.py --image "path/to/image.jpg"
```

------------------------------------------------------------------------

## 4) Documentation

For detailed instructions, see:

-   `docs/INSTALLATION.md`
-   `docs/USAGE_STREAMLIT.md`
-   `docs/TRAINING.md`
-   `docs/EVALUATION.md`
-   `docs/ARCHITECTURE.md`

------------------------------------------------------------------------

## 5) License / Disclaimer

This is a university project. The model predictions are for
demonstration and learning purposes.
