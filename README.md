# Context-Aware Sarcasm Detection using BERT

This project implements a context-aware sarcasm detection system using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to accurately classify a comment as either "Offensive" (sarcastic) or "Non-Offensive" (genuine) by understanding its surrounding context.

## 🚀 Project Overview

Sarcasm detection is a challenging task in Natural Language Processing (NLP) because the literal meaning of words often contradicts the intended meaning. Traditional sentiment analysis struggles with this, as it typically evaluates text in isolation. Our approach addresses this by leveraging the powerful contextual understanding capabilities of a Transformer model.

The model takes both a user's comment and its relevant context as input. By presenting these two pieces of text together, separated by a special `[SEP]` token, BERT learns to identify semantic inconsistencies or ironic contrasts that signal sarcasm.

## ✨ Features

-   **Context-Aware Analysis:** Processes both comment and context for nuanced understanding.
-   **BERT-based Model:** Utilizes a fine-tuned `bert-base-uncased` model for state-of-the-art language representation.
-   **High Accuracy:** Achieves excellent performance on the validation set, demonstrating robust sarcasm detection.
-   **Interactive Inference Script:** A Python script to easily test the trained model with new comment-context pairs.
-   **Detailed Report:** A comprehensive LaTeX report detailing methodology, results, and analysis.

## 📁 Repository Structure

. ├── apr_proj.ipynb # Jupyter Notebook for data loading, training, and evaluation ├── context_dependent_comments_dataset_5000.csv # The dataset used for training ├── predict.py # Python script for interactive model inference ├── README.md # This README file ├── sarcasm_bert_model/ # Directory where the trained BERT model and tokenizer are saved (ignored by Git) ├── class_distribution.png # Plot of class distribution in the dataset ├── model_loss_curve.png # Plot of training and validation loss over epochs ├── confusion_matrix.png # Confusion matrix illustrating model performance └── apr_project_report.pdf # (Optional) Compiled LaTeX report


## 📋 Authors

This project was developed by the following team members:

* (2201AI02, Akash Sinha)
* (2201AI04, Ammar Ahmad)
* (2201AI51, Mridul Kumar)
* (2201AI54, Aman Vaibhav Jha)
* (2201CS07, Aditya Chauhan)
* (2201CS08, Aditya Yadav)
* (2201CS11, Akhand Singh)
* (2201CS15, Anchal Dubey)
* (2201CS16, Animesh Tripathy)
* (2201CS45, Mayur Borse)
* (2201CS54, Prakhar Shukla)
* (2201CS94, Anirudh D Bhat)

## 🛠️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/akhandsinghjr/Adv-Pattern-Recognition---Sarcasm-Detection.git](https://github.com/akhandsinghjr/Adv-Pattern-Recognition---Sarcasm-Detection.git)
    cd Adv-Pattern-Recognition---Sarcasm-Detection
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install pandas numpy torch scikit-learn transformers matplotlib seaborn
    ```
    *(Note: `torch` installation might require specific commands based on your CUDA version if you plan to use a GPU. Refer to the official PyTorch website for details.)*

3.  **Download the Dataset:**
    Ensure `context_dependent_comments_dataset_5000.csv` is present in the root directory of the project.

## 🚀 Usage

### 1. Training the Model (Jupyter Notebook)

The `apr_proj.ipynb` notebook contains the full pipeline for data loading, preprocessing, BERT model fine-tuning, and evaluation.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open `apr_proj.ipynb`:** Run all cells in the notebook sequentially.
    * The notebook will perform data transformation, split data, tokenize text, and train the BERT model.
    * It will generate `class_distribution.png`, `model_loss_curve.png`, and `confusion_matrix.png` in the project directory.
    * Crucially, it will save the trained model and tokenizer to a new directory named `sarcasm_bert_model/` in your project root. This directory is essential for the inference script and is configured to be ignored by Git (using `.gitignore`) due to its large size.

### 2. Using the Inference Script (`predict.py`)

After running the notebook and saving the model, you can use `predict.py` to interactively test the model.

1.  **Ensure Model is Saved:** Make sure the `sarcasm_bert_model/` directory exists from running the notebook.
2.  **Run the Script:**
    ```bash
    python predict.py
    ```
3.  **Interact:** The script will prompt you to enter a comment and its context. Type `quit` for either input to exit the script.

    **Example Interaction:**
    ```
    --- Sarcasm Detection Inference Script ---
    Loading model from ./sarcasm_bert_model...
    Model loaded successfully on cuda. # or cpu
    
    Enter a comment and its context. Type 'quit' for either to exit.
    
    Enter Comment: Wow, that's a very creative solution.
    Enter Context: A team member proposes a nonsensical idea that ignores all constraints.
    
    --- Analyzing ---
    Comment (C): Wow, that's a very creative solution.
    Context (X): A team member proposes a nonsensical idea that ignores all constraints.
    --- Model Prediction: Offensive (Sarcastic) ---
    
    Enter Comment: Wow, that's a very creative solution.
    Enter Context: A designer presents a genuinely novel and effective idea.
    
    --- Analyzing ---
    Comment (C): Wow, that's a very creative solution.
    Context (X): A designer presents a genuinely novel and effective idea.
    --- Model Prediction: Non-Offensive (Genuine) ---
    ```

## 📈 Results

The fine-tuned BERT model achieved outstanding performance on the validation set:

* **Average Training Loss (Epoch 3):** 0.0026
* **Average Validation Loss (Epoch 3):** 0.0016
* **Validation Accuracy:** 100.00%
* **Precision, Recall, F1-score:** 1.00 for both "Offensive" and "Non-Offensive" classes.

These results indicate that the model effectively learned to distinguish sarcasm from genuine statements based on the provided context within the training data.

## 📄 Report

A detailed project report, `apr_project_report.pdf`, (if compiled from the provided LaTeX source) contains a comprehensive explanatio
