## NLP-NER

A simple Named Entity Recognition (NER) project leveraging pretrained transformers and a user-friendly UI.

---

### Features

* Interactive UI for testing NER models
* Configurable hyperparameters via `constants.py`
* Support for training, evaluation, and inference
* Integration with Weights & Biases and Hugging Face

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/nlp-ner.git
   cd nlp-ner
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launch the UI

Run the following command to start the web interface:

```bash
python UI/app.py
```

Open your browser and navigate to `http://localhost:8000` to interact with the NER model.

### 2. Training or Fine-tuning Models

By default, hyperparameters are defined in [`constants.py`](./constants.py). To adjust them and retrain:

1. **Create a new Git branch**

   ```bash
   git checkout -b tune-hyperparams
   ```
2. **Edit hyperparameters**

   * Open `constants.py` and modify values under the `HYPERPARAMETERS` section.
3. **Run training in a Kaggle Notebook**

   * Clone your branch:

     ```bash
     git clone https://github.com/auphong2707/nlp-ner.git
     cd nlp-ner
     git checkout tune-hyperparams
     ```
   * Configure environment variables:

     ```bash
     export WANDB_API_KEY=<your_wandb_key>
     export HUGGINGFACE_API_KEY=<your_huggingface_key>
     ```
   * Execute the appropriate training script:

     ```bash
     python main_<model_name>.py
     ```

---

## Contributing

Feel free to open issues or submit pull requests for new features or bug fixes. Please follow the existing code style and write tests for new functionality.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
