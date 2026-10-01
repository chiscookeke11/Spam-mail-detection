# Spam Mail Detection

This project builds a binary classifier that identifies SMS messages as **spam** or **ham** (legitimate mail). The complete workflow is documented in the [Jupyter notebook](spam_mail_prediction_system.ipynb), from loading the data through evaluating a trained model.

## Project structure

```text
.
├── sample_data/
│   └── mail_data.csv                 # Labeled SMS messages
├── spam_mail_prediction_system.ipynb # Data preparation, training, and evaluation
└── README.md
```

The dataset contains 5,572 messages and two columns:

| Column | Description |
| --- | --- |
| `Category` | Original class label: `spam` or `ham`. |
| `Message` | The SMS text to classify. |

## Approach

The notebook takes the following steps:

1. **Import dependencies.** It uses pandas and NumPy for data handling, plus scikit-learn for splitting data, extracting features, training, and measuring accuracy.
2. **Load the dataset.** `sample_data/mail_data.csv` is read into a pandas DataFrame.
3. **Prepare the data.** Missing values are replaced with empty strings so every message can be processed as text.
4. **Encode the labels.** The original labels are converted to numeric targets: `spam` becomes `0` and `ham` becomes `1`.
5. **Separate inputs and targets.** The `Message` column is used as the feature set (`X`), while the encoded `Category` column is the target (`Y`).
6. **Create a holdout set.** The messages are divided into 80% training data and 20% test data with `random_state=2` for reproducible splits.
7. **Convert text to features.** A `TfidfVectorizer` lowercases text, removes English stop words, and creates a TF-IDF feature matrix. The vectorizer is fitted only on the training messages, then reused to transform test messages.
8. **Train the classifier.** A scikit-learn `LogisticRegression` model is fitted on the training feature matrix.
9. **Evaluate accuracy.** Predictions are scored against both the training and test targets using `accuracy_score`.

The notebook's saved run reports approximately **96.86% training accuracy** and **95.34% test accuracy**. These values can vary if the data, package versions, or model settings change.

## Getting started

### Prerequisites

Use Python 3 and install the packages required by the notebook:

```bash
python -m pip install pandas numpy scikit-learn jupyter
```

### Run the notebook

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Then open `spam_mail_prediction_system.ipynb` in the browser and run the cells from top to bottom. Keeping the current working directory at the repository root is important because the notebook loads the dataset from `./sample_data/mail_data.csv`.

### Run non-interactively (optional)

After installing the dependencies, execute all notebook cells and save the results with:

```bash
jupyter nbconvert --to notebook --execute --inplace spam_mail_prediction_system.ipynb
```

## Notes

- The model is trained for the dataset included in this repository; evaluate it on representative data before using it in a production setting.
- The notebook currently reports accuracy only. For an imbalanced spam-detection use case, precision, recall, and a confusion matrix are useful additional evaluation metrics.
