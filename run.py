import os

import numpy as np

from helpers import create_csv_submission
from metrics import f_score
from models import LogisticRegression
from preprocessing import preprocess

# Preprocess the data
x_train, x_test, y_train, test_ids = preprocess()

# Initialize and train the model
print("Training model...")
model = LogisticRegression(metric=f_score)
model.train(X=x_train, y=y_train)

# Make predictions on the test set
predictions = model.predict(x_test)

# Convert to -1 and 1
predictions = np.where(predictions == 1, 1, -1)

# Create submission file
os.makedirs("data/submissions", exist_ok=True)
create_csv_submission(test_ids, predictions, "data/submissions/logistic_regression.csv")