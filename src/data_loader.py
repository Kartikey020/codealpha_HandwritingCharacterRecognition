import numpy as np
from emnist import extract_training_samples, extract_test_samples

def load_emnist(split="balanced"):
    """
    Load EMNIST dataset (Balanced split).
    Classes: 47 (digits + uppercase/lowercase letters).
    """
    X_train, y_train = extract_training_samples(split)
    X_test, y_test = extract_test_samples(split)

    # Ensure correct format (float32)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")

    return (X_train, y_train), (X_test, y_test)
