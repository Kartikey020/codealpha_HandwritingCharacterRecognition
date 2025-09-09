import numpy as np

def preprocess_data(X_train, X_test):
    """
    Normalize pixel values and reshape for CNN input.
    """
    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape (batch, height, width, channels)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, X_test
