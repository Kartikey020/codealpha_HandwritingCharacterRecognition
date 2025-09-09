from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data and print classification report.
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
