from src.data_loader import load_emnist
from src.preprocessing import preprocess_data
from src.model import build_cnn_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model, plot_sample

def main():
    print("===== HANDWRITTEN CHARACTER RECOGNITION (EMNIST) =====")

    # Load EMNIST Balanced dataset
    (X_train, y_train), (X_test, y_test) = load_emnist("balanced")
    print("Dataset Loaded:", X_train.shape, X_test.shape)

    # Preprocess
    X_train, X_test = preprocess_data(X_train, X_test)

    # Plot a sample
    plot_sample(X_train, y_train, 0)

    # Build CNN model
    model = build_cnn_model(num_classes=47)

    # Train
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=5)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, "emnist_cnn.h5")

if __name__ == "__main__":
    main()
