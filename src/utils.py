import matplotlib.pyplot as plt

def plot_sample(X, y, idx):
    plt.imshow(X[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y[idx]}")
    plt.show()

def save_model(model, filename="mnist_cnn.h5"):
    model.save(filename)
    print(f"[INFO] Model saved as {filename}")
