import numpy as np
from PIL import Image
import os

# ------------------- Dataset Loader -------------------
def load_cats_vs_dogs_dataset(dataset_path, target_size=(64,64)):
    images = []
    labels = []

    class_names = ["cats", "dogs"]  # 0 = cat, 1 = dog

    for label, folder_name in enumerate(class_names):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert("L")  # grayscale
                img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32)/255.0
                images.append(img_array.flatten())
                labels.append(label)
            except:
                continue
    return np.array(images), np.array(labels)

# ------------------- Train/Val/Test Split -------------------
def train_val_test_split(features, labels, val_ratio=0.1, test_ratio=0.1):
    n_samples = features.shape[0]
    indices = np.random.permutation(n_samples)

    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    return (
        features[train_idx], labels[train_idx],
        features[val_idx], labels[val_idx],
        features[test_idx], labels[test_idx]
    )

# ------------------- Neural Network Class -------------------
class NN:
    def __init__(self, layer_sizes, hidden_activation="relu"):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.hidden_activation = hidden_activation
        self.parameters = self.initialize_parameters()

    # He initialization for ReLU
    def initialize_parameters(self):
        params = {}
        for l in range(1, self.num_layers):
            input_dim = self.layer_sizes[l-1]
            output_dim = self.layer_sizes[l]
            if self.hidden_activation=="relu" and l != self.num_layers-1:
                params[f"W{l}"] = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
            else:
                params[f"W{l}"] = np.random.randn(output_dim, input_dim) * 0.01
            params[f"b{l}"] = np.zeros((output_dim,1))
        return params

    # Activations
    def output_activation(self, x):
        return self.sigmoid(x)

    def output_activation_backward(self, AL, Y):
        return AL - Y  # derivative of sigmoid + binary cross-entropy

    def relu(self,x):
        return np.maximum(0,x)
    def relu_deriv(self,x):
        return (x>0).astype(float)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def hidden_activation_forward(self,x):
        if self.hidden_activation=="relu":
            return self.relu(x)
        else:
            return self.sigmoid(x)

    def hidden_activation_backward(self,x):
        if self.hidden_activation=="relu":
            return self.relu_deriv(x)
        else:
            return self.sigmoid(x)*(1-self.sigmoid(x))
        
        

    # Feedforward
    def feed_forward(self,input_batch):
        cache = {}
        A = input_batch.T
        cache["A0"] = A
        for l in range(1, self.num_layers):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(W,A) + b
            if l == self.num_layers-1:
                A = self.output_activation(Z)
            else:
                A = self.hidden_activation_forward(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
        return cache

    # Loss
    def compute_loss(self, AL, Y):
        m = AL.shape[1]
        Y = Y.reshape(1,-1)
        eps = 1e-8
        loss = -1/m * np.sum(Y*np.log(AL+eps) + (1-Y)*np.log(1-AL+eps))
        return loss

    # Backprop
    def backprop(self,cache,Y):
        grads = {}
        m = cache["A0"].shape[1]
        Y = Y.reshape(1,-1)

        AL = cache[f"A{self.num_layers-1}"]
        dZ = self.output_activation_backward(AL, Y)
        grads[f"dW{self.num_layers-1}"] = 1/m * np.dot(dZ,cache[f"A{self.num_layers-2}"].T)
        grads[f"db{self.num_layers-1}"] = 1/m * np.sum(dZ,axis=1,keepdims=True)

        for l in range(self.num_layers-2,0,-1):
            W_next = self.parameters[f"W{l+1}"]
            dA = np.dot(W_next.T,dZ)
            Z_curr = cache[f"Z{l}"]
            dZ = dA * self.hidden_activation_backward(Z_curr)
            grads[f"dW{l}"] = 1/m * np.dot(dZ,cache[f"A{l-1}"].T)
            grads[f"db{l}"] = 1/m * np.sum(dZ,axis=1,keepdims=True)
        return grads

    # Update parameters
    def update_parameters(self,grads,lr):
        for l in range(1,self.num_layers):
            self.parameters[f"W{l}"] -= lr*grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= lr*grads[f"db{l}"]

    # Train
    def train(self, X_train, y_train, X_val=None, y_val=None,epochs=20, lr=0.01, batch_size=32, print_every=1):

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                cache = self.feed_forward(X_batch)
                AL = cache[f"A{self.num_layers-1}"]

                loss = self.compute_loss(AL, y_batch)
                epoch_loss += loss * X_batch.shape[0]

                grads = self.backprop(cache, y_batch)
                self.update_parameters(grads, lr)

            # ---- Epoch Metrics ----
            epoch_loss /= n
            self.train_losses.append(epoch_loss)

            train_preds = self.predict_batch(X_train)
            train_acc = np.mean(train_preds == y_train)
            self.train_accs.append(train_acc)

            if X_val is not None:
                val_cache = self.feed_forward(X_val)
                val_loss = self.compute_loss(
                    val_cache[f"A{self.num_layers-1}"], y_val
                )
                self.val_losses.append(val_loss)

                val_preds = self.predict_batch(X_val)
                val_acc = np.mean(val_preds == y_val)
                self.val_accs.append(val_acc)

            if epoch % print_every == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}/{epochs} | "
                        f"loss={epoch_loss:.4f} | "
                        f"train_acc={train_acc:.4f} | "
                        f"val_acc={val_acc:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs} | loss={epoch_loss:.4f}")

    # Predict single sample
    def predict_image(self, image_path, target_size=(64,64), mean=None, std=None):

        # Load image
        img = Image.open(image_path).convert("L")
        img = img.resize(target_size)

        # Same preprocessing as training
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)
        
        # Apply normalization if provided (same as training data)
        if mean is not None and std is not None:
            img_flat = (img_flat - mean) / (std + 1e-8)

        # Feedforward
        cache = self.feed_forward(img_flat)
        prob = cache[f"A{self.num_layers-1}"].reshape(-1)[0]

        # Decision
        if prob >= 0.5:
            print(f"Prediction: Dog (probability = {prob:.4f})")
            return 1
        else:
            print(f"Prediction: Cat (probability = {prob:.4f})")
            return 0

    # Predict batch
    def predict_batch(self,X):
        cache = self.feed_forward(X)
        AL = cache[f"A{self.num_layers-1}"]
        preds = (AL.reshape(-1)>=0.5).astype(int)
        return preds

    # Predict batch from folder
    def predict_folder(self, folder_path, target_size=(64,64), mean=None, std=None):
        """
        Predict all images in a folder
        Args:
            folder_path: Path to folder containing images
            target_size: Size to resize images to
            mean: Mean for normalization (from training data)
            std: Std for normalization (from training data)
        """
        images = []
        filenames = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert("L")
                img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_flat = img_array.flatten()
                images.append(img_flat)
                filenames.append(filename)
            except:
                continue
        
        if len(images) == 0:
            print("No valid images found in folder")
            return [], []
        
        # Convert to numpy array
        images = np.array(images)
        
        # Apply normalization if provided
        if mean is not None and std is not None:
            images = (images - mean) / (std + 1e-8)
        
        # Predict
        cache = self.feed_forward(images)
        AL = cache[f"A{self.num_layers-1}"]
        probs = AL.reshape(-1)
        preds = (probs >= 0.5).astype(int)
        
        # Print results
        print(f"\nPredictions for {len(images)} images in folder:")
        print("-" * 60)
        for i, (filename, pred, prob) in enumerate(zip(filenames, preds, probs)):
            class_name = "Dog" if pred == 1 else "Cat"
            print(f"{i+1}. {filename}: {class_name} (probability: {prob:.4f})")
        
        # Summary
        num_cats = np.sum(preds == 0)
        num_dogs = np.sum(preds == 1)
        print("-" * 60)
        print(f"Summary: {num_cats} cats, {num_dogs} dogs")
        
        return preds, filenames


from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(64 ,64,3))
vgg.trainable = False

def extract_features(images):
    images = images.reshape(-1,64,64,1)
    images = np.repeat(images, 3, axis=-1)
    images = preprocess_input(images * 255.0)
    feats = vgg.predict(images, verbose=0)
    return feats.reshape(feats.shape[0], -1)


def predict(model, sample):
    pred = model.predict_batch(sample.reshape(1,-1))[0]
    print("Dog" if pred == 1 else "Cat")
    return pred


import matplotlib.pyplot as plt

def plot_results(model):
    plt.figure()
    plt.plot(model.train_losses, label="Train Loss")
    plt.plot(model.val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Scratch Model Loss Curve")
    plt.show()

    plt.figure()
    plt.plot(model.train_accs, label="Train Accuracy")
    plt.plot(model.val_accs, label="Validation Accuracy")
    plt.legend()
    plt.title("Scratch Model Accuracy Curve")
    plt.show()

def plot_tf_results(history):
    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("TensorFlow Model Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("TensorFlow Model Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


import random

def random_search(X_train, y_train, X_val, y_val, trials=10):
    best_acc = 0
    best_cfg = None

    for _ in range(trials):
        layers = random.randint(1,5)
        neurons = random.choice([32,64,128,256,512])
        activation = random.choice(["relu","sigmoid"])
        lr = random.choice([1e-2,1e-3,1e-4,1e-5])
        epochs = random.randint(3,20)

        layer_sizes = [X_train.shape[1]] + [neurons]*layers + [1]
        nn = NN(layer_sizes, activation)
        nn.train(X_train, y_train, X_val, y_val, epochs, lr)

        acc = max(nn.val_accs)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (layers, neurons, activation, lr, epochs)

    return best_cfg, best_acc


def grid_search(X_train, y_train, X_val, y_val):
    best_acc = 0
    best_cfg = None

    for layers in [1, 2, 3]:                     # NEW: number of hidden layers
        for neurons in [128, 256]:
            for activation in ["relu", "sigmoid"]:
                for lr in [1e-2, 1e-3]:

                    # Build layer sizes dynamically
                    layer_sizes = (
                        [X_train.shape[1]] +
                        [neurons] * layers +
                        [1]
                    )

                    nn = NN(layer_sizes, activation)
                    nn.train(
                        X_train, y_train,
                        X_val, y_val,
                        epochs=10,
                        lr=lr
                    )

                    acc = max(nn.val_accs)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (layers, neurons, activation, lr)

    return best_cfg, best_acc



# ------------------- Main -------------------
if __name__ == "__main__":

    # ------------------- Dataset -------------------
    dataset_path = r"C:/momen/college/third year/ML/Project/code/cats_dogs"  # adjust if needed
    images, labels = load_cats_vs_dogs_dataset(dataset_path, target_size=(64,64))

    print("Images:", images.shape)
    print("Labels:", labels.shape)

    # ------------------- Feature Extraction -------------------
    features = extract_features(images)
    print("Extracted features shape:", features.shape)

    # ------------------- Normalize Features -------------------
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    features = (features - mean) / std

    # ------------------- Split Data -------------------
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        features, labels, val_ratio=0.1, test_ratio=0.1
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # ------------------- From-Scratch Neural Network -------------------
    layer_sizes = [X_train.shape[1], 128, 64, 1]
    nn = NN(layer_sizes, hidden_activation="relu")

    nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        lr=0.001,
        batch_size=32
    )

    plot_results(nn)

    # ------------------- Scratch Model Evaluation -------------------
    scratch_preds = nn.predict_batch(X_test)
    scratch_acc = np.mean(scratch_preds == y_test)
    print("Scratch Test Accuracy:", round(scratch_acc, 4))

    # ------------------- TensorFlow Model -------------------
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import Dense # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore

    tf_model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    tf_model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = tf_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
    )

    # ------------------- TensorFlow Evaluation -------------------
    tf_preds = (tf_model.predict(X_test) >= 0.5).astype(int).reshape(-1)
    tf_acc = np.mean(tf_preds == y_test)
    print("TensorFlow Test Accuracy:", round(tf_acc, 4))
    plot_tf_results(history)


    # ------------------- Comparison Metrics -------------------
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n=== Scratch Model ===")
    print(confusion_matrix(y_test, scratch_preds))
    print(classification_report(y_test, scratch_preds))

    print("\n=== TensorFlow Model ===")
    print(confusion_matrix(y_test, tf_preds))
    print(classification_report(y_test, tf_preds))

    # ------------------- Optimization: Random Search -------------------
    print("\nRunning Random Search...")
    best_rs_cfg, best_rs_acc = random_search(
        X_train, y_train, X_val, y_val, trials=5
    )
    print("Best Random Search Config:", best_rs_cfg)
    print("Best Random Search Val Acc:", round(best_rs_acc, 4))

    # ------------------- Optimization: Grid Search -------------------
    print("\nRunning Grid Search...")
    best_gs_cfg, best_gs_acc = grid_search(
        X_train, y_train, X_val, y_val
    )
    print("Best Grid Search Config:", best_gs_cfg)
    print("Best Grid Search Val Acc:", round(best_gs_acc, 4))

    print("\n=== Experiment Completed Successfully ===")
