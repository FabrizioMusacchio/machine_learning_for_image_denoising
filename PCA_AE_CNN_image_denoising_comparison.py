"""
 A script to compare PCA, AE and CNN for image denoising.
 
 author: Fabrizio Musacchio (fabriziomusacchio.com)
 date: June 19, 2023


For reproducibility:

conda create -n vscode_testruns python=3.10
conda activate vscode_testruns
conda install -y matplotlib numpy pandas scikit-learn seaborn ipykernel scikit-image
conda install -c desilinguist factor_analyzer

on linux/windows:
conda install -y tensorflow

on a mac:
conda install -c apple tensorflow-deps
conda install pip
pip install tensorflow-macos tensorflow-metal tensorflow_datasets
"""
# %% IMPORT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras
import tensorflow as tf
import random
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
# %% FUNCTIONS
def plot_digits(X, title):
    """helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=30)
# %% MAIN
# for reproducibility:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

# load sample data from OpenML and normalize it:
X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True, parser="pandas")
X = MinMaxScaler().fit_transform(X)

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=100)

# add some noise to the data:
rng = np.random.RandomState(0)
noise_scale = 0.25
noise = rng.normal(scale=noise_scale, size=X_test.shape)
X_test_noisy = X_test + noise
noise = rng.normal(scale=noise_scale, size=X_train.shape)
X_train_noisy = X_train + noise    

# calculate the MSE and PSNR and plot:
max_I = X_test.max()
mse_noisy_image  = np.mean((X_test - X_test_noisy) ** 2)
psnr_noisy_image = 10 * np.log10(max_I / mse_noisy_image)
plot_digits(X_test, "Uncorrupted test images")
plot_digits(X_test_noisy, f"Noisy test images\nMSE: {mse_noisy_image.round(2)}, PSNR: {psnr_noisy_image.round(2)} dB")
#plot_digits(X_test_noisy, f"")



# PCA:
pca = PCA(n_components=32)
pca.fit(X_train_noisy)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))

# calculate the MSE and PSNR and plot:
mse_pca  = np.mean((X_test - X_reconstructed_pca) ** 2)
psnr_pca = 10 * np.log10(max_I / mse_pca)
plot_digits(X_reconstructed_pca,
            f"PCA reconstruction\nMSE: {mse_pca.round(2)}, PSNR: {psnr_pca.round(2)} dB")




# Kernel PCA:
kernel_pca = KernelPCA(n_components=400, kernel="rbf", gamma=1e-3, 
                       fit_inverse_transform=True, alpha=5e-3 )
_ = kernel_pca.fit(X_train_noisy)
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test_noisy))

# calculate the MSE and PSNR and plot:
mse_kernel_pca  = np.mean((X_test - X_reconstructed_kernel_pca) ** 2)
psnr_kernel_pca = 10 * np.log10(max_I / mse_kernel_pca)
plot_digits(X_reconstructed_kernel_pca,
            f"Kernel PCA reconstruction\n MSE: {mse_kernel_pca.round(2)}, PSNR: {psnr_kernel_pca.round(2)} dB")




# define the Autoencoder model architecture
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

autoencoder = keras.models.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(X_train.shape[1])])

# compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=32)

# apply denoising using Autoencoder:
X_reconstructed_autoencoder = autoencoder.predict(X_test_noisy)

# calculate the MSE and PSNR and plot:
mse_ae  = np.mean((X_test - X_reconstructed_autoencoder) ** 2)
psnr_ae = 10 * np.log10(max_I / mse_ae)
plot_digits(X_reconstructed_autoencoder,
            f"Autoencoder reconstruction\nMSE: {mse_ae.round(2)}, PSNR: {psnr_ae.round(2)} dB")



# define the CNN model architecture:
cnn = keras.models.Sequential([
    keras.layers.Reshape((16, 16, 1), input_shape=(256,)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
    keras.layers.Reshape((256,))])

# compile and train the CNN:
cnn.compile(optimizer='adam', loss='mse')
cnn.fit(np.expand_dims(X_train_noisy, axis=-1), X_train, epochs=10, batch_size=32)

# apply denoising using CNN:
X_reconstructed_cnn = cnn.predict(np.expand_dims(X_test_noisy, axis=-1))

# calculate the MSE and PSNR and plot:
mse_cnn  = np.mean((X_test - X_reconstructed_cnn) ** 2)
psnr_cnn = 10 * np.log10(max_I / mse_cnn)
plot_digits(X_reconstructed_cnn,
            f"CNN reconstruction\nMSE: {mse_cnn.round(2)}, PSNR: {psnr_cnn.round(2)} dB")

# %% END