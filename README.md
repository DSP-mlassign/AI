Sri Pranav Reddy Dokuru
700782457



Exercise 1: House Price Prediction with Linear Regression

Description:

This script implements a Linear Regression model to predict house prices using the California Housing dataset.

Steps Implemented:

Load Dataset: The California housing dataset is loaded using fetch_california_housing().

Data Splitting: The dataset is split into training (80%) and testing (20%) sets.

Model Training: A LinearRegression() model is trained on the dataset.

Predictions & Evaluation: The model predicts housing prices and evaluates performance using:

Mean Squared Error (MSE)

R-Squared (R²) Score

Key Functions Used:

train_test_split() – Splits the dataset into training and testing sets.

LinearRegression().fit() – Trains the model.

predict() – Makes predictions on the test dataset.

mean_squared_error() & r2_score() – Evaluate model performance.

Exercise 2: Unsupervised Learning with K-Means Clustering

Description:

This script demonstrates K-Means clustering using synthetic data.

Steps Implemented:

Generate Data: Synthetic data is created using make_blobs(n_samples=300, centers=4).

K-Means Clustering: A KMeans model with 4 clusters is initialized and trained.

Visualization: The clustered data and centroids are visualized using matplotlib.

Key Functions Used:

make_blobs() – Generates synthetic clustering data.

KMeans(n_clusters=4).fit() – Performs clustering.

scatter() – Plots data points and cluster centers.

Exercise 3: Simple Neural Network for Regression using Keras

Description:

This script builds a simple feedforward neural network using TensorFlow/Keras to perform regression.

Steps Implemented:

Data Preparation: num_features is determined based on X_train.

Model Architecture:

Input Layer: Dense(64, activation='relu')

Hidden Layer: Dense(64, activation='relu')

Output Layer: Dense(1, activation='linear') for regression.

Compilation & Training:

Optimizer: Adam(learning_rate=0.001)

Loss Function: Mean Squared Error (MSE)

Training with epochs=50, batch_size=32.

Key Functions Used:

Sequential() – Initializes the model.

Dense() – Defines layers of the neural network.

compile() – Configures the learning process.

fit() – Trains the neural network.

Exercise 4: Updating the State in the GridWorld Environment

Description:

This script simulates a GridWorld environment, where an agent moves in a grid based on an action.

Steps Implemented:

State Representation: The environment maintains an (r, c) state (row, column).

Action Handling:

0 = Up: Decreases row index (r = max(r - 1, 0)).

(Other actions omitted for brevity.)

Terminal Check: If the agent reaches a terminal state, the episode ends with a reward.

Key Functions Used:

max(r - 1, 0) – Ensures row index does not go below 0.

self.state in self.terminal – Checks if the agent reached a terminal state.


