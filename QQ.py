import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets, preprocessing, model_selection
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# More complex QNN structure
def create_quantum_model(params, x=None):
    qml.templates.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True)
    for i in range(len(params[0])):
        qml.templates.BasicEntanglerLayers(params[0][i], wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(params[1], wires=range(n_qubits))

# Define the device and qnode.
n_qubits = 8  # Increase the number of qubits
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(params, x=None):
    create_quantum_model(params, x)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define a function to create quantum features.
def create_quantum_features(params, X):
    return np.array([qnode(params, x) for x in X])

# Load the wine dataset from sklearn
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to the range [-pi, pi].
scaler = preprocessing.MinMaxScaler((-np.pi, np.pi))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a cost function for training the quantum model.
def cost_fn(params):
    X_train_quantum = create_quantum_features(params, X_train_scaled)
    model = SVC(kernel='linear')
    model.fit(X_train_quantum, y_train)
    accuracy = model.score(X_train_quantum, y_train)
    return 1 - accuracy

# Hyperparameter tuning for optimization
n_layers = 2
init_params = np.random.uniform(low=-np.pi, high=np.pi, size=(2, n_layers, n_qubits, 3))

grid_search = GridSearchCV(estimator=qml.GradientDescentOptimizer(),
                           param_grid={"learning_rate": [0.01, 0.1, 1]},
                           scoring="accuracy",
                           n_jobs=-1)
grid_search.fit(cost_fn, init_params)
opt = grid_search.best_estimator_

params = opt.step(cost_fn, init_params)

# Create quantum features for the optimized parameters.
X_train_quantum = create_quantum_features(params, X_train_scaled)
X_test_quantum = create_quantum_features(params, X_test_scaled)

# Train a classical SVC model and test it.
model = SVC(kernel='linear')
model.fit(X_train_quantum, y_train)
accuracy = model.score(X_test_quantum, y_test)

print('Test accuracy:', accuracy)
