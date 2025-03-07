import pandas as pd 
import numpy as np
import matplotlib.pylab as plt 
import seaborn as sns 
plt.style.use("ggplot")
pd.set_option("display.max_columns", 40) # to see all the columns(here 40) for bigger datasets

# Load training data
df = pd.read_csv("train.csv")
print(f"The dataframe has [{df.shape[0]}] rows and [{df.shape[1]}] columns")
print(f"The columns are : \n{df.columns}")

print(f"Each column has : \n{df.isnull().sum()} null values") # how many null does each column have

# Heatmap of the missing values 
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.show()

# Convert 'Sex' to numeric values 
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # male = 0, female = 1 
# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Fill missing 'Embarked' values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print(f"Unique values for 'Embarked' are : {df['Embarked'].unique()}")
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) # if both Q, S are zero then we can say that it's C
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True) # removing these because they don't affect the training

# Normalize the dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # Set values between 0, 1 
df[['Age', 'Fare', 'SibSp', 'Parch']] = scaler.fit_transform(df[['Age', 'Fare', 'SibSp', 'Parch']])

# Split data into features and labels
y = df['Survived'] # this is the labels
X = df.drop(columns=['PassengerId', 'Survived'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% of the data for testing

n_x = 8 # number of features 
n_h = 5 # hidden layer neurons 
n_y = 1 # output layer (because this is a binary classification we have 1)

# Initialize random weights and biases 
W1 = np.random.randn(n_h, n_x) * 0.01  # Shape: (n_h, n_x)
b1 = np.zeros((n_h, 1))                # Shape: (n_h, 1)
W2 = np.random.randn(n_y, n_h) * 0.01  # Shape: (n_y, n_h)
b2 = np.zeros((n_y, 1))                # Shape: (n_y, 1)

# Activation functions
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    Z = np.array(Z, dtype=np.float64) # convert the type so we don't get any error
    return 1 / (1 + np.exp(-Z))

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    # Hidden layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Output layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    res = (Z1, A1, Z2, A2)
    return A2, res

# Cost function (cross-entropy loss)
def compute_cost(A2, Y):
    m = Y.shape[1] # number of examples
    
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    cost = np.squeeze(cost) # cost should be scalar
    return cost

# Backward propagation
def backward_propagation(X, Y, cache, W2):
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache

    # Output layer error
    dZ2 = A2 - Y                   # Shape: (1, m)
    dW2 = (1/m) * np.dot(dZ2, A1.T)  # Shape: (1, n_h)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer error
    dA1 = np.dot(W2.T, dZ2)         # Shape: (n_h, m)
    dZ1 = dA1 * (Z1 > 0)            # Derivative of ReLU
    dW1 = (1/m) * np.dot(dZ1, X.T)   # Shape: (n_h, n_x)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update parameters
def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    W1 = W1 - learning_rate * gradients["dW1"]
    b1 = b1 - learning_rate * gradients["db1"]
    W2 = W2 - learning_rate * gradients["dW2"]
    b2 = b2 - learning_rate * gradients["db2"]
    
    return W1, b1, W2, b2

costs = []

def neural_network_model(X, Y, n_h, num_iterations=10000, learning_rate=0.01, print_cost=False):
    np.random.seed(42)  # for reproducibility
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Initialize parameters
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        
        cost = compute_cost(A2, Y)
        costs.append(cost)
        
        gradients = backward_propagation(X, Y, cache, W2)
        
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
                  
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Running the model 
X_train_np = X_train.values.T  # Shape: (n_x, m_train)
y_train_np = y_train.values.reshape(1, -1)  # Shape: (1, m_train)

# Similarly for test data:
X_test_np = X_test.values.T    # Shape: (n_x, m_test)
y_test_np = y_test.values.reshape(1, -1)      # Shape: (1, m_test)

# Train the network:
parameters = neural_network_model(X_train_np, y_train_np, 
                                  n_h=5,             # number of neurons in the hidden layer (you can adjust)
                                  num_iterations=10000, 
                                  learning_rate=0.05, 
                                  print_cost=True)

# Plot cost vs iterations
iterations = np.arange(0, len(costs)) 

plt.figure(figsize=(8, 6))
plt.plot(iterations, costs, 'b-', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.grid(True)
plt.show()

# Test the model
A2_test, _ = forward_propagation(X_test_np, parameters["W1"], parameters["b1"],
                                  parameters["W2"], parameters["b2"])

predictions = (A2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test_np)
print("Test Accuracy:", accuracy)

# Prediction function for a single test case
def predict_survival_single(test_data, parameters):
    A2_test, _ = forward_propagation(test_data, parameters["W1"], parameters["b1"],
                                      parameters["W2"], parameters["b2"])
    prediction = (A2_test > 0.5).astype(int)
    return prediction

# Load test dataset and preprocess
df_test = pd.read_csv("test.csv")

# Preprocessing steps similar to training data
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)
df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first=True)
df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test[['Age', 'Fare', 'SibSp', 'Parch']] = scaler.transform(df_test[['Age', 'Fare', 'SibSp', 'Parch']])

# Prepare test data for prediction
X_test_df = df_test.drop(columns=['PassengerId'], axis=1).values.T

# Make predictions on test data
test_predictions = forward_propagation(X_test_df, parameters["W1"], parameters["b1"],
                                      parameters["W2"], parameters["b2"])[0]
test_predictions = (test_predictions > 0.5).astype(int)

# Output the predictions
df_test['Survived'] = test_predictions.T
print(df_test[['PassengerId', 'Survived']])

# Save predictions to a CSV file
df_test[['PassengerId', 'Survived']].to_csv('test_predictions.csv', index=False)