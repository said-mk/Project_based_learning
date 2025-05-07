import numpy as np

data = np.genfromtxt("height_and_weight_data.csv", delimiter=",", skip_header=1)
data = data[:, 1:]  # Skip the first column (index column)
total_samples = data.shape[0]
print("Total samples:", total_samples)

# Split the data into training and testing sets
train_size = int(0.95 * total_samples)
shuffled_indices = np.random.permutation(total_samples)

train_data = data[shuffled_indices[:train_size]]
test_data = data[shuffled_indices[train_size:]]
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Separate features and labels
X_train = train_data[:, 0].reshape(-1, 1)  # Height
y_train = train_data[:, 1]  # Weight

X_test = test_data[:, 0].reshape(-1, 1)  # Height
y_test = test_data[:, 1]  # Weight
# Normalize the data
X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)

X_train_normalized = (X_train - X_train_mean) / X_train_std
X_test_normalized = (X_test - X_train_mean) / X_train_std

# apply linear regression using Ordinary Least square method
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for matplotlib
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train,y_train)
plt.xlabel("Independent Variable(Weight)")
plt.ylabel("dependent variable")
plt.title("Scatter Plot: Independent vs Dependent Variable")
plt.show()

