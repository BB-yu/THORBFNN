import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, \
    mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


class RBFNet(nn.Module):
    def __init__(self, k, gamma=None, input_dim=1):
        super(RBFNet, self).__init__()
        self.k = k
        self.gamma = gamma
        self.centers = None
        self.weights = nn.Parameter(torch.randn(k, dtype=torch.float32))
        self.input_dim = input_dim

    def fit(self, X, y, epochs=100, lr=0.001):
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        kmeans.fit(X)
        self.centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        optimizer = optim.RMSprop(self.parameters(), lr=lr)
        loss_fn = RMSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(torch.tensor(X, dtype=torch.float32))
            loss = loss_fn(y_pred, torch.tensor(y, dtype=torch.float32))
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def forward(self, X):
        G = self._calculate_interpolation_matrix(X)
        return G @ self.weights

    def _calculate_interpolation_matrix(self, X):
        distances = cdist(X, self.centers.detach().numpy())
        G = torch.exp(-self.gamma * torch.tensor(distances, dtype=torch.float32))
        return G

    def predict(self, X):
        with torch.no_grad():
            return self.forward(torch.tensor(X, dtype=torch.float32)).numpy()


def preprocess_data(df, feature_combination):
    encoded_features = []
    for col in feature_combination:
        if df[col].dtype == 'object':
            encoder = LabelEncoder()
            encoded_col = encoder.fit_transform(df[col].astype(str))
            encoded_features.append(encoded_col)
        else:
            encoded_features.append(np.log(df[col].values))


    encoded_features = np.array(encoded_features).T
    encoded_features = encoded_features.astype(np.float32)
    return encoded_features


def train_and_evaluate_model(X_train, y_train, X_val, y_val, k, gamma):
    rbf_net = RBFNet(k=k, gamma=gamma, input_dim=X_train.shape[1])
    rbf_net.fit(X_train, y_train, epochs=10000, lr=0.01)
    y_pred = rbf_net.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    medae = median_absolute_error(y_val, y_pred)
    try:
        msle = mean_squared_log_error(y_val, y_pred)
    except ValueError:
        msle = None
    mape = (np.abs((y_val - y_pred) / y_val).mean()) * 100

    r2 = r2_score(y_val, y_pred)
    print(f'R² Score: {r2}')
    print(f'Mean Squared Logarithmic Error (MSLE): {msle if msle is not None else "N/A"}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Median Absolute Error (MedAE): {medae}')

    return r2, y_pred, rbf_net


file_path = 'Your path'
df = pd.read_excel(file_path)

df_cleaned = df.dropna(subset=['data cleansing'])
print(df_cleaned.columns)
df_cleaned['target variable_log'] = np.log(df_cleaned['target variable'])
feature_columns = ['feature']
correlations = {}
for col in feature_columns:
    correlations[col] = df_cleaned[col].corr(df_cleaned['target variable_log'])

print("Feature Correlations with Target:")
for feat, corr in correlations.items():
    print(f"{feat}: {corr}")


max_features = 3
all_combinations = []
for r in range(2, max_features + 1):
    all_combinations.extend(combinations(feature_columns, r))

print(f"Total Combinations to Test: {len(all_combinations)}")

total_samples = len(df_cleaned)
test_size = int(total_samples * 0.3)
train_size = total_samples - test_size

train_indices = np.random.choice(total_samples, train_size, replace=False)
val_indices = np.setdiff1d(np.arange(total_samples), train_indices)

space = [
    Integer(10, 25, name='k'),
    Real(0.05, 0.5, name='gamma')
]
@use_named_args(space)
def objective(**params):
    k = params['k']
    gamma = params['gamma']

    r2, _, _ = train_and_evaluate_model(X_train, y_train, X_val, y_val, k=k, gamma=gamma)

    return -r2



combination_results = []

for feature_combination in all_combinations:
    print(f"Testing Feature Combination: {feature_combination}")

    encoded_features = preprocess_data(df_cleaned, feature_combination)

    X_train = encoded_features[train_indices]
    y_train = df_cleaned['target variable_log'].values[train_indices]
    X_val = encoded_features[val_indices]
    y_val = df_cleaned['target variable_log'].values[val_indices]

    result = gp_minimize(objective, space, n_calls=10, random_state=42, verbose=True)


    best_k = result.x[0]
    best_gamma = result.x[1]
    best_r2 = -result.fun

    _, best_y_pred, _ = train_and_evaluate_model(X_train, y_train, X_val, y_val, k=best_k, gamma=best_gamma)

    combination_results.append({
        'features': feature_combination,
        'best_r2': best_r2,
        'best_params': {'k': best_k, 'gamma': best_gamma},
        'y_pred': best_y_pred,
        'y_val': y_val
    })

combination_results.sort(key=lambda x: x['best_r2'], reverse=True)
top_5_combinations = combination_results[:5]

for i, result in enumerate(top_5_combinations, 1):
    print(f"Rank {i}: Features={result['features']}, Best R²={result['best_r2']}, Params={result['best_params']}")


def save_results_to_excel(y_val, y_pred, file_path):
    df_results = pd.DataFrame({
        "True": np.exp(y_val),
        "Predicted": np.exp(y_pred)
    })
    df_results.to_excel(file_path, index=False)


output_dir = 'your path'
for i, result in enumerate(top_5_combinations, 1):
    output_file_path = f"{output_dir}Top_{i}_Features_{','.join(result['features'])}.xlsx"
    save_results_to_excel(result['y_val'], result['y_pred'], output_file_path)
    print(f"Saved results to {output_file_path}")