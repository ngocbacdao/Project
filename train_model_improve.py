import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import percentileofscore
from multiprocessing import Pool, cpu_count
import pickle

def ref_ext(row):
    var_list = []
    chr = row['chr']
    start = row['start']
    end = row['end']

    for i in range(start, end + 1):
        variable = f'{chr}_{i}'
        var_list.append(variable)
    return var_list

def peak_loc(row):
    score = row['score']
    start = row['start']
    end = row['end']
    chrom = row['chrom']

    positions = []
    scores = []

    for i in range(start, end + 1):
        positions.append(f'{chrom}_{i}')
        scores.append(score)

    return positions, scores

def unnest(row):
    positions = row['position']
    scores = row['score']
    data = [{'position': p, 'score': s} for p, s in zip(positions, scores)]
    return pd.DataFrame(data)

def lazy_load_data(directory, label):
    for wig_file in os.listdir(directory):
        if wig_file.endswith('.wig'):
            file_path = os.path.join(directory, wig_file)
            yield file_path

def lazy_process_data(args):
    file_path, test, label = args
    df = pd.read_csv(file_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'score'])
    df = df[~df['chrom'].str.startswith('#')]
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    if df['start'].min() > test['end'].iloc[0]:
        df_pivot_mod = pd.DataFrame([[0] * len(ref_list)], columns=ref_list)
        df_pivot_mod['label'] = label
    else:
        for i in range(len(df)):
            if df['start'].iloc[i] < test['start'].iloc[0] and df['end'].iloc[i] in range(test['start'].iloc[0], test['end'].iloc[0]):
                df.loc[i, 'start'] = int(test['start'].iloc[0])
            if df['end'].iloc[i] > test['end'].iloc[0] and df['start'].iloc[i] in range(test['start'].iloc[0], test['end'].iloc[0]):
                df.loc[i, 'end'] = int(test['end'].iloc[0])

        df_filtered = df[(df['start'] >= test['start'].iloc[0]) & (df['end'] <= test['end'].iloc[0])]
        df_transform = df_filtered.apply(peak_loc, axis=1)
        df_transform = pd.DataFrame(df_transform.tolist(), columns=['position', 'score'])
        df1_transform_unnested = df_transform.apply(unnest, axis=1).reset_index(drop=True)
        result_df = pd.concat(df1_transform_unnested.tolist(), ignore_index=True)
        new_values_unique = list(set(ref_list) - set(result_df['position']))
        new_df = pd.DataFrame({'position': new_values_unique, 'score': [0] * len(new_values_unique)})
        new_df_total = pd.concat([result_df, new_df], ignore_index=True)
        df_pivot = new_df_total.pivot_table(index=None, columns='position', values='score', aggfunc='first').reset_index()
        df_pivot_mod = df_pivot.drop(columns=['index'])

        df_pivot_mod['label'] = label

    return df_pivot_mod

def lazy_concatenate_dfs(df_list):
    return pd.concat(df_list, axis=0, ignore_index=True)

# Set the directory containing the WIG files
direction = '/group/sbs007/bdao/project/data/H3K4me3/wig/test'
test = pd.read_csv('/group/sbs007/bdao/project/scripts/cpu/Book.csv', sep=',', header=None, names=['chr', 'start', 'end'])
test['start'] = test['start'].astype(int)
test['end'] = test['end'].astype(int)
ref = test.apply(ref_ext, axis=1)
ref_list = [item for sublist in ref for item in sublist]

# Process each WIG file
df_list = []

# Iterate over each label directory
for label in ['Healthy', 'CRC']:
    label_directory = os.path.join(direction, label)
    # Lazy loading and processing of data
    lazy_loaded_data = ((file_path, test, label) for file_path in lazy_load_data(label_directory, label))
    with Pool(cpu_count()) as pool:
        df_list.extend(pool.map(lazy_process_data, lazy_loaded_data))

# Concatenate pivoted tables lazily
matrix = lazy_concatenate_dfs(df_list)

# Replace NaN values with 0
final_matrix = matrix.fillna(0)

# Recode the label column
recode_matrix = final_matrix
recode_matrix['label'] = recode_matrix['label'].map({'Healthy': 0, 'CRC': 1})

# Split dataset into features and target variable
X = recode_matrix.drop(columns=['label'])  # Features
y = recode_matrix['label']  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 80% training and 20% test

## Hyperparameter tuning
param_dist = {'n_estimators': [200, 500, 1000], 
              'criterion': ["gini", "entropy", "log_loss"],
              'max_features': ['sqrt', 'log2', None],
              'max_depth': [15, 20, 25, 30],
              'bootstrap': [True, False]
}
# Create a random forest classifier
rf = RandomForestClassifier()

# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(rf, 
                           param_grid=param_dist,
                           n_jobs=-1,
                           return_train_score=True,
                           scoring='accuracy',  
                           cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', grid_search.best_params_)
print('Best accuracy:', grid_search.best_score_)

# Generate predictions with the best model
y_test_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)

print(classification_report(y_test, y_test_pred))

# Save the model to a file
with open('/group/sbs007/bdao/project/scripts/cpu/train_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

# Define the permutation test function
def permutation_test(args):
    X, y, model = args
    np.random.shuffle(y)  # Shuffle the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)  # Refit the model on the permuted data
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds) * 100  # Return accuracy in percentage

# Observed accuracy (make sure test_accuracy is defined and in percentage)
observed_accuracy = accuracy * 100  # Assuming `accuracy` is the observed accuracy from the best model

# Number of iterations for permutation test
permutation_iters = 100

# Prepare arguments for parallel processing
args = [(X.copy(), y.copy(), best_rf) for _ in range(permutation_iters)]

# Perform permutation test using multiprocessing
with Pool(cpu_count()) as pool:
    permuted_accuracies = pool.map(permutation_test, args)

# Calculate average permuted accuracy
average_permuted_accuracy = np.mean(permuted_accuracies)
print("Average permuted accuracy is:", average_permuted_accuracy)

# Calculate the p-value
p_value = percentileofscore(permuted_accuracies, observed_accuracy) / 100
print("P-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("The observed accuracy is statistically significant.")
else:
    print("The observed accuracy is not statistically significant.")

print('This is train_model_improve.py')

print('Start', test['start'])
print('End', test['end'])
