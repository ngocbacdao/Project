import pandas as pd
import numpy as np
import os
import gc
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation functions
def ref_ext(row):
    return [f'{row["chr"]}_{i}' for i in range(row['start'], row['end'] + 1)]

def lazy_load_data(directory):
    files = []
    for wig_file in os.listdir(directory):
        if wig_file.endswith('.wig'):
            files.append((os.path.join(directory, wig_file), wig_file))
    return files

def lazy_process_data(args):
    file_path, test, ref_set, label = args
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['chrom', 'start', 'end', 'score'], comment='#')
    matrix = np.zeros(len(ref_set), dtype=np.float32)

    if df['start'].min() > test['end'].iloc[0] or df['end'].max() < test['start'].iloc[0]:
        return np.concatenate([matrix.reshape(1, -1), np.full((1, 1), label, dtype=np.int32)], axis=1)
    
    for _, row in df.iterrows():
        for i in range(max(row['start'], test['start'].iloc[0]), min(row['end'], test['end'].iloc[0]) + 1):
            key = f'{row["chrom"]}_{i}'
            if key in ref_set:
                matrix[ref_set[key]] = row['score']

    return np.concatenate([matrix.reshape(1, -1), np.full((1, 1), label, dtype=np.int32)], axis=1)

def lazy_concatenate_dfs(df_list):
    df_list = [arr.astype(np.float32) for arr in df_list]
    return csr_matrix(np.vstack(df_list))

def data_preparation(directory, test, ref_list):
    start_time = time.time()
    ref_set = {v: i for i, v in enumerate(ref_list)}
    df_list = []

    for label in ['Healthy', 'CRC']:
        label_directory = os.path.join(directory, label)
        numeric_label = 0 if label == 'Healthy' else 1
        lazy_loaded_data = ((file_path, test, ref_set, numeric_label) for file_path, _ in lazy_load_data(label_directory))

        with Pool(cpu_count()) as pool:
            df_list.extend(pool.map(lazy_process_data, lazy_loaded_data))

    matrix = lazy_concatenate_dfs(df_list)
    matrix.data[np.isnan(matrix.data)] = 0
    labels = matrix[:, -1].toarray().ravel().astype(np.int32)
    X = matrix[:, :-1].toarray().astype(np.float32)
    y = labels

    del df_list, matrix, labels
    gc.collect()

    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds.")
    return X, y

def split_data(X, y):
    start_time = time.time()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=21, stratify=y_temp)
    end_time = time.time()
    print(f"Data splitting took {end_time - start_time:.2f} seconds.")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_autoencoder(input_dim, encoding_dim, log_dir):
    autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    return autoencoder, optimizer, criterion, writer

def train_autoencoder(autoencoder, optimizer, criterion, X_train, X_val, epochs, batch_size, log_dir, reduce_lr_patience=10, early_stopping_patience=50):
    autoencoder.train()
    writer = SummaryWriter(log_dir=log_dir)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=reduce_lr_patience, factor=0.1, verbose=True)

    X_train, X_val = X_train.to(device), X_val.to(device)  # Move tensors to device

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]

            optimizer.zero_grad()
            outputs = autoencoder(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_outputs = autoencoder(X_val)
        val_loss = criterion(val_outputs, X_val).item()
        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', epoch_loss / len(X_train), epoch)
        writer.add_scalar('Loss/val', val_loss / len(X_val), epoch)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(X_train):.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f'Early stopping on epoch {epoch + 1}')
                break

    writer.close()
    autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))  # Load the best model

def encode_data(autoencoder, data_loader):
    autoencoder.eval()
    encoded_data = []

    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch[0].to(device)  # Move tensor to device
            encoded = autoencoder.encoder(X_batch)
            encoded_data.append(encoded.cpu().numpy())

    return np.vstack(encoded_data)

# CNN Model
class GeneralizedCNN(nn.Module):
    def __init__(self, input_shape, dropout_rate=0.5):
        super(GeneralizedCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear((input_shape[0] // 4) * 64, 64)  # Adjusted to match the output size
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def build_generalized_cnn(input_shape, dropout_rate=0.5, lr=0.001, l2_reg=0.01):
    model = GeneralizedCNN(input_shape, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = nn.BCELoss()
    return model, optimizer, criterion

def train_evaluate_cnn(X_train, X_val, X_test, y_train, y_val, y_test, log_dir_name, batch_size, lr, l2_reg, dropout_rate=0.5, use_early_stopping=True):
    model, optimizer, criterion = build_generalized_cnn((X_train.shape[1], 1), dropout_rate, lr, l2_reg)
    writer = SummaryWriter(log_dir=log_dir_name)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience, trials = 50, 0

    for epoch in range(300):
        model.train()
        epoch_loss = 0
        y_train_pred = []
        y_train_true = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float().view(X_batch.size(0), 1, -1)  # Reshape to (batch_size, 1, length)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_train_pred.extend(outputs.detach().cpu().numpy())  # Corrected line
            y_train_true.extend(y_batch.cpu().numpy())

        model.eval()
        val_loss = 0
        y_val_pred = []
        y_val_true = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device).float().view(X_batch.size(0), 1, -1)  # Reshape to (batch_size, 1, length)
                y_batch = y_batch.to(device).float()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                val_loss += loss.item()
                y_val_pred.extend(outputs.cpu().numpy())
                y_val_true.extend(y_batch.cpu().numpy())

        scheduler.step(val_loss)
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)

        train_accuracy = accuracy_score(np.array(y_train_true), (np.array(y_train_pred) > 0.5).astype(np.float32))
        val_accuracy = accuracy_score(np.array(y_val_true), (np.array(y_val_pred) > 0.5).astype(np.float32))

        writer.add_scalar('Metrics/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('Metrics/val_accuracy', val_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/300], Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cnn_model.pth')
            trials = 0
        else:
            trials += 1
            if use_early_stopping and trials >= patience:
                print(f'Early stopping on epoch {epoch + 1}')
                break

    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()

    y_test_pred = []
    y_test_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).float().view(X_batch.size(0), 1, -1)  # Reshape to (batch_size, 1, length)
            outputs = model(X_batch)
            y_test_pred.extend(outputs.cpu().numpy())
            y_test_true.extend(y_batch.cpu().numpy())

    y_test_pred = np.array(y_test_pred)
    y_test_true = np.array(y_test_true)

    y_test_pred_binary = (y_test_pred > 0.5).astype(np.float32)

    accuracy = accuracy_score(y_test_true, y_test_pred_binary)
    precision = precision_score(y_test_true, y_test_pred_binary)
    recall = recall_score(y_test_true, y_test_pred_binary)
    auc = roc_auc_score(y_test_true, y_test_pred)

    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test AUC:", auc)
    print(classification_report(y_test_true, y_test_pred_binary))

    writer.add_scalar('Metrics/test_accuracy', accuracy, 0)
    writer.add_scalar('Metrics/test_precision', precision, 0)
    writer.add_scalar('Metrics/test_recall', recall, 0)
    writer.add_scalar('Metrics/test_auc', auc, 0)
    writer.close()
    return accuracy, precision, recall, auc, model

def main():
    chr_name = 'chr5'
    directory_template = '/group/sbs007/bdao/project/data/H3K4me3/wig/chr/{}'
    chrom_sizes = pd.read_csv('/group/sbs007/bdao/project/data/H3K4me3/wig/hg19.chrom.sizes', sep='\t', header=None, names=['chr', 'size'])

    chr_size_row = chrom_sizes[chrom_sizes['chr'] == chr_name]
    if chr_size_row.empty:
        print(f"Chromosome {chr_name} not found in chrom.sizes. Skipping.")
    else:
        chr_size = 149500000

        for start in range(149500000, chr_size, 300000):
            end = min(start + 300000, chr_size)
            test = pd.DataFrame({'chr': [chr_name], 'start': [start], 'end': [end]})
            test['start'] = test['start'].astype(int)
            test['end'] = test['end'].astype(int)
            ref = test.apply(ref_ext, axis=1)
            ref_list = [item for sublist in ref for item in sublist]

            directory = directory_template.format(chr_name)

            start_time = time.time()
            X, y = data_preparation(directory, test, ref_list)
            end_time = time.time()
            print(f"Data preparation for {chr_name} {start}-{end} took {end_time - start_time:.2f} seconds.")

            print("Labels distribution in the original dataset:", np.bincount(y))

            start_time = time.time()
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
            end_time = time.time()
            print(f"Data splitting for {chr_name} {start}-{end} took {end_time - start_time:.2f} seconds.")

            X_train, X_val, X_test = torch.tensor(X_train).to(device), torch.tensor(X_val).to(device), torch.tensor(X_test).to(device)  # Move tensors to device
            y_train, y_val, y_test = torch.tensor(y_train).to(device), torch.tensor(y_val).to(device), torch.tensor(y_test).to(device)  # Move tensors to device

            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir_autoencoder = f"autoencoder_logs/{timestamp}_{chr_name}_{start}-{end}/"
            log_dir_cnn_encoded = f"cnn_encoded_logs/{timestamp}_{chr_name}_{start}-{end}/"

            start_time = time.time()
            autoencoder, optimizer, criterion, writer = build_autoencoder(X_train.shape[1], 100, log_dir_autoencoder)
            train_autoencoder(autoencoder, optimizer, criterion, X_train, X_val, epochs=1000, batch_size=32, log_dir=log_dir_autoencoder, reduce_lr_patience=10, early_stopping_patience=50)
            end_time = time.time()
            print(f"Autoencoder training took {end_time - start_time:.2f} seconds.")

            start_time = time.time()
            X_train_encoded = encode_data(autoencoder, DataLoader(TensorDataset(X_train), batch_size=32))
            X_val_encoded = encode_data(autoencoder, DataLoader(TensorDataset(X_val), batch_size=32))
            X_test_encoded = encode_data(autoencoder, DataLoader(TensorDataset(X_test), batch_size=32))
            end_time = time.time()
            print(f"Encoding data took {end_time - start_time:.2f} seconds.")

            # print(f"\nTraining and evaluating CNN on original data for {chr_name} {start}-{end}...")
            # original_accuracy, original_precision, original_recall, original_auc, cnn_original = train_evaluate_cnn(
            #     X_train, X_val, X_test, y_train, y_val, y_test,
            #     log_dir_name=f"original_{chr_name}_{start}_{end}",
            #     batch_size=16,
            #     lr=0.0001,
            #     l2_reg=0.001,
            #     use_early_stopping=False
            # )

            print(f"\nTraining and evaluating CNN on encoded data for {chr_name} {start}-{end}...")
            encoded_accuracy, encoded_precision, encoded_recall, encoded_auc, cnn_encoded = train_evaluate_cnn(
                torch.tensor(X_train_encoded).to(device), torch.tensor(X_val_encoded).to(device), torch.tensor(X_test_encoded).to(device), y_train, y_val, y_test,
                log_dir_name=f"encoded_{chr_name}_{start}_{end}",
                batch_size=16,
                lr=0.001,
                l2_reg=0.01,
                use_early_stopping=True
            )

            del X, y, X_train, X_val, X_test, y_train, y_val, y_test, X_train_encoded, X_val_encoded, X_test_encoded, autoencoder, cnn_encoded
            gc.collect()

if __name__ == "__main__":
    main()

print('This is torch_CNN_auto.py')
