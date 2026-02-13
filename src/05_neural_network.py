#!/usr/bin/env python3
"""
Neural Network models for DCE analysis.
Implements both Linear NN (equivalent to MNL) and Deep NN with hidden layers.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DCEDataset(Dataset):
    """PyTorch Dataset for DCE data."""
    
    def __init__(self, df):
        # Group by choice situation
        self.csids = df['csid'].unique()
        self.data = df
        
    def __len__(self):
        return len(self.csids)
    
    def __getitem__(self, idx):
        csid = self.csids[idx]
        choice_set = self.data[self.data['csid'] == csid].copy()
        choice_set = choice_set.sort_values('alt_num')
        
        # Get features (exclude structural columns)
        feature_cols = [c for c in choice_set.columns 
                       if c not in ['resp_id', 'task_id', 'alt_id', 'csid', 
                                    'alt_num', 'choice', 'csid_cat', 'csid_numeric',
                                    'comfort', 'att', 'speech', 'app', 'group', 
                                    'purchase', 'rel_price', 'rel_price_lvl']]
        
        X = choice_set[feature_cols].values.astype(np.float32)
        y = choice_set['choice'].values.astype(np.int64)
        
        # Find which alternative was chosen
        chosen_idx = np.argmax(y)
        
        return torch.FloatTensor(X), torch.LongTensor([chosen_idx])[0]


class LinearNN(nn.Module):
    """Linear Neural Network (equivalent to MNL without hidden layers)."""
    
    def __init__(self, input_dim, num_alternatives=3):
        super(LinearNN, self).__init__()
        self.num_alternatives = num_alternatives
        # Single linear layer: (num_alts * input_dim) -> num_alts
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size=num_alts, input_dim)
        # Apply linear transformation to each alternative
        utilities = self.fc(x).squeeze(-1)  # (num_alts,)
        # Apply softmax to get choice probabilities
        probs = torch.softmax(utilities, dim=0)
        return utilities, probs


class DeepNN(nn.Module):
    """Deep Neural Network with hidden layers for non-linear patterns."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_alternatives=3):
        super(DeepNN, self).__init__()
        self.num_alternatives = num_alternatives
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size=num_alts, input_dim)
        utilities = self.network(x).squeeze(-1)  # (num_alts,)
        probs = torch.softmax(utilities, dim=0)
        return utilities, probs


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """Train the neural network model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            utilities, probs = model(X_batch)
            
            loss = criterion(utilities.unsqueeze(0), y_batch.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = torch.argmax(probs)
            train_correct += (pred == y_batch).item()
            train_total += 1
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                utilities, probs = model(X_batch)
                loss = criterion(utilities.unsqueeze(0), y_batch.unsqueeze(0))
                
                val_loss += loss.item()
                pred = torch.argmax(probs)
                val_correct += (pred == y_batch).item()
                val_total += 1
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/train_total:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss/val_total:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc


def calculate_log_likelihood(model, data_loader, device='cpu'):
    """Calculate log-likelihood for model comparison."""
    
    model.eval()
    log_likelihood = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            utilities, probs = model(X_batch)
            # Get probability of chosen alternative
            chosen_prob = probs[y_batch]
            log_likelihood += torch.log(chosen_prob + 1e-10).item()
    
    return log_likelihood


def extract_wtp(model, feature_names, price_idx, device='cpu'):
    """
    Extract WTP values from neural network.
    For Linear NN: directly from weights
    For Deep NN: using gradient-based sensitivity analysis
    """
    
    model.eval()
    
    if isinstance(model, LinearNN):
        # Extract weights directly
        weights = model.fc.weight.data.cpu().numpy().flatten()
        price_coef = weights[price_idx]
        
        wtps = {}
        for i, feat in enumerate(feature_names):
            if i != price_idx and 'is_optout' not in feat:
                wtp = -weights[i] / price_coef
                wtps[feat] = wtp
        
        return wtps, weights
    
    else:  # Deep NN
        # Use gradient-based approach
        # Create zero input and perturb each feature
        num_features = len(feature_names)
        
        # Get baseline utility with all features at zero
        x_baseline = torch.zeros(1, num_features).to(device)
        baseline_util, _ = model(x_baseline)
        baseline_util = baseline_util.item()
        
        # Calculate marginal utility for each feature
        marginal_utils = []
        for i in range(num_features):
            x_pert = x_baseline.clone()
            x_pert[0, i] = 1.0
            pert_util, _ = model(x_pert)
            marginal_util = pert_util.item() - baseline_util
            marginal_utils.append(marginal_util)
        
        marginal_utils = np.array(marginal_utils)
        price_marginal = marginal_utils[price_idx]
        
        wtps = {}
        for i, feat in enumerate(feature_names):
            if i != price_idx and 'is_optout' not in feat:
                wtp = -marginal_utils[i] / price_marginal
                wtps[feat] = wtp
        
        return wtps, marginal_utils


def main():
    parser = argparse.ArgumentParser(description='Fit Neural Network models for DCE')
    parser.add_argument('--input', default='Data/dce_long_model.csv',
                       help='Input processed data CSV')
    parser.add_argument('--model_type', default='both', 
                       choices=['linear', 'deep', 'both'],
                       help='Which NN model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input)
    df = df.reset_index(drop=True)
    
    # Identify feature columns
    structural_cols = ['resp_id', 'task_id', 'alt_id', 'csid', 'alt_num', 'choice', 
                       'csid_cat', 'csid_numeric']
    original_categorical = ['comfort', 'att', 'speech', 'app', 'group', 
                           'purchase', 'rel_price', 'rel_price_lvl']
    feature_cols = [c for c in df.columns 
                    if c not in structural_cols and c not in original_categorical]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Split data by respondents (80/20 split)
    unique_resp = df['resp_id'].unique()
    train_resp, val_resp = train_test_split(unique_resp, test_size=0.2, random_state=42)
    
    train_df = df[df['resp_id'].isin(train_resp)]
    val_df = df[df['resp_id'].isin(val_resp)]
    
    print(f"\nTrain: {len(train_df)} obs, {len(train_resp)} respondents")
    print(f"Val: {len(val_df)} obs, {len(val_resp)} respondents")
    
    # Create datasets
    train_dataset = DCEDataset(train_df)
    val_dataset = DCEDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    input_dim = len(feature_cols)
    results = {}
    
    # Train Linear NN
    if args.model_type in ['linear', 'both']:
        print("\n" + "="*60)
        print("Training Linear NN (MNL-equivalent)")
        print("="*60)
        
        linear_model = LinearNN(input_dim)
        linear_model, linear_val_acc = train_model(
            linear_model, train_loader, val_loader, 
            num_epochs=args.epochs, lr=args.lr, device=device
        )
        
        # Calculate metrics
        linear_ll = calculate_log_likelihood(linear_model, val_loader, device)
        linear_wtp, linear_weights = extract_wtp(linear_model, feature_cols, 
                                                  feature_cols.index('price_eur'), device)
        
        results['linear'] = {
            'model': linear_model,
            'val_accuracy': linear_val_acc,
            'log_likelihood': linear_ll,
            'wtp': linear_wtp,
            'weights': linear_weights
        }
        
        print(f"\n✓ Linear NN - Val Accuracy: {linear_val_acc:.2f}%, LL: {linear_ll:.2f}")
    
    # Train Deep NN
    if args.model_type in ['deep', 'both']:
        print("\n" + "="*60)
        print("Training Deep NN (Non-linear)")
        print("="*60)
        
        deep_model = DeepNN(input_dim, hidden_dims=[64, 32])
        deep_model, deep_val_acc = train_model(
            deep_model, train_loader, val_loader,
            num_epochs=args.epochs, lr=args.lr, device=device
        )
        
        # Calculate metrics
        deep_ll = calculate_log_likelihood(deep_model, val_loader, device)
        deep_wtp, deep_marginals = extract_wtp(deep_model, feature_cols,
                                                feature_cols.index('price_eur'), device)
        
        results['deep'] = {
            'model': deep_model,
            'val_accuracy': deep_val_acc,
            'log_likelihood': deep_ll,
            'wtp': deep_wtp,
            'marginals': deep_marginals
        }
        
        print(f"\n✓ Deep NN - Val Accuracy: {deep_val_acc:.2f}%, LL: {deep_ll:.2f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model performance
    perf_records = []
    for model_name, res in results.items():
        perf_records.append({
            'model': model_name,
            'val_accuracy': res['val_accuracy'],
            'log_likelihood': res['log_likelihood']
        })
    
    perf_df = pd.DataFrame(perf_records)
    perf_df.to_csv(output_dir / 'nn_performance.csv', index=False)
    print(f"\n✓ Saved performance to {output_dir / 'nn_performance.csv'}")
    
    # Save WTP values
    for model_name, res in results.items():
        wtp_records = []
        for feat, wtp_val in res['wtp'].items():
            wtp_records.append({
                'feature': feat,
                'wtp': wtp_val
            })
        wtp_df = pd.DataFrame(wtp_records)
        wtp_df.to_csv(output_dir / f'wtp_{model_name}_nn.csv', index=False)
        print(f"✓ Saved WTP to {output_dir / f'wtp_{model_name}_nn.csv'}")
    
    # Save models
    for model_name, res in results.items():
        torch.save(res['model'].state_dict(), output_dir / f'model_{model_name}_nn.pt')
    
    print("\n=== Neural Network training complete ===")
    
    return results


if __name__ == '__main__':
    results = main()
