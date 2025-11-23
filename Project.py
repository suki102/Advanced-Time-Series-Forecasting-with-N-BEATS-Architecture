#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-23T11:31:35.462Z
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 1. SALES DATASET GENERATION
# =============================================================================

def generate_sales_data(n_points=2000):
    """Generate realistic retail sales data with multiple seasonality and promotions"""
    
    t = np.arange(n_points)
    
    # Base sales level
    base = 1000
    
    # Growth trend (business growth)
    trend = 0.15 * t + 0.0005 * t**2
    
    # Multiple seasonality patterns
    # Weekly seasonality (7 days)
    weekly_seasonal = 200 * np.sin(2 * np.pi * t / 7) + 100 * np.cos(2 * np.pi * t / 3.5)
    
    # Monthly seasonality (30 days)
    monthly_seasonal = 150 * np.sin(2 * np.pi * t / 30 + np.pi/4)
    
    # Yearly seasonality (365 days)
    yearly_seasonal = 300 * np.sin(2 * np.pi * t / 365) + 200 * np.cos(2 * np.pi * t / 182.5)
    
    # Promotional effects (major sales events)
    promotions = np.zeros(n_points)
    
    # Black Friday type events (every 360 days)
    for i in range(360, n_points, 360):
        if i + 10 < n_points:
            # 10-day promotional period
            promo_pattern = [50, 150, 400, 800, 1200, 800, 400, 200, 100, 50]
            promotions[i:i+len(promo_pattern)] += promo_pattern
    
    # Random promotional campaigns (5-7 day campaigns)
    campaign_starts = np.random.choice(range(100, n_points-20), 15, replace=False)
    for start in campaign_starts:
        duration = np.random.randint(5, 8)
        campaign_effect = np.random.normal(300, 100)
        pattern = np.linspace(0, campaign_effect, duration//2).tolist() + \
                 np.linspace(campaign_effect, 0, duration - duration//2).tolist()
        end = min(start + len(pattern), n_points)
        promotions[start:end] += pattern[:end-start]
    
    # Holiday effects (shorter, sharper peaks)
    holidays = np.zeros(n_points)
    holiday_positions = [i for i in range(50, n_points, 180)]  # Every ~6 months
    for pos in holiday_positions:
        if pos + 5 < n_points:
            holiday_effect = np.random.normal(500, 150)
            holidays[pos:pos+5] += [100, 300, holiday_effect, 300, 100]
    
    # Day-of-week effects (weekend boost)
    day_of_week = np.zeros(n_points)
    for i in range(n_points):
        if i % 7 in [5, 6]:  # Saturday and Sunday
            day_of_week[i] = np.random.normal(150, 30)
    
    # Noise with heteroscedasticity (higher volatility during promotions)
    base_noise = np.random.normal(0, 30, n_points)
    promo_noise_boost = np.where(promotions > 0, 50, 0)
    noise = base_noise + promo_noise_boost * np.random.normal(0, 0.5, n_points)
    
    # Combine all components
    sales = base + trend + weekly_seasonal + monthly_seasonal + yearly_seasonal + \
            promotions + holidays + day_of_week + noise
    
    # Ensure no negative sales
    sales = np.maximum(sales, 50)
    
    # Create timestamps for better visualization
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    return sales, dates

# =============================================================================
# 2. ELECTRICITY DATASET GENERATION
# =============================================================================

def generate_electricity_data(n_points=2000):
    """Generate realistic electricity consumption data with multiple seasonality"""
    
    t = np.arange(n_points)
    
    # Base consumption level (in MW)
    base = 500
    
    # Long-term trend (gradual increase due to population growth)
    trend = 0.08 * t + 0.0002 * t**2
    
    # Multiple seasonality patterns
    # Daily seasonality (24 hours - strong pattern)
    daily_seasonal = 150 * np.sin(2 * np.pi * t / 24 - np.pi/2)  # Peak during day
    
    # Weekly seasonality (168 hours)
    weekly_seasonal = 80 * np.sin(2 * np.pi * t / 168 + np.pi/4)
    
    # Yearly seasonality (8760 hours - simplified to daily)
    yearly_seasonal = 100 * np.cos(2 * np.pi * t / 365)  # Higher in winter/summer
    
    # Special day effects
    special_days = np.zeros(n_points)
    
    # Weekends (lower consumption)
    for i in range(n_points):
        if i % 7 in [0, 6]:  # Sunday and Saturday
            special_days[i] = -40
    
    # Holiday effects (significant reduction)
    holiday_positions = [i for i in range(100, n_points, 120)]
    for pos in holiday_positions:
        if pos + 3 < n_points:
            special_days[pos:pos+3] = -80
    
    # Heat wave/cold spell effects
    weather_events = np.zeros(n_points)
    event_starts = np.random.choice(range(200, n_points-50), 8, replace=False)
    for start in event_starts:
        duration = np.random.randint(3, 8)
        event_intensity = np.random.normal(100, 30)
        weather_events[start:start+duration] += event_intensity
    
    # Time-of-day specific noise (higher variance during peak hours)
    time_varying_noise = np.zeros(n_points)
    for i in range(n_points):
        hour_of_day = i % 24
        if 8 <= hour_of_day <= 20:  # Daytime hours
            time_varying_noise[i] = np.random.normal(0, 25)
        else:  # Night hours
            time_varying_noise[i] = np.random.normal(0, 15)
    
    # Random noise
    random_noise = np.random.normal(0, 10, n_points)
    
    # Combine all components
    electricity = base + trend + daily_seasonal + weekly_seasonal + yearly_seasonal + \
                 special_days + weather_events + time_varying_noise + random_noise
    
    # Ensure reasonable minimum consumption
    electricity = np.maximum(electricity, 200)
    
    # Create timestamps
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='H')  # Hourly data
    
    return electricity, dates

# =============================================================================
# 3. N-BEATS ARCHITECTURE IMPLEMENTATION
# =============================================================================

class NBeatsBlock(nn.Module):
    """Single N-BEATS block implementation"""
    
    def __init__(self, input_size, theta_size, hidden_size=256, num_layers=4, 
                 forecast_length=20, backcast_length=100):
        super(NBeatsBlock, self).__init__()
        
        # Fully connected layers
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
        
        self.fc_layers = nn.Sequential(*layers)
        self.theta_fc = nn.Linear(hidden_size, theta_size)
        self.forecast_basis = nn.Linear(theta_size, forecast_length, bias=False)
        self.backcast_basis = nn.Linear(theta_size, backcast_length, bias=False)
        
    def forward(self, x):
        features = self.fc_layers(x)
        theta = self.theta_fc(features)
        forecast = self.forecast_basis(theta)
        backcast = self.backcast_basis(theta)
        return backcast, forecast

class NBeatsModel(nn.Module):
    """Complete N-BEATS model with trend and seasonality stacks"""
    
    def __init__(self, backcast_length=100, forecast_length=20, hidden_size=256, 
                 num_blocks=3):
        super(NBeatsModel, self).__init__()
        
        # Trend stack (captures long-term patterns)
        self.trend_blocks = nn.ModuleList([
            NBeatsBlock(backcast_length, 4, hidden_size, 3, forecast_length, backcast_length)
            for _ in range(num_blocks)
        ])
        
        # Seasonality stack (captures periodic patterns)
        self.seasonality_blocks = nn.ModuleList([
            NBeatsBlock(backcast_length, 16, hidden_size, 3, forecast_length, backcast_length)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # Initialize outputs
        total_backcast = torch.zeros_like(x)
        total_forecast = torch.zeros(x.size(0), self.trend_blocks[0].forecast_basis.out_features)
        
        trend_forecasts = []
        seasonality_forecasts = []
        
        # Process trend stack
        current_input = x
        trend_forecast = torch.zeros_like(total_forecast)
        
        for block in self.trend_blocks:
            backcast, forecast = block(current_input)
            total_backcast += backcast
            trend_forecast += forecast
            current_input = current_input - backcast
        
        trend_forecasts.append(trend_forecast)
        total_forecast += trend_forecast
        
        # Process seasonality stack
        seasonality_forecast = torch.zeros_like(total_forecast)
        
        for block in self.seasonality_blocks:
            backcast, forecast = block(current_input)
            total_backcast += backcast
            seasonality_forecast += forecast
            current_input = current_input - backcast
        
        seasonality_forecasts.append(seasonality_forecast)
        total_forecast += seasonality_forecast
        
        return total_backcast, total_forecast, trend_forecasts, seasonality_forecasts

# =============================================================================
# 4. DATA LOADING AND PREPROCESSING
# =============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, data, lookback_length=100, forecast_length=20):
        self.data = data
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length
        
        # Normalize the data
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
    def __len__(self):
        return len(self.data) - self.lookback_length - self.forecast_length + 1
    
    def __getitem__(self, idx):
        lookback_start = idx
        lookback_end = idx + self.lookback_length
        forecast_start = lookback_end
        forecast_end = forecast_start + self.forecast_length
        
        x = self.scaled_data[lookback_start:lookback_end]
        y = self.scaled_data[forecast_start:forecast_end]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

# =============================================================================
# 5. TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def train_nbeats(model, train_loader, val_loader, dataset_name, num_epochs=100):
    """Train N-BEATS model"""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    train_losses = []
    val_losses = []
    
    print(f"Training N-BEATS on {dataset_name}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            _, forecast, _, _ = model(batch_x)
            loss = criterion(forecast, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                _, forecast, _, _ = model(batch_x)
                loss = criterion(forecast, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_nbeats(model, test_loader, dataset):
    """Evaluate N-BEATS model performance"""
    model.eval()
    predictions = []
    actuals = []
    trend_components = []
    seasonality_components = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            _, forecast, trend, seasonality = model(batch_x)
            predictions.extend(forecast.numpy())
            actuals.extend(batch_y.numpy())
            trend_components.extend(trend[0].numpy())
            seasonality_components.extend(seasonality[0].numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform to original scale
    predictions_original = dataset.inverse_transform(predictions)
    actuals_original = dataset.inverse_transform(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-8))) * 100
    
    return mae, rmse, mape, predictions_original, actuals_original, trend_components, seasonality_components

def evaluate_lstm(model, test_loader, dataset):
    """Evaluate LSTM model performance"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            forecast = model(batch_x)  # LSTM returns only forecasts
            predictions.extend(forecast.numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform to original scale
    predictions_original = dataset.inverse_transform(predictions)
    actuals_original = dataset.inverse_transform(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-8))) * 100
    
    return mae, rmse, mape, predictions_original, actuals_original

# =============================================================================
# 6. BASELINE MODELS
# =============================================================================

class LSTMBaseline(nn.Module):
    """LSTM baseline model"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=20):
        super(LSTMBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def naive_forecast(test_loader, dataset):
    """Naive forecast baseline (last value repetition)"""
    predictions = []
    actuals = []
    
    for batch_x, batch_y in test_loader:
        last_values = batch_x[:, -1].unsqueeze(1).repeat(1, batch_y.shape[1])
        predictions.extend(last_values.numpy())
        actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions_original = dataset.inverse_transform(predictions)
    actuals_original = dataset.inverse_transform(actuals)
    
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-8))) * 100
    
    return mae, rmse, mape, predictions_original, actuals_original

# =============================================================================
# 7. VISUALIZATION AND ANALYSIS FUNCTIONS
# =============================================================================

def plot_dataset_analysis(sales_data, electricity_data, sales_dates, electricity_dates):
    """Plot comprehensive analysis of both datasets"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Sales data overview
    axes[0, 0].plot(sales_dates[:500], sales_data[:500], 'b-', linewidth=1)
    axes[0, 0].set_title('Sales Data (First 500 Days)')
    axes[0, 0].set_ylabel('Sales Units')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Sales data seasonal patterns
    axes[0, 1].plot(sales_dates[:365], sales_data[:365], 'g-', linewidth=1)
    axes[0, 1].set_title('Sales Data (One Year)')
    axes[0, 1].set_ylabel('Sales Units')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sales distribution
    axes[0, 2].hist(sales_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_title('Sales Distribution')
    axes[0, 2].set_xlabel('Sales Units')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Electricity data overview
    axes[1, 0].plot(electricity_dates[:168], electricity_data[:168], 'r-', linewidth=1)
    axes[1, 0].set_title('Electricity Data (One Week)')
    axes[1, 0].set_ylabel('Consumption (MW)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Electricity seasonal patterns
    axes[1, 1].plot(electricity_dates[:24*7], electricity_data[:24*7], 'orange', linewidth=1)
    axes[1, 1].set_title('Electricity Data (One Week - Daily Pattern)')
    axes[1, 1].set_ylabel('Consumption (MW)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Electricity distribution
    axes[1, 2].hist(electricity_data, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 2].set_title('Electricity Distribution')
    axes[1, 2].set_xlabel('Consumption (MW)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_forecast_comparison(dataset_name, predictions, actuals, model_name):
    """Plot forecast vs actual comparisons"""
    plt.figure(figsize=(12, 8))
    
    # Plot multiple forecast sequences
    n_sequences = min(5, len(actuals) // 20)
    
    for i in range(n_sequences):
        start_idx = i * 20
        end_idx = start_idx + 20
        
        plt.subplot(2, 3, i + 1)
        plt.plot(actuals[start_idx:end_idx], 'ko-', label='Actual', linewidth=2, markersize=4)
        plt.plot(predictions[start_idx:end_idx], 'r^-', label=model_name, linewidth=1, markersize=4)
        plt.title(f'Forecast Sequence {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - {model_name} Forecast vs Actual', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_{model_name}_forecasts.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_components_analysis(dataset_name, predictions, actuals, trend_components, seasonality_components):
    """Plot trend and seasonality component analysis"""
    plt.figure(figsize=(15, 10))
    
    # Select a sample forecast sequence
    sample_idx = 0
    start_idx = sample_idx * 20
    end_idx = start_idx + 20
    
    # Convert components to numpy arrays if they're not already
    trend_components = np.array(trend_components)
    seasonality_components = np.array(seasonality_components)
    
    # Plot 1: Overall forecast decomposition
    plt.subplot(2, 2, 1)
    plt.plot(actuals[start_idx:end_idx], 'ko-', label='Actual', linewidth=2, markersize=6)
    plt.plot(predictions[start_idx:end_idx], 'ro-', label='N-BEATS Forecast', linewidth=2, markersize=4)
    plt.plot(trend_components[start_idx:end_idx], 'b--', label='Trend Component', linewidth=2)
    plt.plot(seasonality_components[start_idx:end_idx], 'g--', label='Seasonality Component', linewidth=2)
    plt.title(f'{dataset_name} - Forecast Decomposition')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Trend component analysis
    plt.subplot(2, 2, 2)
    plt.plot(trend_components[start_idx:end_idx], 'b-o', linewidth=2, markersize=4)
    plt.title('Trend Component')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Trend Value')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Seasonality component analysis
    plt.subplot(2, 2, 3)
    plt.plot(seasonality_components[start_idx:end_idx], 'g-o', linewidth=2, markersize=4)
    plt.title('Seasonality Component')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Seasonality Value')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Component contribution
    plt.subplot(2, 2, 4)
    components = ['Trend', 'Seasonality']
    contributions = [
        np.mean(np.abs(trend_components[start_idx:end_idx])),
        np.mean(np.abs(seasonality_components[start_idx:end_idx]))
    ]
    plt.bar(components, contributions, color=['blue', 'green'], alpha=0.7)
    plt.title('Component Contribution (Mean Absolute Value)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_components_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 8. MAIN EXPERIMENT PIPELINE
# =============================================================================

def run_sales_experiment():
    """Run complete experiment on sales data"""
    print("=" * 60)
    print("SALES DATASET EXPERIMENT")
    print("=" * 60)
    
    # Generate sales data
    sales_data, sales_dates = generate_sales_data(2000)
    print(f"Sales data generated: {len(sales_data)} points")
    print(f"Sales range: {sales_data.min():.0f} - {sales_data.max():.0f}")
    print(f"Sales mean: {sales_data.mean():.0f} ± {sales_data.std():.0f}")
    
    # Create dataset
    lookback_length = 100
    forecast_length = 20
    sales_dataset = TimeSeriesDataset(sales_data, lookback_length, forecast_length)
    
    # Split data
    train_size = int(0.7 * len(sales_dataset))
    val_size = int(0.15 * len(sales_dataset))
    test_size = len(sales_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        sales_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Initialize models
    nbeats_model = NBeatsModel(
        backcast_length=lookback_length,
        forecast_length=forecast_length,
        hidden_size=128,
        num_blocks=2
    )
    
    lstm_model = LSTMBaseline(output_size=forecast_length)
    
    # Train models
    nbeats_train_loss, nbeats_val_loss = train_nbeats(nbeats_model, train_loader, val_loader, "Sales", 80)
    
    # Train LSTM
    print("Training LSTM on Sales data...")
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.HuberLoss()
    
    for epoch in range(60):
        lstm_model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            lstm_optimizer.zero_grad()
            pred = lstm_model(batch_x)
            loss = lstm_criterion(pred, batch_y)
            loss.backward()
            lstm_optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'LSTM Epoch [{epoch+1}/60], Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluate models
    nbeats_mae, nbeats_rmse, nbeats_mape, nbeats_preds, nbeats_actuals, trend_comp, season_comp = evaluate_nbeats(nbeats_model, test_loader, sales_dataset)
    lstm_mae, lstm_rmse, lstm_mape, lstm_preds, lstm_actuals = evaluate_lstm(lstm_model, test_loader, sales_dataset)
    naive_mae, naive_rmse, naive_mape, naive_preds, naive_actuals = naive_forecast(test_loader, sales_dataset)
    
    # Print results
    print(f"\nSALES RESULTS:")
    print(f"{'Model':<10} | {'MAE':<8} | {'RMSE':<8} | {'MAPE':<8}")
    print(f"{'-'*40}")
    print(f"{'Naive':<10} | {naive_mae:.1f}    | {naive_rmse:.1f}    | {naive_mape:.1f}%")
    print(f"{'LSTM':<10} | {lstm_mae:.1f}    | {lstm_rmse:.1f}    | {lstm_mape:.1f}%")
    print(f"{'N-BEATS':<10} | {nbeats_mae:.1f}    | {nbeats_rmse:.1f}    | {nbeats_mape:.1f}%")
    
    # Plot forecasts
    plot_forecast_comparison("Sales", nbeats_preds, nbeats_actuals, "N-BEATS")
    plot_forecast_comparison("Sales", lstm_preds, lstm_actuals, "LSTM")
    plot_components_analysis("Sales", nbeats_preds, nbeats_actuals, trend_comp, season_comp)
    
    return {
        'metrics': {
            'nbeats': (nbeats_mae, nbeats_rmse, nbeats_mape),
            'lstm': (lstm_mae, lstm_rmse, lstm_mape),
            'naive': (naive_mae, naive_rmse, naive_mape)
        },
        'data': sales_data,
        'dates': sales_dates,
        'components': {
            'trend': trend_comp,
            'seasonality': season_comp
        }
    }

def run_electricity_experiment():
    """Run complete experiment on electricity data"""
    print("\n" + "=" * 60)
    print("ELECTRICITY DATASET EXPERIMENT")
    print("=" * 60)
    
    # Generate electricity data
    electricity_data, electricity_dates = generate_electricity_data(2000)
    print(f"Electricity data generated: {len(electricity_data)} points")
    print(f"Electricity range: {electricity_data.min():.0f} - {electricity_data.max():.0f}")
    print(f"Electricity mean: {electricity_data.mean():.0f} ± {electricity_data.std():.0f}")
    
    # Create dataset
    lookback_length = 100
    forecast_length = 20
    electricity_dataset = TimeSeriesDataset(electricity_data, lookback_length, forecast_length)
    
    # Split data
    train_size = int(0.7 * len(electricity_dataset))
    val_size = int(0.15 * len(electricity_dataset))
    test_size = len(electricity_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        electricity_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Initialize models
    nbeats_model = NBeatsModel(
        backcast_length=lookback_length,
        forecast_length=forecast_length,
        hidden_size=128,
        num_blocks=2
    )
    
    lstm_model = LSTMBaseline(output_size=forecast_length)
    
    # Train models
    nbeats_train_loss, nbeats_val_loss = train_nbeats(nbeats_model, train_loader, val_loader, "Electricity", 80)
    
    # Train LSTM
    print("Training LSTM on Electricity data...")
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.HuberLoss()
    
    for epoch in range(60):
        lstm_model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            lstm_optimizer.zero_grad()
            pred = lstm_model(batch_x)
            loss = lstm_criterion(pred, batch_y)
            loss.backward()
            lstm_optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'LSTM Epoch [{epoch+1}/60], Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluate models
    nbeats_mae, nbeats_rmse, nbeats_mape, nbeats_preds, nbeats_actuals, trend_comp, season_comp = evaluate_nbeats(nbeats_model, test_loader, electricity_dataset)
    lstm_mae, lstm_rmse, lstm_mape, lstm_preds, lstm_actuals = evaluate_lstm(lstm_model, test_loader, electricity_dataset)
    naive_mae, naive_rmse, naive_mape, naive_preds, naive_actuals = naive_forecast(test_loader, electricity_dataset)
    
    # Print results
    print(f"\nELECTRICITY RESULTS:")
    print(f"{'Model':<10} | {'MAE':<8} | {'RMSE':<8} | {'MAPE':<8}")
    print(f"{'-'*40}")
    print(f"{'Naive':<10} | {naive_mae:.1f}    | {naive_rmse:.1f}    | {naive_mape:.1f}%")
    print(f"{'LSTM':<10} | {lstm_mae:.1f}    | {lstm_rmse:.1f}    | {lstm_mape:.1f}%")
    print(f"{'N-BEATS':<10} | {nbeats_mae:.1f}    | {nbeats_rmse:.1f}    | {nbeats_mape:.1f}%")
    
    # Plot forecasts
    plot_forecast_comparison("Electricity", nbeats_preds, nbeats_actuals, "N-BEATS")
    plot_forecast_comparison("Electricity", lstm_preds, lstm_actuals, "LSTM")
    plot_components_analysis("Electricity", nbeats_preds, nbeats_actuals, trend_comp, season_comp)
    
    return {
        'metrics': {
            'nbeats': (nbeats_mae, nbeats_rmse, nbeats_mape),
            'lstm': (lstm_mae, lstm_rmse, lstm_mape),
            'naive': (naive_mae, naive_rmse, naive_mape)
        },
        'data': electricity_data,
        'dates': electricity_dates,
        'components': {
            'trend': trend_comp,
            'seasonality': season_comp
        }
    }

# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================

def main():
    print("SALES AND ELECTRICITY TIME SERIES FORECASTING")
    print("N-BEATS Architecture Implementation")
    print("=" * 60)
    
    # Run experiments
    sales_results = run_sales_experiment()
    electricity_results = run_electricity_experiment()
    
    # Comparative analysis
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    # Sales results
    sales_metrics = sales_results['metrics']
    naive_sales, lstm_sales, nbeats_sales = sales_metrics['naive'][0], sales_metrics['lstm'][0], sales_metrics['nbeats'][0]
    sales_improvement = ((naive_sales - nbeats_sales) / naive_sales) * 100
    
    # Electricity results
    electricity_metrics = electricity_results['metrics']
    naive_elec, lstm_elec, nbeats_elec = electricity_metrics['naive'][0], electricity_metrics['lstm'][0], electricity_metrics['nbeats'][0]
    elec_improvement = ((naive_elec - nbeats_elec) / naive_elec) * 100
    
    print(f"\nSALES DATASET:")
    print(f"  Naive MAE: {naive_sales:.1f}")
    print(f"  LSTM MAE: {lstm_sales:.1f} ({(naive_sales - lstm_sales)/naive_sales*100:.1f}% improvement)")
    print(f"  N-BEATS MAE: {nbeats_sales:.1f} ({sales_improvement:.1f}% improvement)")
    
    print(f"\nELECTRICITY DATASET:")
    print(f"  Naive MAE: {naive_elec:.1f}")
    print(f"  LSTM MAE: {lstm_elec:.1f} ({(naive_elec - lstm_elec)/naive_elec*100:.1f}% improvement)")
    print(f"  N-BEATS MAE: {nbeats_elec:.1f} ({elec_improvement:.1f}% improvement)")
    
    # Plot dataset analysis
    plot_dataset_analysis(
        sales_results['data'], 
        electricity_results['data'],
        sales_results['dates'],
        electricity_results['dates']
    )
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✓ Realistic Sales and Electricity datasets generated")
    print("✓ N-BEATS architecture implemented from scratch")
    print("✓ Comparative analysis with LSTM and naive baselines")
    print("✓ Comprehensive evaluation on both datasets")
    print("✓ Trend and seasonality component analysis")
    print("✓ All visualizations saved")
    
    return sales_results, electricity_results

if __name__ == "__main__":
    sales_results, electricity_results = main()