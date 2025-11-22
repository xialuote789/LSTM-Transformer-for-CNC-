"""
Module 2: Lightweight LSTM-Transformer Architecture Design (Fixed Dimensions)
Creates three lightweight variants with different compression levels
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# SECTION 0: ORIGINAL MODEL (FROM MODULE 1)
# ============================================

class LSTMTransformer(nn.Module):
    """Original LSTM-Transformer Model"""

    def __init__(self, input_dim=3, hidden_dim=128, lstm_layers=2,
                 transformer_layers=3, num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                          bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)

        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out_proj = self.lstm_proj(lstm_out)
        lstm_final = torch.cat([h_n[-2], h_n[-1]], dim=1)

        trans_out = self.transformer(lstm_out_proj)
        trans_final = trans_out.mean(dim=1)

        combined = torch.cat([lstm_final, trans_final], dim=1)
        output = self.classifier(combined)

        return output

# ============================================
# SECTION 1: LIGHTWEIGHT ARCHITECTURES
# ============================================

class LightLSTMTransformer(nn.Module):
    """Light version: ~50% parameter reduction"""

    def __init__(self, input_dim=3, hidden_dim=96, lstm_layers=2,
                 transformer_layers=2, num_heads=6, dropout=0.2, num_classes=2):
        super().__init__()

        # Reduced hidden dimension: 128 -> 96
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM with reduced dimension
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                          bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Fewer transformer layers: 3 -> 2
        # Reduced FFN dimension: 4x -> 3x
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 3,  # Reduced from 4x
            dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)

        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out_proj = self.lstm_proj(lstm_out)
        lstm_final = torch.cat([h_n[-2], h_n[-1]], dim=1)

        trans_out = self.transformer(lstm_out_proj)
        trans_final = trans_out.mean(dim=1)

        combined = torch.cat([lstm_final, trans_final], dim=1)
        output = self.classifier(combined)

        return output


class UltraLightLSTMTransformer(nn.Module):
    """Ultra-light version: ~75% parameter reduction"""

    def __init__(self, input_dim=3, hidden_dim=64, lstm_layers=1,
                 transformer_layers=1, num_heads=4, dropout=0.15, num_classes=2):
        super().__init__()

        # Significantly reduced dimensions: 128 -> 64
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Single LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, bidirectional=True)

        # Efficient projection with simple linear layer
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Single transformer layer with reduced FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # Minimal expansion
            dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # Lightweight classifier - fixed input dimension
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),  # 192 -> 32
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)  # 32 -> 2
        )

    def forward(self, x):
        x = self.input_proj(x)

        lstm_out, (h_n, _) = self.lstm(x)

        # Simple linear projection
        lstm_out_proj = self.lstm_proj(lstm_out)

        # For single layer bidirectional LSTM: h_n shape is [2, batch, hidden_dim]
        lstm_final = torch.cat([h_n[0], h_n[1]], dim=1)  # Concatenate forward and backward

        trans_out = self.transformer(lstm_out_proj)
        trans_final = trans_out.mean(dim=1)

        combined = torch.cat([lstm_final, trans_final], dim=1)  # [batch, hidden_dim*3]
        output = self.classifier(combined)

        return output


class NanoLSTMTransformer(nn.Module):
    """Nano version: ~90% parameter reduction with weight sharing"""

    def __init__(self, input_dim=3, hidden_dim=32, lstm_layers=1,
                 transformer_layers=1, num_heads=2, dropout=0.1, num_classes=2):
        super().__init__()

        # Minimal dimensions: 128 -> 32
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Single bidirectional GRU (lighter than LSTM)
        self.gru = nn.GRU(hidden_dim, hidden_dim, lstm_layers,
                         batch_first=True, bidirectional=True)

        # Simple linear projection
        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Shared transformer layer (weight sharing)
        self.shared_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',  # Simpler activation
            batch_first=True
        )

        # Global pooling options
        self.pool_weight = nn.Parameter(torch.ones(2))

        # Minimal classifier - fixed input dimension
        # gru_final: hidden_dim * 2 (64), trans_final: hidden_dim (32)
        # combined: hidden_dim * 3 (96)
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)  # 96 -> 2

    def forward(self, x):
        x = self.input_proj(x)

        # GRU processing
        gru_out, h_n = self.gru(x)
        gru_out_proj = self.gru_proj(gru_out)

        # For single layer bidirectional GRU: h_n shape is [2, batch, hidden_dim]
        gru_final = torch.cat([h_n[0], h_n[1]], dim=1)  # [batch, hidden_dim*2]

        # Apply shared transformer layer twice (weight sharing)
        trans_out = self.shared_encoder(gru_out_proj)
        trans_out = self.shared_encoder(trans_out)  # Reuse same layer

        # Weighted pooling
        weights = torch.softmax(self.pool_weight, dim=0)
        trans_final = weights[0] * trans_out.mean(dim=1) + weights[1] * trans_out.max(dim=1)[0]

        # Combine and classify
        combined = torch.cat([gru_final, trans_final], dim=1)  # [batch, hidden_dim*3]
        output = self.classifier(combined)

        return output

# ============================================
# SECTION 2: MODEL COMPARISON UTILITIES
# ============================================

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())

def compare_models():
    """Compare all model variants"""

    print("="*80)
    print("LIGHTWEIGHT MODEL COMPARISON")
    print("="*80)

    # Create all variants
    models = {
        'Original': LSTMTransformer(),
        'Light': LightLSTMTransformer(),
        'Ultra-Light': UltraLightLSTMTransformer(),
        'Nano': NanoLSTMTransformer()
    }

    # Comparison data
    comparison_data = []

    for name, model in models.items():
        model = model.to(device)
        params = count_parameters(model)
        size_mb = params * 4 / 1024**2

        # Test forward pass
        dummy_input = torch.randn(1, 2000, 3).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        comparison_data.append({
            'Model': name,
            'Parameters': params,
            'Size (MB)': size_mb,
            'Reduction (%)': (1 - params/count_parameters(models['Original'])) * 100 if name != 'Original' else 0,
            'Output Shape': str(output.shape)
        })

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print table
    print("\nModel Comparison Table:")
    print("-" * 80)
    print(df.to_string(index=False))

    return df, models

def visualize_model_comparison(df):
    """Create comparison visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Parameter comparison
    ax = axes[0, 0]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax.bar(df['Model'], df['Parameters'], color=colors)
    ax.set_ylabel('Parameters')
    ax.set_title('Model Parameters Comparison')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')

    # 2. Size comparison
    ax = axes[0, 1]
    bars = ax.bar(df['Model'], df['Size (MB)'], color=colors)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Model Size Comparison')
    ax.tick_params(axis='x', rotation=45)

    for bar, size in zip(bars, df['Size (MB)']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{size:.2f}', ha='center', va='bottom')

    # 3. Reduction percentage
    ax = axes[1, 0]
    reduction_data = df[df['Model'] != 'Original']
    bars = ax.bar(reduction_data['Model'], reduction_data['Reduction (%)'],
                  color=colors[1:])
    ax.set_ylabel('Parameter Reduction (%)')
    ax.set_title('Parameter Reduction from Original')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% reduction')
    ax.axhline(y=75, color='g', linestyle='--', alpha=0.5, label='75% reduction')
    ax.legend()

    for bar, red in zip(bars, reduction_data['Reduction (%)']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{red:.1f}%', ha='center', va='bottom')

    # 4. Compression ratio
    ax = axes[1, 1]
    original_params = df[df['Model'] == 'Original']['Parameters'].values[0]
    compression_ratios = [original_params / p if p > 0 else 0 for p in df['Parameters']]
    bars = ax.bar(df['Model'], compression_ratios, color=colors)
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio (Original / Current)')
    ax.tick_params(axis='x', rotation=45)

    for bar, ratio in zip(bars, compression_ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{ratio:.1f}x', ha='center', va='bottom')

    plt.suptitle('Lightweight LSTM-Transformer Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

def test_model_outputs(models):
    """Test that all models produce correct output shapes"""

    print("\n" + "="*80)
    print("OUTPUT SHAPE VERIFICATION")
    print("="*80)

    batch_size = 32
    seq_length = 2000
    input_dim = 3

    dummy_input = torch.randn(batch_size, seq_length, input_dim).to(device)

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"\n{name} Model:")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Verify output is correct shape for binary classification
        assert output.shape == (batch_size, 2), f"Unexpected output shape for {name}"
        print(f"  ✓ Output shape verified")

def analyze_layer_distribution(models):
    """Analyze layer distribution for each model"""

    print("\n" + "="*80)
    print("LAYER DISTRIBUTION ANALYSIS")
    print("="*80)

    layer_stats = {}

    for name, model in models.items():
        model = model.to(device)

        # Count layers by type
        layer_counts = {
            'Linear': 0,
            'LSTM': 0,
            'GRU': 0,
            'Transformer': 0,
            'Conv1d': 0,
            'Other': 0
        }

        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_counts['Linear'] += 1
            elif isinstance(module, nn.LSTM):
                layer_counts['LSTM'] += 1
            elif isinstance(module, nn.GRU):
                layer_counts['GRU'] += 1
            elif isinstance(module, nn.TransformerEncoderLayer):
                layer_counts['Transformer'] += 1
            elif isinstance(module, nn.Conv1d):
                layer_counts['Conv1d'] += 1
            elif len(list(module.children())) == 0 and module != model:
                layer_counts['Other'] += 1

        layer_stats[name] = layer_counts

        print(f"\n{name} Model:")
        for layer_type, count in layer_counts.items():
            if count > 0:
                print(f"  {layer_type}: {count}")

    return layer_stats

# ============================================
# SECTION 3: ARCHITECTURE DETAILS
# ============================================

def print_architecture_details():
    """Print detailed architecture information"""

    print("\n" + "="*80)
    print("ARCHITECTURE OPTIMIZATION DETAILS")
    print("="*80)

    details = {
        'Light': {
            'Hidden Dim': '128 → 96 (25% reduction)',
            'LSTM Layers': '2 (unchanged)',
            'Transformer Layers': '3 → 2',
            'Attention Heads': '8 → 6',
            'FFN Expansion': '4x → 3x',
            'Key Innovation': 'Balanced reduction across all components'
        },
        'Ultra-Light': {
            'Hidden Dim': '128 → 64 (50% reduction)',
            'LSTM Layers': '2 → 1',
            'Transformer Layers': '3 → 1',
            'Attention Heads': '8 → 4',
            'FFN Expansion': '4x → 2x',
            'Key Innovation': 'Simple linear projection for efficiency'
        },
        'Nano': {
            'Hidden Dim': '128 → 32 (75% reduction)',
            'LSTM → GRU': 'Replace LSTM with GRU (25% fewer parameters)',
            'Transformer Layers': 'Weight sharing (1 layer used twice)',
            'Attention Heads': '8 → 2',
            'FFN Expansion': '4x → 2x',
            'Key Innovation': 'Weight sharing & learnable pooling'
        }
    }

    for model_name, specs in details.items():
        print(f"\n{model_name} Model Optimizations:")
        print("-" * 40)
        for key, value in specs.items():
            print(f"  • {key}: {value}")

# ============================================
# SECTION 4: DETAILED PARAMETER BREAKDOWN
# ============================================

def detailed_parameter_breakdown(models):
    """Show detailed parameter breakdown for each model"""

    print("\n" + "="*80)
    print("DETAILED PARAMETER BREAKDOWN")
    print("="*80)

    for name, model in models.items():
        print(f"\n{name} Model Parameters:")
        print("-" * 40)

        total = 0
        components = {}

        for module_name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    # Group by component
                    if 'input_proj' in module_name:
                        component = 'Input Projection'
                    elif 'lstm' in module_name.lower():
                        component = 'LSTM/GRU'
                    elif 'gru' in module_name.lower():
                        component = 'LSTM/GRU'
                    elif 'transformer' in module_name or 'encoder' in module_name or 'shared_encoder' in module_name:
                        component = 'Transformer'
                    elif 'classifier' in module_name:
                        component = 'Classifier'
                    elif 'proj' in module_name:
                        component = 'Projections'
                    else:
                        component = 'Other'

                    if component not in components:
                        components[component] = 0
                    components[component] += params
                    total += params

        for comp, params in sorted(components.items(), key=lambda x: x[1], reverse=True):
            percentage = params / total * 100
            print(f"  {comp}: {params:,} ({percentage:.1f}%)")
        print(f"  Total: {total:,}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution for Module 2"""

    print("="*80)
    print("MODULE 2: LIGHTWEIGHT LSTM-TRANSFORMER DESIGN")
    print("="*80)

    # Compare models
    df, models = compare_models()

    # Visualize comparison
    fig = visualize_model_comparison(df)

    # Test outputs
    test_model_outputs(models)

    # Analyze layer distribution
    layer_stats = analyze_layer_distribution(models)

    # Print architecture details
    print_architecture_details()

    # Detailed parameter breakdown
    detailed_parameter_breakdown(models)

    # Summary
    print("\n" + "="*80)
    print("MODULE 2 COMPLETE")
    print("="*80)
    print("\nKey Achievements:")
    print(f"1. Light Model: {df[df['Model']=='Light']['Reduction (%)'].values[0]:.1f}% reduction")
    print(f"2. Ultra-Light Model: {df[df['Model']=='Ultra-Light']['Reduction (%)'].values[0]:.1f}% reduction")
    print(f"3. Nano Model: {df[df['Model']=='Nano']['Reduction (%)'].values[0]:.1f}% reduction")
    print("\nAll models maintain the LSTM-Transformer architecture while achieving significant compression")
    print("\nNext: Module 3 - Knowledge Distillation Training")

    # Save models for next module
    torch.save(models, 'lightweight_models.pth')
    print("\nModels saved to 'lightweight_models.pth' for Module 3")

    return models, df

if __name__ == "__main__":
    models, comparison_df = main()