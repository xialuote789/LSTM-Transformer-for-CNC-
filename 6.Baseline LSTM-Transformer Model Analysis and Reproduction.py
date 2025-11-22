"""
Module 1: Baseline LSTM-Transformer Model Analysis and Reproduction
Analyzes the original model to identify optimization opportunities


"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# SECTION 1: ORIGINAL MODEL DEFINITION
# ============================================

class LSTMTransformer(nn.Module):
    """Original LSTM-Transformer Model"""

    def __init__(self, input_dim=3, hidden_dim=128, lstm_layers=2,
                 transformer_layers=3, num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                          bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out_proj = self.lstm_proj(lstm_out)
        lstm_final = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # Transformer processing
        trans_out = self.transformer(lstm_out_proj)
        trans_final = trans_out.mean(dim=1)

        # Combine features
        combined = torch.cat([lstm_final, trans_final], dim=1)
        output = self.classifier(combined)

        return output

# ============================================
# SECTION 2: MODEL ANALYSIS FUNCTIONS
# ============================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_model_layers(model):
    """Analyze parameters per layer"""
    layer_params = OrderedDict()
    total_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_params[name] = params
                total_params += params

    return layer_params, total_params

def calculate_flops(model, input_shape=(1, 2000, 3)):
    """Estimate FLOPs for the model"""
    # This is a simplified estimation
    def hook_fn(module, input, output):
        if isinstance(module, nn.Linear):
            flops = input[0].shape[0] * input[0].shape[1] * module.out_features * 2
        elif isinstance(module, nn.LSTM):
            # Simplified LSTM FLOPs calculation
            batch, seq, features = input[0].shape
            hidden = module.hidden_size
            flops = batch * seq * (4 * features * hidden + 4 * hidden * hidden) * 2
            if module.bidirectional:
                flops *= 2
        elif isinstance(module, nn.MultiheadAttention):
            # Simplified attention FLOPs
            batch, seq, features = input[0].shape if len(input[0].shape) == 3 else (1, *input[0].shape)
            flops = batch * seq * seq * features * 2
        else:
            flops = 0

        module.__flops__ = flops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.LSTM, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)

    # Collect FLOPs
    total_flops = 0
    for module in model.modules():
        if hasattr(module, '__flops__'):
            total_flops += module.__flops__
            delattr(module, '__flops__')

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops

def visualize_parameter_distribution(layer_params):
    """Visualize parameter distribution across layers"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort layers by parameter count
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    top_10_layers = sorted_layers[:10]

    # 1. Top 10 layers by parameters
    ax = axes[0, 0]
    names = [name.split('.')[-1][:20] for name, _ in top_10_layers]
    values = [params for _, params in top_10_layers]
    bars = ax.bar(range(len(names)), values, color='steelblue')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Parameters')
    ax.set_title('Top 10 Layers by Parameter Count')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)

    # 2. Parameter distribution by component
    ax = axes[0, 1]
    components = {'LSTM': 0, 'Transformer': 0, 'Classifier': 0, 'Other': 0}
    for name, params in layer_params.items():
        if 'lstm' in name.lower():
            components['LSTM'] += params
        elif 'transformer' in name.lower() or 'encoder' in name.lower():
            components['Transformer'] += params
        elif 'classifier' in name.lower():
            components['Classifier'] += params
        else:
            components['Other'] += params

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax.pie(components.values(), labels=components.keys(),
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Parameter Distribution by Component')

    # 3. Cumulative parameter percentage
    ax = axes[1, 0]
    cumsum = np.cumsum([params for _, params in sorted_layers])
    cumsum_pct = cumsum / cumsum[-1] * 100
    ax.plot(range(len(cumsum_pct)), cumsum_pct, 'b-', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', label='90% of parameters')
    ax.set_xlabel('Layer Index (sorted by size)')
    ax.set_ylabel('Cumulative Parameters (%)')
    ax.set_title('Cumulative Parameter Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Find how many layers contain 90% of parameters
    layers_90pct = np.argmax(cumsum_pct >= 90) + 1
    ax.text(layers_90pct, 90, f'{layers_90pct} layers\ncontain 90%',
            fontsize=10, ha='left', va='bottom')

    # 4. Parameter statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    stats_data = [
        ['Total Parameters', f'{sum(layer_params.values()):,}'],
        ['Number of Layers', f'{len(layer_params)}'],
        ['Avg Parameters/Layer', f'{sum(layer_params.values())//len(layer_params):,}'],
        ['Max Layer Parameters', f'{max(layer_params.values()):,}'],
        ['Min Layer Parameters', f'{min(layer_params.values()):,}'],
        ['Layers for 90% params', f'{layers_90pct}']
    ]

    table = ax.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(stats_data) + 1):
        if i == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#4ECDC4')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.suptitle('LSTM-Transformer Model Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

def analyze_memory_footprint(model, batch_size=32, sequence_length=2000):
    """Analyze memory footprint during inference"""
    model.eval()

    # Calculate model size
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    model_memory = param_memory + buffer_memory

    # Estimate activation memory (simplified)
    input_memory = batch_size * sequence_length * 3 * 4 / 1024**2  # float32
    hidden_dim = 128

    # LSTM activations
    lstm_memory = batch_size * sequence_length * hidden_dim * 2 * 4 / 1024**2

    # Transformer activations
    trans_memory = batch_size * sequence_length * hidden_dim * 4 / 1024**2

    # Total activation memory
    activation_memory = input_memory + lstm_memory + trans_memory

    total_memory = model_memory + activation_memory

    memory_stats = {
        'Model Parameters (MB)': param_memory,
        'Model Buffers (MB)': buffer_memory,
        'Input Activation (MB)': input_memory,
        'LSTM Activation (MB)': lstm_memory,
        'Transformer Activation (MB)': trans_memory,
        'Total Memory (MB)': total_memory
    }

    return memory_stats

# ============================================
# SECTION 3: LOAD AND ANALYZE MODEL
# ============================================

def load_and_analyze_model(model_path=None):
    """Load trained model and perform comprehensive analysis"""

    print("="*80)
    print("LSTM-TRANSFORMER MODEL ANALYSIS")
    print("="*80)

    # Create model
    model = LSTMTransformer()
    model = model.to(device)

    # Load weights if path provided
    if model_path and Path(model_path).exists():
        print(f"\n✓ Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    else:
        print("\n⚠ No model path provided or file not found, using random initialization")

    # 1. Basic statistics
    print("\n" + "="*50)
    print("1. BASIC MODEL STATISTICS")
    print("="*50)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"Model size: {total_params * 2 / 1024**2:.2f} MB (FP16)")
    print(f"Model size: {total_params / 1024**2:.2f} MB (INT8)")

    # 2. Layer analysis
    print("\n" + "="*50)
    print("2. LAYER-WISE PARAMETER ANALYSIS")
    print("="*50)

    layer_params, _ = analyze_model_layers(model)

    # Show top 5 largest layers
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 layers by parameter count:")
    for i, (name, params) in enumerate(sorted_layers[:5], 1):
        percentage = params / total_params * 100
        print(f"{i}. {name}: {params:,} params ({percentage:.1f}%)")

    # 3. Component breakdown
    print("\n" + "="*50)
    print("3. COMPONENT BREAKDOWN")
    print("="*50)

    components = {'Input Projection': 0, 'LSTM': 0, 'Transformer': 0, 'Classifier': 0}
    for name, params in layer_params.items():
        if 'input_proj' in name:
            components['Input Projection'] += params
        elif 'lstm' in name.lower():
            components['LSTM'] += params
        elif 'transformer' in name.lower() or 'encoder' in name.lower():
            components['Transformer'] += params
        elif 'classifier' in name.lower():
            components['Classifier'] += params

    for comp, params in components.items():
        percentage = params / total_params * 100
        print(f"{comp}: {params:,} params ({percentage:.1f}%)")

    # 4. Memory analysis
    print("\n" + "="*50)
    print("4. MEMORY FOOTPRINT ANALYSIS")
    print("="*50)

    memory_stats = analyze_memory_footprint(model)
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f}")

    # 5. FLOPs estimation
    print("\n" + "="*50)
    print("5. COMPUTATIONAL COMPLEXITY")
    print("="*50)

    try:
        flops = calculate_flops(model)
        print(f"Estimated FLOPs: {flops/1e9:.2f} GFLOPs")
        print(f"Estimated MACs: {flops/2e9:.2f} GMACs")
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")

    # 6. Visualizations
    print("\n" + "="*50)
    print("6. GENERATING VISUALIZATIONS")
    print("="*50)

    fig = visualize_parameter_distribution(layer_params)

    return model, layer_params, components

# ============================================
# SECTION 4: OPTIMIZATION OPPORTUNITIES
# ============================================

def identify_optimization_opportunities(model, layer_params):
    """Identify specific opportunities for model lightweighting"""

    print("\n" + "="*80)
    print("OPTIMIZATION OPPORTUNITIES ANALYSIS")
    print("="*80)

    opportunities = []

    # 1. Check hidden dimensions
    print("\n1. Dimension Reduction Opportunities:")
    print("-" * 40)

    hidden_dim = 128
    print(f"Current hidden dimension: {hidden_dim}")
    print(f"Suggested reductions:")
    print(f"  - Light: 96 (25% reduction)")
    print(f"  - Ultra-Light: 64 (50% reduction)")
    print(f"  - Nano: 32 (75% reduction)")
    opportunities.append("Reduce hidden dimensions")

    # 2. Check transformer layers
    print("\n2. Layer Reduction Opportunities:")
    print("-" * 40)

    transformer_layers = 3
    lstm_layers = 2
    print(f"Current configuration:")
    print(f"  - Transformer layers: {transformer_layers}")
    print(f"  - LSTM layers: {lstm_layers}")
    print(f"Suggested reductions:")
    print(f"  - Reduce Transformer to 2 or 1 layer")
    print(f"  - Reduce LSTM to 1 layer")
    opportunities.append("Reduce number of layers")

    # 3. Attention heads
    print("\n3. Attention Head Optimization:")
    print("-" * 40)

    num_heads = 8
    print(f"Current attention heads: {num_heads}")
    print(f"Suggested reductions:")
    print(f"  - Light: 6 heads")
    print(f"  - Ultra-Light: 4 heads")
    print(f"  - Nano: 2 heads")
    opportunities.append("Reduce attention heads")

    # 4. FFN dimension
    print("\n4. Feed-Forward Network Optimization:")
    print("-" * 40)

    ffn_dim = hidden_dim * 4
    print(f"Current FFN dimension: {ffn_dim}")
    print(f"Suggested reductions:")
    print(f"  - Use 2x instead of 4x expansion")
    print(f"  - Use bottleneck architecture")
    opportunities.append("Reduce FFN dimension")

    # 5. Parameter sharing
    print("\n5. Advanced Techniques:")
    print("-" * 40)
    print("  - Weight sharing across transformer layers")
    print("  - Low-rank factorization for linear layers")
    print("  - Depthwise separable convolutions")
    print("  - Knowledge distillation from teacher model")
    opportunities.append("Apply advanced compression techniques")

    # Calculate potential savings
    print("\n" + "="*50)
    print("POTENTIAL PARAMETER SAVINGS")
    print("="*50)

    original_params = sum(layer_params.values())

    savings_scenarios = {
        'Light (hidden=96, trans=2)': 0.45,
        'Ultra-Light (hidden=64, trans=1)': 0.75,
        'Nano (hidden=32, trans=1, lstm=1)': 0.90
    }

    for scenario, reduction in savings_scenarios.items():
        new_params = original_params * (1 - reduction)
        print(f"\n{scenario}:")
        print(f"  Estimated parameters: {int(new_params):,}")
        print(f"  Reduction: {reduction*100:.0f}%")
        print(f"  Size: {new_params * 4 / 1024**2:.2f} MB")

    return opportunities

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function for Module 1"""

    print("="*80)
    print("MODULE 1: BASELINE LSTM-TRANSFORMER ANALYSIS")
    print("="*80)

    # You can specify your trained model path here
    model_path = "/content/drive/MyDrive/CNC_Machining/best_model.pth"  # Adjust path as needed

    # Load and analyze model
    model, layer_params, components = load_and_analyze_model(model_path)

    # Identify optimization opportunities
    opportunities = identify_optimization_opportunities(model, layer_params)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 Key Findings:")
    print(f"1. Model has {sum(layer_params.values()):,} parameters")
    print(f"2. Transformer accounts for {components['Transformer']/sum(layer_params.values())*100:.1f}% of parameters")
    print(f"3. LSTM accounts for {components['LSTM']/sum(layer_params.values())*100:.1f}% of parameters")
    print(f"\n🎯 Next Steps:")
    print("1. Proceed to Module 2: Design lightweight variants")
    print("2. Implement identified optimization strategies")
    print("3. Prepare for knowledge distillation")

    return model, layer_params, opportunities

if __name__ == "__main__":
    model, layer_params, opportunities = main()