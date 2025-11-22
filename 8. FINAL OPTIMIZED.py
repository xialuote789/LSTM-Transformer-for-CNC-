"""
Module 3 FINAL OPTIMIZED: Complete Training Pipeline with Teacher Model
Includes teacher training and GPU acceleration optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.backends import cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GPU OPTIMIZATION SETTINGS
# ============================================

# Enable cudnn optimizations
cudnn.benchmark = True
cudnn.deterministic = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Set memory growth
    torch.cuda.empty_cache()

    # Use TF32 for A100 (faster matrix operations)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled for A100 optimization")

# ============================================
# MODEL DEFINITIONS (Same as before)
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

class LightLSTMTransformer(nn.Module):
    """Light version: ~50% parameter reduction"""

    def __init__(self, input_dim=3, hidden_dim=96, lstm_layers=2,
                 transformer_layers=2, num_heads=6, dropout=0.2, num_classes=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
                          bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
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

class UltraLightLSTMTransformer(nn.Module):
    """Ultra-light version: ~75% parameter reduction"""

    def __init__(self, input_dim=3, hidden_dim=64, lstm_layers=1,
                 transformer_layers=1, num_heads=4, dropout=0.15, num_classes=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                          batch_first=True, bidirectional=True)

        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)

        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out_proj = self.lstm_proj(lstm_out)
        lstm_final = torch.cat([h_n[0], h_n[1]], dim=1)

        trans_out = self.transformer(lstm_out_proj)
        trans_final = trans_out.mean(dim=1)

        combined = torch.cat([lstm_final, trans_final], dim=1)
        output = self.classifier(combined)

        return output

class NanoLSTMTransformer(nn.Module):
    """Nano version: ~90% parameter reduction with weight sharing"""

    def __init__(self, input_dim=3, hidden_dim=32, lstm_layers=1,
                 transformer_layers=1, num_heads=2, dropout=0.1, num_classes=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, lstm_layers,
                         batch_first=True, bidirectional=True)

        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.shared_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.pool_weight = nn.Parameter(torch.ones(2))

        self.classifier = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, x):
        x = self.input_proj(x)

        gru_out, h_n = self.gru(x)
        gru_out_proj = self.gru_proj(gru_out)
        gru_final = torch.cat([h_n[0], h_n[1]], dim=1)

        trans_out = self.shared_encoder(gru_out_proj)
        trans_out = self.shared_encoder(trans_out)

        weights = torch.softmax(self.pool_weight, dim=0)
        trans_final = weights[0] * trans_out.mean(dim=1) + weights[1] * trans_out.max(dim=1)[0]

        combined = torch.cat([gru_final, trans_final], dim=1)
        output = self.classifier(combined)

        return output

# ============================================
# DATA HANDLING
# ============================================

class CNCDataset(Dataset):
    """CNC Vibration Dataset for PyTorch"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_preprocessed_data():
    """Load preprocessed data that was already created"""
    preprocessed_path = Path("/content/drive/MyDrive/CNC_preprocessed_full")

    if preprocessed_path.exists():
        print("Loading preprocessed data...")
        X_train = np.load(preprocessed_path / "X_train.npy")
        X_val = np.load(preprocessed_path / "X_val.npy")
        X_test = np.load(preprocessed_path / "X_test.npy")
        y_train = np.load(preprocessed_path / "y_train.npy")
        y_val = np.load(preprocessed_path / "y_val.npy")
        y_test = np.load(preprocessed_path / "y_test.npy")

        print(f"Data loaded:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        raise FileNotFoundError("Preprocessed data not found! Please run data preprocessing first.")

# ============================================
# TEACHER MODEL TRAINING
# ============================================

class TeacherTrainer:
    """Trainer for the teacher model"""

    def __init__(self, model, train_loader, val_loader, test_loader, learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=10,
            steps_per_epoch=len(train_loader)
        )
        self.scaler = GradScaler()

        self.best_val_acc = 0
        self.best_model_state = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training Teacher')
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # More efficient

            with autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=10):
        print("\nTraining Teacher Model...")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()

            print(f"Epoch {epoch:2d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"Teacher training completed. Best Val Acc: {self.best_val_acc:.4f}")

        return self.model

# ============================================
# KNOWLEDGE DISTILLATION (Optimized)
# ============================================

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # Soft targets loss
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)

        # Hard targets loss
        hard_loss = self.ce_loss(student_outputs, targets)

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss, soft_loss, hard_loss

class FastDistillationTrainer:
    """Optimized trainer for knowledge distillation"""

    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 learning_rate=1e-3, alpha=0.7, temperature=4.0):

        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Teacher in eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.distillation_loss = DistillationLoss(alpha, temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.student.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=15,
            steps_per_epoch=len(train_loader)
        )
        self.scaler = GradScaler()

        self.best_val_acc = 0
        self.best_model_state = None

    def train_epoch(self):
        self.student.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Distillation Training')
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Get teacher predictions (cached if possible)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    teacher_outputs = self.teacher(data)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                student_outputs = self.student(data)
                loss, _, _ = self.distillation_loss(student_outputs, teacher_outputs, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        self.student.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.student(data)
                    loss = self.ce_loss(outputs, target)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=15):
        print("\nStarting Knowledge Distillation...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.student.state_dict().copy()

            print(f"Epoch {epoch:2d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if self.best_model_state is not None:
            self.student.load_state_dict(self.best_model_state)

        print(f"Training completed. Best Val Acc: {self.best_val_acc:.4f}")

        return self.student

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution - optimized version"""

    print("="*80)
    print("MODULE 3 OPTIMIZED: COMPLETE TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load preprocessed data
    print("\nStep 1: Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

    # Create datasets
    train_dataset = CNCDataset(X_train, y_train)
    val_dataset = CNCDataset(X_val, y_val)
    test_dataset = CNCDataset(X_test, y_test)

    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights),
        len(sample_weights)
    )

    # Create data loaders with optimal settings
    batch_size = 64  # Increased for A100
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,  # Increased workers
        pin_memory=True,
        persistent_workers=True  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Data loaders created (batch_size={batch_size})")

    # Step 2: Train or load teacher model
    print("\nStep 2: Preparing teacher model...")
    teacher_model = LSTMTransformer()
    teacher_path = Path("/content/drive/MyDrive/CNC_Machining/teacher_model_optimized.pth")

    if teacher_path.exists():
        print("Loading existing teacher model...")
        checkpoint = torch.load(teacher_path, map_location=device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Teacher model loaded (Val Acc: {checkpoint['best_val_acc']:.4f})")
    else:
        print("Training new teacher model...")
        teacher_trainer = TeacherTrainer(
            teacher_model,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=1e-3
        )
        teacher_model = teacher_trainer.train(epochs=10)

        # Save teacher model
        torch.save({
            'model_state_dict': teacher_model.state_dict(),
            'best_val_acc': teacher_trainer.best_val_acc
        }, teacher_path)
        print(f"Teacher model saved to {teacher_path}")

    # Step 3: Train lightweight models
    print("\nStep 3: Training lightweight models with distillation...")

    models = {
        'Light': LightLSTMTransformer(),
        'Ultra-Light': UltraLightLSTMTransformer(),
        'Nano': NanoLSTMTransformer()
    }

    results = {}

    for model_name, student_model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} Model")
        print(f"{'='*60}")

        trainer = FastDistillationTrainer(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            learning_rate=1e-3,
            alpha=0.7,
            temperature=4.0
        )

        trained_student = trainer.train(epochs=15)

        # Test evaluation
        trained_student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = trained_student(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_acc = correct / total

        results[model_name] = {
            'model': trained_student,
            'best_val_acc': trainer.best_val_acc,
            'test_acc': test_acc
        }

        # Save model
        save_path = f"/content/drive/MyDrive/CNC_Machining/{model_name.lower()}_optimized.pth"
        torch.save({
            'model_state_dict': trained_student.state_dict(),
            'best_val_acc': trainer.best_val_acc,
            'test_acc': test_acc
        }, save_path)
        print(f"Model saved to {save_path}")
        print(f"Test Accuracy: {test_acc:.4f}")

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    print("\nFinal Results:")
    print("-" * 60)
    print(f"{'Model':<15} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['best_val_acc']:<10.4f} {result['test_acc']:<10.4f}")

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return results

if __name__ == "__main__":
    results = main()