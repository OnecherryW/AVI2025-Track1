import os
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from dataset.baseline_dataset2_vote import MultimodalDatasetForTrainT2, MultimodalDatasetForTestT2
from dataset.baseline_dataset2_vote import collate_fn_train, collate_fn_test
from tqdm import tqdm, trange
from model.vote_model.M_model import GPT2Shared
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for features, mask, labels in train_bar:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features['audio'], features['video'], features['text'])
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)

def evaluate_model(model, loader, criterion, device, is_test=False):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    dim_losses = [0] * 5
    
    with torch.no_grad():
        for features, mask, labels in loader:
            features = {k: v.to(device) for k, v in features.items()}
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32).to(device)
            else:
                labels = labels.to(device)
                
            outputs = model(features['audio'], features['video'], features['text'])
            
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            predictions.append(outputs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            
            if is_test:
                for i in range(5):
                    dim_loss = criterion(outputs[:, i], labels[:, i])
                    dim_losses[i] += dim_loss.item()
    
    all_preds = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    overall_mse = mean_squared_error(all_targets, all_preds)
    
    if is_test:
        dim_mse = [mean_squared_error(all_targets[:, i], all_preds[:, i]) for i in range(5)]
        return total_loss / len(loader), overall_mse, [dl / len(loader) for dl in dim_losses], dim_mse
    else:
        return total_loss / len(loader), overall_mse

def test_model(model, test_loader, device, output_csv_path, test_csv_path):
    model.eval()
    predictions = []
    ids_list = []
    
    with torch.no_grad():
        for features, mask, ids in test_loader:
            features = {k: v.to(device) for k, v in features.items()}
            outputs = model(features['audio'], features['video'], features['text'])
            outputs = outputs * 4 + 1
            predictions.append(outputs.squeeze().detach().cpu().numpy())
            ids_list.extend(ids)
    
    all_preds = np.concatenate(predictions)
    result_df = pd.DataFrame(all_preds, 
                             columns=["Integrity", "Collegiality", "Social_versatility", 
                                      "Development_orientation", "Hireability"])
    result_df.insert(0, "id", ids_list)
    result_df.to_csv(output_csv_path, index=False)
    print(f"✅ Predictions saved to {output_csv_path}")

def cross_validation_train(args, full_dataset, device):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"🚀 Fold {fold+1}/5")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                                 collate_fn=collate_fn_train, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, 
                                collate_fn=collate_fn_train,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        model = GPT2Shared(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in trange(args.num_epochs, desc="Epochs"):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
        
        torch.save(best_model, f"fold{fold}_best_model.pth")
        best_models.append(best_model)
        
        model.load_state_dict(best_model)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
        fold_results.append(val_mse)
        print(f"📊 Fold {fold+1} | Val MSE: {val_mse*16:.4f}")
    
    avg_mse = np.mean(fold_results) * 16
    print(f"📈 Final Avg MSE: {avg_mse:.4f}")
    
    return best_models, fold_results

def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve saved to {save_path}")

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--label_col', nargs='+', required=True)
    parser.add_argument('--question', nargs='+', required=True)
    parser.add_argument('--rating_csv', required=True)

    # 输入特征参数
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--audio_dim', type=int, default=384)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=768)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--optim', type=str, default='adamw')

    # 测试参数
    parser.add_argument('--only_test', action='store_true', default=False)
    parser.add_argument('--test_output_csv', type=str, default='test_predictions.csv')
    parser.add_argument('--test_model', default='best_model.pth')

    # 模型参数
    parser.add_argument('--HCPdropout_audio', type=float, default=0.2)
    parser.add_argument('--HCPdropout_video', type=float, default=0.2)
    parser.add_argument('--HCPdropout_text', type=float, default=0.2)
    parser.add_argument('--HCPdropout_pure_text', type=float, default=0.1)
    parser.add_argument('--use_prompt', type=bool, default=False)
    parser.add_argument('--unified_dim', type=int, default=512)
    parser.add_argument('--heads_num', type=int, default=4)
    parser.add_argument('--ATCdropout', type=float, default=0.3)
    parser.add_argument('--VTCdropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--enhancer_dim', type=int, default=512)
    parser.add_argument('--TFEdropout', type=float, default=0.2)
    parser.add_argument('--RHdropout', type=float, default=0.2)
    parser.add_argument('--target_dim', type=int, default=5)
    parser.add_argument('--num_modalities', type=int, default=3)
    parser.add_argument('--modalities', type=str, default="audio,video,text")
    parser.add_argument('--output_model', default='best_model.pth')
    parser.add_argument('--loss_plot_path', type=str, default='loss_plot.png')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--training_time', type=str)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args_file = os.path.join(args.log_dir, f"args_{timestamp}.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"📝 Args saved to {args_file}")
    
    # 数据集初始化
    full_train_set = MultimodalDatasetForTrainT2(
        args.train_csv, args.audio_dir, args.video_dir, 
        args.text_dir, args.question, args.label_col, 
        args.rating_csv, args
    )
    test_set = MultimodalDatasetForTestT2(
        args.test_csv, args.audio_dir, args.video_dir, 
        args.text_dir, args.question, args.rating_csv, args
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                            collate_fn=collate_fn_test,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    if not args.only_test:
        best_models, fold_results = cross_validation_train(args, full_train_set, device)
        
        best_model_idx = np.argmin(fold_results)
        print(f"🏆 Selecting best model from fold {best_model_idx+1}")
        model = GPT2Shared(args).to(device)
        model.load_state_dict(torch.load(f"fold{best_model_idx}_best_model.pth"))
        torch.save(model.state_dict(), args.output_model)
        
        # 直接对无标签测试集进行预测
        print("Generating predictions on the test set...")
        test_model(model, test_loader, device, args.test_output_csv, args.test_csv)
    else:
        model = GPT2Shared(args).to(device)
        model.load_state_dict(torch.load(args.test_model))
        test_model(model, test_loader, device, args.test_output_csv, args.test_csv)

if __name__ == '__main__':
    main()
