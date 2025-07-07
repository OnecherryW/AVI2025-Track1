import torch.nn as nn
import torch
'''
Note:官方baseline model
'''

class DeepEnsembleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_mlps=32):
        super().__init__()
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            ) for _ in range(num_mlps)
        ])

    def forward(self, features_dict):

        audio_features = features_dict.get('audio')
        video_features = features_dict.get('video')
        text_features = features_dict.get('text')
    

        batch_size = audio_features.size(0)
    

        video_features = video_features.unsqueeze(1).expand(-1, 5, -1).reshape(batch_size * 5, -1)
        text_features = text_features.unsqueeze(1).expand(-1, 5, -1).reshape(batch_size * 5, -1)
    

        audio_features = audio_features.repeat_interleave(5, dim=0)
    

        features = torch.cat([audio_features, video_features, text_features], dim=-1)
    
        # 原始的集成MLP操作
        x = features.view(features.size(0), -1)
        outputs = torch.stack([mlp(x) for mlp in self.ensemble], dim=0)
        predictions = outputs.mean(dim=0)
    
        # 重塑回batch_size维度
        predictions = predictions.view(batch_size, 5, -1).mean(dim=1)  # 对5个副本取平均
    
        return predictions
    

class DeepEnsembleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_mlps=32):
        super().__init__()
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            ) for _ in range(num_mlps)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = torch.stack([mlp(x) for mlp in self.ensemble], dim=0)
        return outputs.mean(dim=0)