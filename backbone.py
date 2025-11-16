import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

def get_processor(backbone_name):
    backbone_name = backbone_name.lower()
    if backbone_name in ['mocov3', 'dino']:
        processor = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone_name}")
    
    return processor

class Adapter(nn.Module):
    def __init__(self, hidden_dim, rank, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank  
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.enabled = True

        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        if self.enabled:
            return self.up(self.down(x)) * (self.alpha / self.rank)
        else:
            return torch.zeros_like(x)


def inject_adapters(model, rank, alpha=None):
    for block in model.blocks:
        mlp = block.mlp
        adapter = Adapter(mlp.fc2.out_features, rank, alpha)
        block.adapter = adapter

        forward = mlp.forward
        def adapter_forward(x, forward=forward, adapter=adapter):
            return forward(x) + adapter(x)

        mlp.forward = adapter_forward


def map_labels(labels):
    unique = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique)}
    mapped = [label_map[label] for label in labels]
    return torch.tensor(mapped), label_map


class Backbone:
    def __init__(self, args):
        self.args = args
        self.backbone_name = args.backbone.lower()
        self.device = args.device  

        if self.backbone_name == 'dino':
            model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
            self.model = model.to(self.device)
            self.num_features=768
        elif self.backbone_name == 'mocov3':
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            ckpt = torch.load('mocov3-vit-base-300ep.pth', map_location='cpu')['model']
            state_dict = model.state_dict()
            state_dict.update(ckpt)
            model.load_state_dict(state_dict)
            self.model = model.to(self.device)
            self.num_features=768
        else:
            raise ValueError(f"Unknown backbone name: {self.backbone_name}")

    def forward(self, inputs, train=False):
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            inputs = inputs.to(self.device)
            features = self.model.forward_features(inputs)[:, 0]
        return features

    def get_features(self, dataloader):
        all_features, all_labels = [], []
        for batch in tqdm(dataloader):
            images, labels = batch
            features = self.forward(images).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)

    def finetune(self, dataloader):
        args = self.args
        all_labels = [label.item() for _, labels in dataloader for label in labels]
        _, label_map = map_labels(all_labels)

        num_classes = len(label_map)
        self.classifier = nn.Linear(self.num_features, num_classes).to(self.device)

        param_groups = []
        method = args.training_method.lower()
        if method == 'aper':
            inject_adapters(self.model, args.rank)
            self.model.to(self.device)
            
            for p in self.model.parameters():
                p.requires_grad = False  
            self.adapters = [block.adapter for block in self.model.blocks]
            adaptor_params = [param for adapter in self.adapters for param in adapter.parameters()]
            for param in adaptor_params: param.requires_grad = True
            param_groups = [
                {"params": self.classifier.parameters(), "lr": args.learning_rate},
                {"params": adaptor_params, "lr": args.lora_learning_rate}
            ]
        else:
            raise ValueError(f"Unknown FSA method name: {method}")

        optimizer = torch.optim.Adam(param_groups)

        self.model.train()
        self.classifier.train()
        for epoch in range(args.num_epochs):
            total_loss = 0
            for images, labels in dataloader:
                labels = torch.tensor([label_map[l.item()] for l in labels]).to(self.device)
                features = self.forward(images, train=True)
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {total_loss:.4f}")
