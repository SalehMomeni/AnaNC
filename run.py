import torch
import argparse
from trainer import incremental_ood, incremental_cls

def parse_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--id_dataset', type=str, required=True)
    parser.add_argument('--ood_dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['ood', 'cls'], required=True)

    # Training config
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--n_id_tasks', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default="dino")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--training_method', type=str, default="aper")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lora_learning_rate', type=float, default=2e-4)
    parser.add_argument('--backbone_learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ANC
    parser.add_argument('--D', type=int, default=5000)
    parser.add_argument('--reg', type=float, default=1e2)

    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--n_comp', type=int, default=256)
    parser.add_argument('--fc_path', type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'ood':
        incremental_ood(args)
    elif args.mode == 'cls':
        incremental_cls(args)
