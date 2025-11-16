import os
import math
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from models import get_model
from data_loader import UnifiedDataLoader, OODDataLoader
from backbone import Backbone, get_processor

def fpr_at_95_tpr(id_scores, ood_scores):
    labels = np.concatenate([
        np.ones_like(id_scores),
        np.zeros_like(ood_scores)
    ])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    
    idx = np.where(tpr >= 0.95)[0]
    return fpr[idx[0]] if len(idx) > 0 else 1.0

def incremental_ood(args):
    device = torch.device(args.device)
    backbone = Backbone(args)
    processor = get_processor(args.backbone)

    train_loader = UnifiedDataLoader(
        dataset_name=args.id_dataset,
        split="train",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ).get_dataloaders()

    test_loader = UnifiedDataLoader(
        dataset_name=args.id_dataset,
        split="test",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    ).get_dataloaders()

    if args.ood_dataset.lower() == 'none' or args.ood_dataset == args.id_dataset:
        args.ood_dataset = args.id_dataset
    else:
        args.n_id_tasks = args.n_tasks
        if args.id_dataset == 'cifar100':
            downscale_res = 32
        else:
            raise ValueError(f"Unkown downscaling for dataset '{args.id_dataset}'.")
        ood_loader = OODDataLoader(
            dataset_name=args.ood_dataset,
            transform=processor,
            downscale_res=downscale_res,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ).get_dataloader()

    model = get_model(args)

    if args.training_method.lower() != 'none':
        print("FSA on the first task...")
        backbone.finetune(train_loader[0])

    print("Extracting all features...")
    train_features = [backbone.get_features(loader) for loader in train_loader]
    test_features = [backbone.get_features(loader) for loader in test_loader]
    if args.ood_dataset != args.id_dataset:
        ood_features, _ = backbone.get_features(ood_loader)

    auc_list = []
    fpr95_list = []
    for t in range(args.n_id_tasks):
        print(f"\nLearning task {t + 1}/{args.n_id_tasks}...")
        X_train, Y_train = train_features[t]
        print("Updating model...")
        model.update(X_train, Y_train)

        X_id = np.concatenate([test_features[i][0] for i in range(t + 1)], axis=0)
        if args.ood_dataset != args.id_dataset:
            X_ood = ood_features
        else:
            X_ood = np.concatenate([test_features[i][0] for i in range(t + 1, args.n_tasks)], axis=0)
        
        print("Computing OOD scores...")
        id_scores = model.score(X_id)
        ood_scores = model.score(X_ood)

        auc = roc_auc_score(
            np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)]),
            np.concatenate([id_scores, ood_scores])
        )
        fpr95 = fpr_at_95_tpr(id_scores, ood_scores)
        auc_list.append(auc)
        fpr95_list.append(fpr95)

        print(f"AUC after task {t + 1}: {auc:.4f}")
        print(f"FPR@95 after task {t + 1}: {fpr95:.4f}")

    if args.ood_dataset == args.id_dataset:
        args.log_dir += '/setting1'
    else:
        args.log_dir += '/setting2'
    log_dir = os.path.join(args.log_dir, args.ood_dataset, args.backbone, str(args.seed))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, args.model_name)

    with open(log_file, "w") as f:
        f.write("Incremental OOD Results\n")
        f.write("Task\tAUC\tFPR@95\n")
        for t, (auc, fpr95) in enumerate(zip(auc_list, fpr95_list)):
            f.write(f"{t+1}\t{auc:.4f}\t{fpr95:.4f}\n")
        f.write("\n")
        f.write(f"Average AUC: {np.mean(auc_list):.4f}\n")
        f.write(f"Average FPR@95: {np.mean(fpr95_list):.4f}\n")


def incremental_cls(args):
    device = torch.device(args.device)
    backbone = Backbone(args)
    processor = get_processor(args.backbone)

    train_loader = UnifiedDataLoader(
        dataset_name=args.id_dataset,
        split="train",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ).get_dataloaders()

    test_loader = UnifiedDataLoader(
        dataset_name=args.id_dataset,
        split="test",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    ).get_dataloaders()

    model = get_model(args)

    n_tasks = len(train_loader)
    A = np.zeros((n_tasks, n_tasks))
    test_features = [None] * n_tasks  # Cache test features

    for t in range(n_tasks):
        if t == 0 and args.training_method.lower() != 'none':
            print("FSA on the first task...")
            backbone.finetune(train_loader[t])
        print(f"\nLearning task {t + 1}/{n_tasks}...")
        print("Extracting Features...")
        X_train, Y_train = backbone.get_features(train_loader[t])

        print("Updating model...")
        model.update(X_train, Y_train)

        # Evaluate on all seen tasks so far
        print("Evaluating...")
        for i in range(t + 1):
            if test_features[i] is None:
                test_features[i] = backbone.get_features(test_loader[i])
            X_test, Y_test = test_features[i]
            Y_pred = model.predict(X_test)
            A[t, i] = accuracy_score(Y_test, Y_pred)

        row_avg = np.mean(A[t][:t+1])
        print(f"Accuracy after learning task {t + 1}: {row_avg:.4f}")

    # Compute final metrics
    row_averages = [np.mean(row[row > 0]) for row in A]
    avg_acc = np.mean(row_averages)
    last_acc = row_averages[-1]

    # Logging
    args.log_dir += '/cls'
    log_dir = os.path.join(args.log_dir, args.id_dataset, args.backbone, str(args.seed))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, args.model_name)

    with open(log_file, "w") as f:
        f.write("Accuracy Matrix (A[t][i]):\n")
        f.write(str(A) + "\n\n")
        f.write(f"Average Accuracy: {avg_acc:.4f}\n")
        f.write(f"Last Accuracy: {last_acc:.4f}\n")
