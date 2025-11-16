#!/bin/bash
MODEL_NAME="ananc"
SEEDS=(0 1 2)

BACKBONE="dino"
MODE="ood"
DATASETS=("cifar100" "cub" "imagenet-r" "cars")
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running setting1 AnaNC on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --mode "${MODE}" \
            --id_dataset "${DATASET}" \
            --ood_dataset "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --lora_learning_rate 2e-4
    done
done

DATASETS=("cifar10" "t-imagenet" "places365" "fashionmnist")
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running setting2 AnaNC on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --mode "${MODE}" \
            --id_dataset "cifar100" \
            --ood_dataset "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --lora_learning_rate 2e-4
    done
done

BACKBONE="mocov3"
DATASETS=("cifar100" "cub" "imagenet-r" "cars")
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running setting1 AnaNC on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --mode "${MODE}" \
            --id_dataset "${DATASET}" \
            --ood_dataset "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-2 \
            --lora_learning_rate 1e-3
    done
done

DATASETS=("cifar10" "t-imagenet" "places365" "fashionmnist")
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running setting2 AnaNC on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --mode "${MODE}" \
            --id_dataset "cifar100" \
            --ood_dataset "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-2 \
            --lora_learning_rate 1e-3
    done
done


BACKBONE="dino"
MODE="cls"
DATASETS=("cifar100" "cub" "imagenet-r" "cars")
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running cls AnaNC on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --mode "${MODE}" \
            --id_dataset "${DATASET}" \
            --ood_dataset "none" \
            --model_name "${MODEL_NAME}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --lora_learning_rate 2e-4
    done
done