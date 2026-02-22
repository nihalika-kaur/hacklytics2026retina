OphFoundation: Multi-Imaging Foundation Model for Ophthalmology
This project develops a multi-imaging foundation model for two imaging modalities used in ophthalmology: 3D OCT and 2D IR. The model is validated for its performance and advantages in downstream ophthalmic disease diagnosis tasks (classification).

Project Overview
Two-Stage Workflow
The project consists of two main stages: Pre-training and Downstream Fine-tuning.

1. Pre-training
Dataset: Pre-training is conducted on the paired OCT-IR dataset of the UF cohort (80% of patients).
Algorithm: Uses a custom-designed pre-training algorithm to generate the foundation model weights.
Access Restriction: The pre-training dataset is only accessible to UF students.
Key Scripts:
Job submission script: slm/2025-0311-our-MM-imagenet-meanstd-1gpu.sh
Training script: main_pretrain_our_MM_relation_ViTB.py
2. Downstream Fine-tuning
Objective: Fine-tune the pre-trained foundation model (with decoder discarded and classifier added) on downstream datasets and evaluate its performance.
Datasets:
UF dataset: Remaining 20% of the UF cohort. Data labels are derived using ICD codes. Latest splits are located at data/UF-cohort/new/split. This step is restricted to UF students.
Public benchmarks: Separate downstream fine-tuning and evaluation for each modality. Public benchmark datasets include:
IR (fundus): APTOS2019, Glaucoma_fundus, IDRiD_data, JSIEC, MESSIDOR2, PAPILA, Retina
OCT (2D/3D): OCTID, DUKE, and OIMHS datasets (3D data available at OCTCube_bench/process)
Install Enviroment (Still in development)
conda create --name <env> --file requirement_conda_env.txt
conda activate <env>
pip install -r requirement.txt

Code Details
Pre-training
Main Algorithm:

Entry point: main_pretrain_our_MM_relation_ViTB.py
Pre-training function: engine_pretrain_MM.py (train_one_epoch())
Model architecture: models_mae_MM_relation_v2.py (MaskedAutoencoderViT_Dual())
Dataset handling: util/dataset_uf.py (Paired_FundusOCT_Dataset, build_transform_paired_fundus, build_transform_paired_oct)
Baseline Algorithms:

Ablation Baseline (without multi-space and intra-space supervision):
Script: main_pretrain_our_MM_relation_ViTB_baseline.py
Model: models_mae_MM_relation_v2_baseline.py
Single-Modality Baselines:
OCT-only: main_pretrain_our_MM_relation_ViTB_baseline_onlyoct.py
IR-only: main_pretrain_our_MM_relation_ViTB_baseline_onlyfundus.py
Pre-trained Encoder Initialization:

Encoder weights can be initialized with pre-trained ViT parameters from:
pretrained_weights/vit_pretrained/
RETFound ViT weights
Alternative Codebase:

Based on RETFound:
Script: main_pretrain.py
Model: models_mae.py
Fine-tuning and Evaluation
On UF Dataset:
Fine-tuning and evaluation script: main_finetune_our_MM_relation.py
On Public Benchmarks:
IR Modality: main_finetune_our_MM_relation_publicbench_2d.py
OCT Modality: main_finetune_our_MM_relation_publicbench_3d.py
Scripts for running jobs: slm/finetune/publicbench
Challenges and Issues
Current Problems
Suboptimal Performance:
The pre-trained model shows unsatisfactory downstream performance on both UF data and public benchmarks.
Classification results are biased toward a specific class, despite class imbalance being relatively mild.
UF Dataset:
Fails to achieve ideal metrics (e.g., AUC > 90%).
Public Benchmark:
Performance is inferior to RETFound.
Potential Causes
Normalization Standards:
The mean and standard deviation used for data normalization differ from ImageNet (used by ViT and RETFound). This discrepancy might lead to lower performance.
Dual Encoder Architecture:
The dual encoder (EncoderViT2D in models_mae_MM_relation_v2.py) is based on RETFound's encoder definitions. RETFound fine-tuning uses the timm.models.vision_transformer.VisionTransformer class, which might impact performance.
Learning Rate:
MAE-based pre-training converges quickly (2-3 epochs for larger learning rates). Learning rate selection may significantly influence training stability and results.
Future Experiments
Pre-training
Validate the proposed pre-training method with different configurations:
Full OCT-IR dual encoder: main_pretrain_our_MM_relation_ViTB.py
OCT-only encoder: main_pretrain_our_MM_relation_ViTB_baseline_onlyoct.py
IR-only encoder: main_pretrain_our_MM_relation_ViTB_baseline_onlyfundus.py
Experiment with different ViT parameter initialization:
Random initialization
Pre-trained ViT weights (e.g., RETFound or pretrained_weights/vit_pretrained/)
Optimize performance through hyperparameter tuning and ablation studies.
Downstream Fine-tuning
Verify pre-trained model performance for:
Dual encoder (OCT-IR)
OCT-only encoder
IR-only encoder
Evaluate performance on:
UF data: Achieve acceptable metrics (e.g., AUC > 90%).
Public benchmarks: Outperform RETFound on OCT-only and IR-only tasks.
Conduct extensive comparison experiments:
Use different initialization weights (pre-trained, random, other).
Identify if performance issues originate from fine-tuning or pre-training.
