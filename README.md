# smart-contract-scanner

End‑to‑end pipeline for converting Solidity smart‑contract bytecode to 256×256 grayscale
images, training a ResNet‑18 CNN to classify nine vulnerability types plus a safe class,
and evaluating the model exactly as described in the accompanying research paper.

Prerequisites (PyPI):
  pip install torch torchvision torchaudio==2.* pillow scikit-learn pandas tqdm py-solc-x

The smart_contract_cnn.py script will:
 1. Clone/download the Consolidated Ground Truth (CGT) dataset and Forta benign set.
 2. Sample 100 contracts per class (900 from CGT, 100 benign).
 3. Compile each Solidity source to **runtime bytecode** via solcx (cached to .json).
 4. Convert bytecode to 256×256 PNGs in ./images/CLASS/uuid.png using `bytecode2img`.
 5. Build a PyTorch `ImageFolder` dataset, stratify 80/10/10 train/val/test.
 6. Fine‑tune ResNet‑18 (single‑channel) with Adam (lr=1e‑4) for up to 50 epochs.
 7. Report accuracy, macro‑P/R/F1, confusion matrix, per‑class ROC–AUC, and save
    the trained model as model.pth.

Run:
    python smart_contract_cnn.py  --epochs 50 --batch 32 --gpu 0