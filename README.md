# GLIDE: Graph Layer for Inference in Dynamic Environments

Official PyTorch implementation of the ECML-PKDD 2026 paper:

> **When GNNs Fail: Quantifying and Overcoming Temporal Correlation Volatility in Time Series**
>
> Chen Shao¹, Yue Wang¹, Zhenyi Zhu², Zhanbo Huang¹, Tobias Käfer¹, Zonghan Wu³, Danai Koutra⁴ ✉
>
> ¹ Karlsruhe Institute of Technology · ² HKUST · ³ East China Normal University · ⁴ University of Michigan

📄 **Paper sources:** [github.com/ChenS676/PKDD-When-GNNs-Fail-Quantifying-and-Overcoming-Temporal-Correlation-Volatility-in-Time-Series](https://github.com/ChenS676/PKDD-When-GNNs-Fail-Quantifying-and-Overcoming-Temporal-Correlation-Volatility-in-Time-Series)

---

## Getting Started

### 1. Installation

```bash
conda create -n glide python=3.9
conda activate glide
pip install -r requirements.txt
```

### 2. Data Preparation

- **Synthetic datasets**: generated with adjustable correlation ratios. Sample CSVs are included under `data/synthetic_time_series_{1..4}.csv`.
- **Real-world datasets**: Exchange Rate, Electricity, Traffic, BigElectricity (ENTSO-E). Sample preprocessed CSVs for Denmark, France, and Portugal are included under `data/`.

```bash
python scripts/preprocess_dataset.py --src data/SYNTHETIC_EASY/synthetic.csv
```

> **Note:** the `scripts/` folder is not bundled in this snapshot — preprocessing utilities will be added in a follow-up commit.

### 3. Run Training

```bash
# Exchange rate
python train_eval.py --data data/EXCHANGE --device cuda:0 \
  --batch_size 64 --epochs 10 --seq_length 12 --pred_length 12 \
  --learning_rate 0.001 --dropout 0.3 --nhid 64 \
  --gcn_bool --addaptadj --randomadj --adjtype doubletransition

# France
python train_eval.py --data data/FRANCE --device cuda:0 \
  --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "france,powermix,k2"

# Germany
python train_eval.py --data data/GERMANY --device cuda:0 \
  --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "germany,powermix,k2"

# Solar
python train_eval.py --data data/SOLAR --device cuda:0 \
  --batch_size 1 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "solar,powermix,k2"
```

### 4. Ablation Study

```bash
EPOCHS=10 WANDB_MODE=disabled WANDB_PROJECT=GLIDE \
RESULTS_CSV=./results_EXCHANGE.csv \
DATA_LIST="data/EXCHANGE" \
BATCH_LIST="64 128 256" \
LR_LIST="0.001 0.0001 0.00001" \
EXP_ID=1 bash scripts/run_experiments_ab.sh
```

## Repository layout

```
.
├── dataloader.py       # Data loading utilities
├── engine.py           # Training engine
├── model.py            # GLIDE model definition
├── model2.py           # Alternative GLIDE variant
├── train_eval.py       # Main training & evaluation entry point
├── test.py             # Standalone evaluation script
├── requirements.txt    # Python dependencies
├── metrics/            # Evaluation metric scripts
│   ├── compute
│   ├── corr
│   └── process
└── data/               # Sample datasets (synthetic + processed real-world)
```

## Abstract

Modeling multivariate time series by representing them as graphs, where individual series act as nodes and pairwise temporal correlations serve as edges, has gained significant traction. Recent advances in Graph Neural Networks (GNNs) have demonstrated strong performance by assuming a static graph topology and aggregating information from neighboring series. In this work, we investigate the representational power of GNNs for forecasting under both static and dynamic settings (i.e., when pairwise correlations evolve drastically over time) and identify critical limitations in current architectures. To formalize this, we first propose **Temporal Correlation Volatility (TCV)**, a model-agnostic metric designed to quantify the distributional evolution of these latent structures. We establish a clear connection between TCV and performance degradation, demonstrating that many popular models, including Transformers, generalize poorly in high-TCV settings and are often outperformed by simple structure-agnostic baselines. To address these limitations, we propose **Graph Layer for Inference in Dynamic Environments (GLIDE)**, a novel GNN layer enhanced by two theoretically grounded design mechanisms: (D1) Path-based Message Passing, which captures path-based neighborhoods, and (D2) Static and Dynamic Propagation Separation, which identifies optimal dynamics via local static approximation. These components significantly improve learning under dynamic topology while preserving robustness in static scenarios. Extensive experiments on synthetic and real-world benchmarks demonstrate that GLIDE achieves up to **85.7%** performance gains over state-of-the-art baselines.

## Citation

```bibtex
@inproceedings{shao2025glide,
  title     = {When {GNNs} Fail: Quantifying and Overcoming Temporal Correlation Volatility in Time Series},
  author    = {Shao, Chen and Wang, Yue and Zhu, Zhenyi and Huang, Zhanbo and K{\"a}fer, Tobias and Wu, Zonghan and Koutra, Danai},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
