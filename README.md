# GLIDE: Graph Layer for Inference in Dynamic Environments

Official PyTorch implementation of the ECML-PKDD 2025 paper:

> **When GNNs Fail: Quantifying and Overcoming Temporal Correlation Volatility in Time Series**
>
> Chen Shao¹, Yue Wang¹, Zhenyi Zhu², Zhanbo Huang¹, Tobias Käfer¹, Zonghan Wu³, Danai Koutra⁴ ✉
>
> ¹ Karlsruhe Institute of Technology · ² HKUST · ³ East China Normal University · ⁴ University of Michigan

📄 **Paper sources:** [github.com/ChenS676/PKDD-When-GNNs-Fail-Quantifying-and-Overcoming-Temporal-Correlation-Volatility-in-Time-Series](https://github.com/ChenS676/PKDD-When-GNNs-Fail-Quantifying-and-Overcoming-Temporal-Correlation-Volatility-in-Time-Series)

> 🚧 **Code release in progress.** The full implementation and reproduction scripts will be uploaded shortly. Star/watch the repo to be notified.

## Abstract

Modeling multivariate time series by representing them as graphs, where individual series act as nodes and pairwise temporal correlations serve as edges, has gained significant traction. Recent advances in Graph Neural Networks (GNNs) have demonstrated strong performance by assuming a static graph topology and aggregating information from neighboring series. In this work, we investigate the representational power of GNNs for forecasting under both static and dynamic settings (i.e., when pairwise correlations evolve drastically over time) and identify critical limitations in current architectures. To formalize this, we first propose **Temporal Correlation Volatility (TCV)**, a model-agnostic metric designed to quantify the distributional evolution of these latent structures. We establish a clear connection between TCV and performance degradation, demonstrating that many popular models, including Transformers, generalize poorly in high-TCV settings and are often outperformed by simple structure-agnostic baselines. To address these limitations, we propose **Graph Layer for Inference in Dynamic Environments (GLIDE)**, a novel GNN layer enhanced by two theoretically grounded design mechanisms: (D1) Path-based Message Passing, which captures path-based neighborhoods, and (D2) Static and Dynamic Propagation Separation, which identifies optimal dynamics via local static approximation. These components significantly improve learning under dynamic topology while preserving robustness in static scenarios. Extensive experiments on synthetic and real-world benchmarks demonstrate that GLIDE achieves up to **85.7%** performance gains over state-of-the-art baselines.

## Citation

```bibtex
@inproceedings{shao2025glide,
  title     = {When {GNNs} Fail: Quantifying and Overcoming Temporal Correlation Volatility in Time Series},
  author    = {Shao, Chen and Wang, Yue and Zhu, Zhenyi and Huang, Zhanbo and K{\"a}fer, Tobias and Wu, Zonghan and Koutra, Danai},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2025)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
