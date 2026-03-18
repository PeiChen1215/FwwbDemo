# TabularS3L Transformer DAE

## Goal

This experiment upgrades the previous MLP-based TabularS3L DAE pipeline to a Transformer-backed pipeline.

Current route:

- `FTEmbeddingConfig(required_token_dim=2)`
- `TransformerBackboneConfig`
- `DAEConfig`
- local `TabularS3L` source code

## Script

- `train_tabulars3l_dae_cluster_transformer.py`

## Key Difference From The MLP Version

MLP version:

- `IdentityEmbeddingConfig`
- `MLPBackboneConfig`

Transformer version:

- `FTEmbeddingConfig`
- `TransformerBackboneConfig`

So this is a real backbone upgrade, not only a parameter tweak.

## Outputs

- `transformer_semester_embeddings.csv`
- `transformer_student_last_semester_clusters.csv`
- `transformer_student_cluster_summary.csv`
- `transformer_inlier_cluster_summary.csv`
- `transformer_cluster_k_search.csv`
- `transformer_student_cluster_pca.png`
- `transformer_student_cluster_pca_zoom.png`
- `transformer_student_risk_pca.png`
- `transformer_outlier_pca.png`
- `transformer_student_cluster_tsne.png`
- `transformer_top_outliers.csv`
- `transformer_run_summary.json`

## Run

```bash
python train_tabulars3l_dae_cluster_transformer.py
```

## What To Compare

Compare this folder with:

- `prepared/06_tabulars3l`

Focus on:

- cluster size balance
- silhouette score
- outlier count
- cluster risk gap
- transition interpretability
