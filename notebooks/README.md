# Notebooks

This folder contains Jupyter notebooks for development and experimentation.

## Files

### `colab_mast3r_template.ipynb`
**Google Colab template for team members**

Use this notebook as a starting point for working with MASt3R on Google Colab.

**To use:**
1. Open in Google Colab: [Open in Colab](https://colab.research.google.com/)
2. Upload this notebook or copy it to your Google Drive
3. Go to `File > Save a copy in Drive` to create your own copy
4. Follow the cells in order (1-5 for setup)

**What's included:**
- GPU check and Google Drive mount
- GitHub repository cloning
- MASt3R dependency installation
- Model loading utilities
- Pre-computed data loading helpers
- 3D visualization with Plotly

---

## Notebook Organization (Recommended)

As the team develops, add your notebooks here:

```
notebooks/
├── colab_mast3r_template.ipynb    # Starting template (CS-1)
├── ds1_matching_experiments.ipynb  # Matching algorithm dev (DS-1)
├── ds1_weight_optimization.ipynb   # Fusion weight tuning (DS-1)
├── ds2_evaluation_protocol.ipynb   # Evaluation metrics (DS-2)
├── ds2_anti_spoof_analysis.ipynb   # Anti-spoofing study (DS-2)
├── ds2_final_results.ipynb         # Final presentation figures (DS-2)
└── figures/                        # Exported figures for presentation
    ├── roc_curve.png
    ├── score_distributions.png
    └── ...
```

---

## Quick Tips

1. **Always push to GitHub** before your Colab session ends
2. **Mount Google Drive** to access shared data files
3. **Use pre-computed .npz files** from `data_shared/` to avoid running MASt3R
4. **Create your own branch** for experiments: `git checkout -b feature/ds1-experiment`
