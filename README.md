# CatBoost vs. LightGBM Benchmark Study

This repository contains the **official implementation code** used in my research paper titled:  
*"Comparative Analysis of CatBoost vs. LightGBM: A Benchmark Study"*

## Research Overview
We conduct a comprehensive empirical comparison of gradient boosting frameworks across five distinct ML tasks:

- Binary Classification  
- Regression Analysis  
- Imbalanced Data Classification  
- Multiclass Classification  
- Medium-scale Mixed Data  

## Dependencies
```bash
pip install catboost==1.2.2 lightgbm==4.1.0 scikit-learn==1.4.0 pandas==2.1.4 numpy==1.26.0 matplotlib==3.8.0 seaborn==0.13.0 imbalanced-learn==0.11.0
```

### Benchmark Tasks & Datasets

| Task Type              | Dataset               | Key Characteristics          | Source |
|------------------------|-----------------------|------------------------------|--------|
| **Binary Classification** | Titanic             | Balanced classes (60KB)      | [Kaggle](https://www.kaggle.com/c/titanic/data) |
| **Regression**           | California Housing  | 20k samples, 8 features (1MB)| [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| **Imbalanced Binary**    | Credit Card Fraud   | 1:500 class ratio (147MB)    | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| **Multiclass**           | Wine Quality        | 11 quality classes (1MB)     | [UCI](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| **Medium-scale Binary**  | Telco Churn         | Mixed categorical/numerical (1MB)| [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) |

## Hardware Environment

**Test System Specifications**  
- **Processor**: AMD Ryzen 7 5800U (8 cores/16 threads, 1.9GHz base/4.4GHz boost)  
- **Discrete GPU**: NVIDIA RTX 3050 Ti Laptop GPU (4GB GDDR6 VRAM)  
- **Integrated GPU**: AMD Radeon Graphics (Vega 8)  
- **Memory**: 16GB DDR4 (15.4GB usable)  
- **OS**: Windows 11 64-bit  

### Framework-Specific Configuration  
```python
# CatBoost GPU Setup
cat_gpu_params = {
    'task_type': 'GPU',
    'devices': '0:1'  # Uses RTX 3050 Ti
}

# LightGBM GPU Setup
lgb_gpu_params = {
    'device': 'gpu',
    'gpu_platform_id': 0,  # NVIDIA
    'gpu_device_id': 0     # Primary GPU
}

```
## Usage
1. Download datasets from the table above
2. Place them in the root directory
3. Launch Jupyter:
   ```bash
   jupyter notebook


