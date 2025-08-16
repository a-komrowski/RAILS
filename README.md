# RAILS - Railway Infrastructure Image Analysis & Clustering System

A comprehensive machine learning pipeline for analyzing and clustering railway infrastructure images from multiple European transportation companies. This project performs unsupervised clustering analysis on center track images to identify patterns, infrastructure types, and tenant-specific characteristics.

## 🚂 Project Overview

The RAILS project is designed to analyze railway track images collected from various European public transportation operators. Using deep learning feature extraction and unsupervised clustering techniques, it automatically discovers patterns and groups similar infrastructure types without prior labeling.

### Key Features

- **Multi-Tenant Data Processing**: Handles images from 8+ European transportation companies (AVA, BernMobil, BVB, CTS, Gent, GVB, RETM, VBZ)
- **Automated Image Downloading**: Downloads center track images (`_C.png`) from AWS S3 buckets
- **Smart Dataset Sampling**: Creates balanced datasets with equal representation from all tenants
- **Deep Learning Feature Extraction**: Uses pre-trained ResNet50 for robust image feature extraction
- **Advanced Clustering**: Implements both K-Means and DBSCAN clustering algorithms
- **Comprehensive Visualization**: Generates t-SNE plots, cluster examples, and statistical analyses
- **Automated Organization**: Organizes clustered images into directory structures for further analysis

## 📊 Dataset Information

The project works with railway center track images from the following transportation operators:

| Tenant | Company | Region | Images Available |
|--------|---------|--------|------------------|
| AVA | Verkehrsbetriebe Bern | Switzerland | ~22k |
| BernMobil | Bern Public Transport | Switzerland | ~16k |
| BVB | Basler Verkehrs-Betriebe | Switzerland | ~27k |
| CTS | Compagnie des Transports Strasbourgeois | France | ~31k |
| Gent | De Lijn Gent | Belgium | ~23k |
| GVB | Gemeentevervoerbedrijf | Netherlands | ~19k |
| RETM | Régie des Transports Métropolitains | France | ~65k |
| VBZ | Verkehrsbetriebe Zürich | Switzerland | ~46k |

**Total Dataset**: 250,000+ images across multiple infrastructure types and environmental conditions.

## 🛠️ Architecture & Workflow

### 1. Data Acquisition (`download_c_images.ipynb`)
- Connects to AWS S3 buckets using boto3
- Downloads center track images based on tenant/SID ranges defined in `img_folders.json`
- Applies sampling strategies to manage dataset size
- Implements robust error handling and progress tracking
- Renames files to flat structure: `{tenant}_{SID}_{filename}_C.png`

### 2. Dataset Sampling (`sample_dataset.ipynb`)
- Creates balanced samples with equal tenant representation
- Resizes images to standard dimensions (224x224 pixels)
- Generates multiple sample sizes (100, 1K, 5K, 10K images)
- Maintains reproducibility with fixed random seeds
- Validates sample distribution and image quality

### 3. Clustering Analysis (`clustering_analysis.ipynb`)
- **Feature Extraction**: Uses pre-trained ResNet50 (ImageNet weights) for deep feature extraction
- **Preprocessing**: L2 normalization and optional PCA dimensionality reduction
- **K-Means Clustering**: Evaluates optimal number of clusters using elbow method and silhouette analysis
- **DBSCAN Clustering**: Density-based clustering for noise detection and irregular cluster shapes
- **Visualization**: t-SNE embeddings for 2D cluster visualization
- **Analysis**: Tenant distribution analysis and cluster interpretation

## 📁 Project Structure

```
RAILS/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── img_folders.json                   # Tenant/SID range configuration
├── 
├── notebooks/
│   ├── download_c_images.ipynb        # S3 image downloading
│   ├── sample_dataset.ipynb           # Dataset sampling and preprocessing
│   └── clustering_analysis.ipynb      # Main clustering analysis
├── 
├── datasets/                          # Sampled datasets
│   ├── clustering_sample_100/         # 100 image sample
│   ├── clustering_sample_1000/        # 1K image sample
│   ├── clustering_sample_5000/        # 5K image sample
│   └── clustering_sample_10000/       # 10K image sample
├── 
├── downloads/
│   └── center_images/                 # Downloaded raw images
├── 
└── results/
    ├── clustering_analysis/           # Analysis results and plots
    └── clustered_images/              # Organized cluster directories
        ├── kmeans/                    # K-Means cluster results
        │   ├── cluster_0/
        │   ├── cluster_1/
        │   └── ...
        └── dbscan/                    # DBSCAN cluster results
            ├── cluster_0/
            ├── noise/
            └── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- AWS credentials configured for S3 access
- 16GB+ RAM recommended for large-scale clustering
- GPU optional (CPU-only mode supported)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAILS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**:
   ```bash
   aws configure --profile s3user
   ```

### Usage

#### 1. Download Images
```bash
jupyter notebook download_c_images.ipynb
```
- Configure AWS profile and S3 bucket settings
- Modify `img_folders.json` for specific tenant/SID ranges
- Run notebook to download images to `downloads/center_images/`

#### 2. Create Dataset Samples
```bash
jupyter notebook sample_dataset.ipynb
```
- Set desired sample size (e.g., 5000 images)
- Creates balanced dataset in `datasets/clustering_sample_X/`
- Resizes images and validates distribution

#### 3. Run Clustering Analysis
```bash
jupyter notebook clustering_analysis.ipynb
```
- Performs feature extraction using ResNet50
- Evaluates optimal cluster numbers
- Generates visualizations and organizes results

## 🔍 Analysis Results

### Clustering Insights

The analysis typically reveals several distinct cluster types:

1. **Infrastructure Types**:
   - Ballasted tracks (gravel/stone base)
   - Embedded tracks (asphalt/concrete)
   - Different rail configurations

2. **Environmental Conditions**:
   - Lighting variations (day/night/artificial)
   - Weather conditions
   - Seasonal differences

3. **Urban vs. Rural**:
   - City center embedded tracks
   - Suburban ballasted tracks
   - Different maintenance standards

4. **Tenant-Specific Features**:
   - Company-specific infrastructure standards
   - Regional construction differences
   - Equipment and signaling variations

### Output Files

- **Cluster Visualizations**: t-SNE plots showing cluster separation
- **Cluster Examples**: Sample images from each discovered cluster
- **Statistical Reports**: JSON files with detailed cluster analysis
- **Organized Directories**: Images sorted by cluster for manual inspection

## 🎯 Applications

### Research Applications
- **Infrastructure Classification**: Automated categorization of track types
- **Maintenance Planning**: Identify infrastructure conditions requiring attention
- **Standardization Analysis**: Compare infrastructure standards across operators
- **Quality Assessment**: Detect anomalies or unusual configurations

### Practical Applications
- **Asset Management**: Inventory and categorize existing infrastructure
- **Predictive Maintenance**: Identify patterns associated with maintenance needs
- **Construction Planning**: Understand regional infrastructure variations
- **Regulatory Compliance**: Ensure infrastructure meets safety standards

## 🧠 Technical Details

### Machine Learning Pipeline

1. **Feature Extraction**: ResNet50 CNN (2048-dimensional features)
2. **Preprocessing**: L2 normalization, optional PCA reduction
3. **Clustering**: 
   - K-Means with elbow method optimization
   - DBSCAN for density-based clustering
4. **Evaluation**: Silhouette analysis, cluster size distribution
5. **Visualization**: t-SNE dimensionality reduction

### Performance Characteristics

- **Processing Speed**: ~7 images/second download rate
- **Memory Usage**: ~2GB for 5K image clustering
- **Accuracy**: Silhouette scores typically 0.3-0.7
- **Scalability**: Tested up to 250K images

## 📈 Future Enhancements

- **Supervised Learning**: Use cluster labels for classification model training
- **Real-time Processing**: Stream processing for live image analysis
- **Multi-modal Analysis**: Incorporate GPS, sensor data, and metadata
- **Anomaly Detection**: Identify unusual or dangerous track conditions
- **Temporal Analysis**: Track infrastructure changes over time

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new clustering method'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- European transportation operators for providing image data
- TensorFlow/Keras team for deep learning frameworks
- scikit-learn contributors for clustering algorithms
- AWS for cloud storage infrastructure

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the project maintainers.

---

**Note**: This project is designed for research and analysis purposes. Ensure compliance with data privacy regulations and operator agreements when working with transportation infrastructure data.