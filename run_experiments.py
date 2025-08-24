import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from skimage.feature import local_binary_pattern
from img_preprocessing import ImagePreprocessor
from sklearn.decomposition import PCA

# Configuration
DATASET_PATH = './datasets/clustering_sample_10000'
RESULTS_DIR = './experiment_results'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
N_CLUSTERS_RANGE = list(range(3, 10))
CLASSIC_LBP_P = 8
CLASSIC_LBP_R = 1
RANDOM_STATE = 42

# Define experiment grid
MODELS = {
    'resnet50': {
        'constructor': ResNet50,
        'preprocess': preprocess_resnet,
        'levels': {
            'top': {'pooling': 'avg', 'layer': None},
            'mid': {'layer': 'conv4_block6_out'},
            'low': {'layer': 'conv3_block4_out'}
        }
    },
    'vgg16': {
        'constructor': VGG16,
        'preprocess': preprocess_vgg,
        'levels': {
            'top': {'pooling': 'avg', 'layer': None},
            'mid': {'layer': 'block4_conv3'},
            'low': {'layer': 'block2_conv2'}
        }
    },
    'classic_lbp': {
        'constructor': None,
        'preprocess': None,
        'levels': {
            'full': {'layer': None}
        }
    }
}

PREPROCESSING = {
    'none': None,
    'method_1': ImagePreprocessor.method_1_histogram_equalization,
    'method_3': ImagePreprocessor.method_3_gamma_correction,
    'method_4': ImagePreprocessor.method_4_unsharp_masking
}

def create_extractor(model_name, level_cfg):
    if model_name == 'classic_lbp':
        return None, CLASSIC_LBP_P + 2
    constructor = MODELS[model_name]['constructor']
    if level_cfg['layer'] is None:
        base = constructor(weights='imagenet', include_top=False,
                           pooling=level_cfg.get('pooling', None),
                           input_shape=(*IMG_SIZE, 3))
    else:
        base_model = constructor(weights='imagenet', include_top=False,
                                 input_shape=(*IMG_SIZE, 3))
        layer = base_model.get_layer(level_cfg['layer']).output
        pooled = GlobalAveragePooling2D()(layer)
        base = Model(inputs=base_model.input, outputs=pooled)
    feat_dim = base.output_shape[-1]
    return base, feat_dim


def extract_features(paths, model, model_name, preprocess_fn, level, preproc_fn):
    features = []
    for p in paths:
        # load image
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if model_name=='classic_lbp' else cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        if preproc_fn is not None:
            img = preproc_fn(img)
        if model_name == 'classic_lbp':
            lbp = local_binary_pattern(img, CLASSIC_LBP_P, CLASSIC_LBP_R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=CLASSIC_LBP_P+2, range=(0, CLASSIC_LBP_P+2), density=True)
            features.append(hist)
        else:
            img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_arr = image.img_to_array(img_arr)
            img_arr = np.expand_dims(img_arr, 0)
            img_arr = preprocess_fn(img_arr)
            feat = model.predict(img_arr, verbose=0)
            features.append(feat.flatten())
    return np.array(features)


def evaluate_kmeans(features):
    results = []
    n_samples = features.shape[0]
    for k in N_CLUSTERS_RANGE:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels)
        db = davies_bouldin_score(features, labels)
        ch = calinski_harabasz_score(features, labels)
        composite = sil - db + np.log1p(ch)
        results.append({'k': k, 'silhouette': sil, 'davies_bouldin': db,
                        'calinski_harabasz': ch, 'composite': composite})
    return results


def run_experiment():
    # prepare paths
    img_paths = [str(p) for p in Path(DATASET_PATH).glob('*_C.png')]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(RESULTS_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = []

    for model_name, model_cfg in MODELS.items():
        for level_name, level_cfg in model_cfg['levels'].items():
            model, feat_dim = create_extractor(model_name, level_cfg)
            preprocess_fn = model_cfg['preprocess']
            for prep_name, prep_fn in PREPROCESSING.items():
                # skip classic preproc variants beyond 'none'
                if model_name=='classic_lbp' and prep_name!='none':
                    continue
                print(f"Running: {model_name} | level: {level_name} | prep: {prep_name}")
                feats = extract_features(img_paths, model, model_name, preprocess_fn, level_name, prep_fn)
                feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)

                # Apply PCA as in clustering_analysis notebook
                n_components = min(50, feat_dim)
                pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
                feats_pca = pca.fit_transform(feats_norm)
                explained_var = float(pca.explained_variance_ratio_.sum())
                features_for_clustering = feats_pca

                km_results = evaluate_kmeans(features_for_clustering)
                # pick best by composite
                best = max(km_results, key=lambda x: x['composite'])

                exp = {
                    'model': model_name,
                    'level': level_name,
                    'preprocessing': prep_name,
                    'feature_dim': int(feat_dim),
                    'n_samples': feats.shape[0],
                    'pca_components': int(n_components),
                    'pca_explained_variance': explained_var,
                    'best_k': int(best['k']),
                    'silhouette': float(best['silhouette']),
                    'davies_bouldin': float(best['davies_bouldin']),
                    'calinski_harabasz': float(best['calinski_harabasz']),
                    'composite_score': float(best['composite']),
                    'metrics_per_k': [
                        {k: v for k, v in res.items()}
                        for res in km_results
                    ]
                }
                all_experiments.append(exp)
                # save interim
                with open(out_dir / 'results.json', 'w') as f:
                    json.dump(all_experiments, f, indent=2)

    print(f"All experiments completed. Results saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    run_experiment()