import pandas as pd
import ast
import torch
from transformers import AutoModel
from tqdm import tqdm
import numpy as np
import os
import faiss

PATHOLOGY_PROMPT_TEMPLATES = [
    lambda c: f"a histopathology slide of {c}",
    lambda c: f"microscopic view showing {c}",
    lambda c: f"H&E stain of {c}",
    lambda c: f"a region of interest featuring {c}",
    lambda c: f"tissue sample with presence of {c}",
    lambda c: f"cellular features characteristic of {c}",
    lambda c: f"pathological findings include {c}",
]

DATASET_CONFIG = {
    "CAMELYON16": {
        "keywords": ['breast', 'lymph', 'lymph node', 'metastatic carcinoma'],
    }
}

dataset_name = 'CAMELYON16'
data_path = '/path/to/quilt_1M_lookup.csv'

df = pd.read_csv(data_path)
df.dropna(subset=['roi_text'], inplace=True)
df = df[df['roi_text'] != '[]']

print(f"Original number of rows before filtering: {len(df)}")

def contains_keywords(roi_text_as_string, keywords):
    try:
        concepts = ast.literal_eval(roi_text_as_string)
        if not isinstance(concepts, list):
            return False
        for concept in concepts:
            if isinstance(concept, str):
                lower_concept = concept.lower()
                if any(keyword in lower_concept for keyword in keywords):
                    return True
    except (ValueError, SyntaxError):
        return False
    return False

keywords_to_filter = DATASET_CONFIG[dataset_name]['keywords']
mask = df['roi_text'].apply(lambda text: contains_keywords(text, keywords_to_filter))
df = df[mask].copy()

print(f"Number of rows after filtering for {keywords_to_filter}': {len(df)}")

print("Extracting unique concepts...")
all_concepts_series = (
    df['roi_text']
    .apply(ast.literal_eval)
    .explode()
    .unique()
)
all_concepts = [concept for concept in all_concepts_series if concept]
print(f"Found {len(all_concepts)} initial unique concepts.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

print("Loading model...")
titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
text_encoder = titan.text_encoder.to(device)
tokenizer = titan.text_encoder.tokenizer
print("Model loaded successfully.")

texts_to_encode = all_concepts
batch_size = 128
n_clusters = 1000
ensembled_embeddings = None

if texts_to_encode:
    print("\n--- Starting Text Encoding with Prompt Ensembling ---")
    for i, template in enumerate(PATHOLOGY_PROMPT_TEMPLATES):
        print(f"Encoding with template {i + 1}/{len(PATHOLOGY_PROMPT_TEMPLATES)}: '{template('concept')}'")

        prompted_texts = [template(concept) for concept in texts_to_encode]

        template_embeddings_list = []
        for j in tqdm(range(0, len(prompted_texts), batch_size), desc=f"Template {i + 1}"):
            batch_texts = prompted_texts[j:j + batch_size]
            tokenized_batch = tokenizer(batch_texts).to(device)

            with torch.no_grad():
                batch_embeddings = titan.encode_text(tokenized_batch)

            template_embeddings_list.append(batch_embeddings.cpu())

        current_template_embeddings = torch.cat(template_embeddings_list, dim=0)

        if ensembled_embeddings is None:
            ensembled_embeddings = torch.zeros_like(current_template_embeddings)
        ensembled_embeddings += current_template_embeddings

    text_embeddings = ensembled_embeddings / len(PATHOLOGY_PROMPT_TEMPLATES)

    print("\nEncoding and ensembling finished.")
    print(f"Shape of the final ensembled embeddings tensor: {text_embeddings.shape}")

    print("\n--- Starting K-Means Clustering to find concept centers ---")

    n_original_concepts, d = text_embeddings.shape

    if n_original_concepts < n_clusters:
        print(
            f"Warning: Number of unique concepts ({n_original_concepts}) is less than desired clusters ({n_clusters}).")
        n_clusters = n_original_concepts

    embeddings_np = text_embeddings.numpy().astype('float32')

    print("Normalizing embeddings for spherical clustering and search...")
    faiss.normalize_L2(embeddings_np)

    print(f"Running K-Means to create {n_clusters} clusters...")
    kmeans = faiss.Kmeans(d=d, k=n_clusters, gpu=True, spherical=True, niter=300, min_points_per_centroid=50, nredo=10, verbose=True)
    kmeans.train(embeddings_np)

    centroid_embeddings = torch.from_numpy(kmeans.centroids)
    print(f"Successfully generated {len(centroid_embeddings)} centroid embeddings.")

    torch.save(centroid_embeddings, f'concepts/text_embeddings_{dataset_name}.pt')

else:
    print("\nNo concepts found to encode after filtering. Process finished.")