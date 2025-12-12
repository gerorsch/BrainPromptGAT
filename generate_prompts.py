"""
Gera embeddings textuais (LLM) para as 116 regiões do atlas AAL
e salva em BrainPromptGAT/data/roi_bert_embeddings.pt
"""
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from nilearn import datasets
except Exception:
    datasets = None


def _aal_labels_offline():
    # Lista das 116 regiões AAL (sem background), para evitar downloads
    return [
        "Precentral_L","Precentral_R","Frontal_Sup_L","Frontal_Sup_R","Frontal_Sup_Orb_L","Frontal_Sup_Orb_R",
        "Frontal_Mid_L","Frontal_Mid_R","Frontal_Mid_Orb_L","Frontal_Mid_Orb_R","Frontal_Inf_Oper_L",
        "Frontal_Inf_Oper_R","Frontal_Inf_Tri_L","Frontal_Inf_Tri_R","Frontal_Inf_Orb_L","Frontal_Inf_Orb_R",
        "Rolandic_Oper_L","Rolandic_Oper_R","Supp_Motor_Area_L","Supp_Motor_Area_R","Olfactory_L","Olfactory_R",
        "Frontal_Sup_Medial_L","Frontal_Sup_Medial_R","Frontal_Med_Orb_L","Frontal_Med_Orb_R","Rectus_L","Rectus_R",
        "Insula_L","Insula_R","Cingulum_Ant_L","Cingulum_Ant_R","Cingulum_Mid_L","Cingulum_Mid_R","Cingulum_Post_L",
        "Cingulum_Post_R","Hippocampus_L","Hippocampus_R","ParaHippocampal_L","ParaHippocampal_R","Amygdala_L",
        "Amygdala_R","Calcarine_L","Calcarine_R","Cuneus_L","Cuneus_R","Lingual_L","Lingual_R","Occipital_Sup_L",
        "Occipital_Sup_R","Occipital_Mid_L","Occipital_Mid_R","Occipital_Inf_L","Occipital_Inf_R","Fusiform_L",
        "Fusiform_R","Postcentral_L","Postcentral_R","Parietal_Sup_L","Parietal_Sup_R","Parietal_Inf_L",
        "Parietal_Inf_R","SupraMarginal_L","SupraMarginal_R","Angular_L","Angular_R","Precuneus_L","Precuneus_R",
        "Paracentral_Lobule_L","Paracentral_Lobule_R","Caudate_L","Caudate_R","Putamen_L","Putamen_R","Pallidum_L",
        "Pallidum_R","Thalamus_L","Thalamus_R","Heschl_L","Heschl_R","Temporal_Sup_L","Temporal_Sup_R",
        "Temporal_Pole_Sup_L","Temporal_Pole_Sup_R","Temporal_Mid_L","Temporal_Mid_R","Temporal_Pole_Mid_L",
        "Temporal_Pole_Mid_R","Temporal_Inf_L","Temporal_Inf_R","Cerebellum_Crus1_L","Cerebellum_Crus1_R",
        "Cerebellum_Crus2_L","Cerebellum_Crus2_R","Cerebellum_3_L","Cerebellum_3_R","Cerebellum_4_5_L",
        "Cerebellum_4_5_R","Cerebellum_6_L","Cerebellum_6_R","Cerebellum_7b_L","Cerebellum_7b_R","Cerebellum_8_L",
        "Cerebellum_8_R","Cerebellum_9_L","Cerebellum_9_R","Cerebellum_10_L","Cerebellum_10_R","Vermis_1_2",
        "Vermis_3","Vermis_4_5","Vermis_6","Vermis_7","Vermis_8","Vermis_9","Vermis_10"
    ]


def generate_roi_embeddings(save_dir=None, allow_download=True):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "roi_bert_embeddings.pt")

    labels = None
    if datasets is not None and allow_download:
        try:
            print("1) Carregando atlas AAL (nilearn.datasets.fetch_atlas_aal)...")
            dataset = datasets.fetch_atlas_aal(version="SPM12")
            labels = list(dataset.labels)
            if len(labels) > 116:
                labels = labels[:116]
            print(f"   Encontradas {len(labels)} regiões via nilearn.")
        except Exception as e:
            print(f"! Falha ao baixar atlas AAL ({e}). Usando lista offline embutida.")
            labels = _aal_labels_offline()
    else:
        print("1) nilearn indisponível ou download desabilitado. Usando lista offline embutida.")
        labels = _aal_labels_offline()

    print("2) Construindo prompts textuais para cada ROI...")
    text_prompts = []
    for label in labels:
        clean = label.replace("_", " ")
        prompt = (
            f"The brain region {clean} is associated with functional connectivity "
            f"in autism spectrum disorder."
        )
        text_prompts.append(prompt)
    print(f"   Exemplo: {text_prompts[0]}")

    print("3) Codificando texto com SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_prompts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    print(f"   Shape dos embeddings: {embeddings.shape}")

    torch.save(embeddings, save_path)
    print(f"4) Salvo em: {save_path}")
    return save_path


if __name__ == "__main__":
    path = generate_roi_embeddings()
    print(f"Concluído. Arquivo: {path}")
