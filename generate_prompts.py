"""
Generate textual embeddings (LLM) for the 116 regions of the AAL atlas
and save to BrainPromptGAT/data/roi_bert_embeddings.pt

This script follows the BrainPrompt paper approach:
- Uses ROI-specific descriptions (generated via ChatGPT)
- Encodes with Llama-encoder-1.0B (LLM2Vec) instead of SentenceTransformer
- Aligns dimensions with linear projection for additive injection
"""
import os
import torch
import numpy as np
import json

# Try to import Llama-encoder-1.0B (LLM2Vec)
try:
    from llm2vec import LLM2Vec
    HAS_LLM2VEC = True
except ImportError:
    HAS_LLM2VEC = False
    print("Warning: llm2vec not installed. Trying transformers directly...")
    try:
        from transformers import AutoModel, AutoTokenizer
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        print("Warning: transformers not installed. Falling back to SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            HAS_SENTENCE_TRANSFORMER = True
        except ImportError:
            HAS_SENTENCE_TRANSFORMER = False

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

    print("2) Loading ROI-specific descriptions...")
    descriptions_path = os.path.join(save_dir, "roi_descriptions.json")
    
    if os.path.exists(descriptions_path):
        print(f"   Loading from: {descriptions_path}")
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            roi_descriptions = json.load(f)
        print(f"   Loaded {len(roi_descriptions)} descriptions")
    else:
        print(f"   Warning: {descriptions_path} not found!")
        print("   Falling back to generic template. Run generate_roi_descriptions.py first for better results.")
        roi_descriptions = None
    
    print("3) Building text prompts for each ROI...")
    text_prompts = []
    for label in labels:
        if roi_descriptions and label in roi_descriptions:
            # Use specific description from ChatGPT
            prompt = roi_descriptions[label]
        else:
            # Fallback to generic template
            clean = label.replace("_", " ")
            prompt = (
                f"The brain region {clean} is associated with functional connectivity "
                f"in autism spectrum disorder."
            )
        text_prompts.append(prompt)
    
    print(f"   Example ({labels[0]}): {text_prompts[0][:80]}...")

    print("4) Encoding text with Llama-encoder-1.0B (LLM2Vec)...")
    
    if HAS_LLM2VEC:
        # Use LLM2Vec (recommended approach from paper)
        print("   Using llm2vec package...")
        model = LLM2Vec.from_pretrained("knowledgator/Llama-encoder-1.0B")
        embeddings = model.encode(text_prompts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    elif HAS_TRANSFORMERS:
        # Fallback: Use transformers directly
        print("   Using transformers package directly...")
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
        model = AutoModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
        model.eval()  # Set to evaluation mode
        
        # Encode each prompt
        embeddings_list = []
        with torch.no_grad():
            for prompt in text_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                # Use mean pooling over sequence length
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings_list.append(embedding.numpy())
        
        embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32)
    elif HAS_SENTENCE_TRANSFORMER:
        # Final fallback: Use SentenceTransformer (not ideal, but works)
        print("   Warning: Using SentenceTransformer as fallback (not matching paper)")
        print("   Install llm2vec or transformers for proper implementation")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(text_prompts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        raise ImportError(
            "No suitable encoder found. Install one of:\n"
            "  - llm2vec: pip install llm2vec\n"
            "  - transformers: pip install transformers\n"
            "  - sentence-transformers: pip install sentence-transformers"
        )
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Expected shape: (116, {embeddings.shape[1]})")

    torch.save(embeddings, save_path)
    print(f"5) Saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    path = generate_roi_embeddings()
    print(f"Concluído. Arquivo: {path}")
