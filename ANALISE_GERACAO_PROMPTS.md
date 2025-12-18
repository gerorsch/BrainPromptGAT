# Análise: Geração de ROI Prompts - Comparação com Paper Original

## Resumo Executivo

Após análise da seção 3.1 do paper original (2504.16096v1.pdf), foram identificadas **diferenças significativas** entre o método descrito no paper e a implementação atual do script `generate_prompts.py`.

---

## 📋 O que o Paper Original Descreve (Seção 3.1)

### Método de Geração de ROI Prompts:

1. **Geração via ChatGPT**:
   - Os autores usam **ChatGPT** para gerar descrições textuais de cada ROI
   - Query enviada ao ChatGPT:
     ```
     "Given the ROI labels for AAL116 atlas, generate a sentence to describe 
     each of them by the given order: Precentral_L, Precentral_R, Frontal_Sup_L ..."
     ```
   - O output são **descrições detalhadas** das características estruturais e funcionais de cada região
   - Exemplo mostrado na Fig. 1 do paper:
     - `Precentral_L: The left precentral gyrus, associated with motor control and planning.`
     - `Precentral_R: The right precentral gyrus, involved in executing voluntary motor movements.`
     - `Frontal_Sup_L: The left superior frontal gyrus, plays a role in self-awareness and cognitive control.`

2. **Text Encoder**:
   - Usam **Llama-encoder-1.0B** do LLM2Vec [1]
   - Link: `https://huggingface.co/knowledgator/Llama-encoder-1.0B`
   - O encoder é **congelado** durante o treinamento

3. **Integração no Modelo**:
   - Os embeddings são projetados para alinhar com a dimensão oculta do GNN
   - Fórmula: `h^(l)_v = GNN^(l-1)_r(h^(l-1)_v + Enc(p^ROI_v))`
   - O prompt é **somado** às features do nó antes de passar pelo GNN

---

## 🔍 O que Nossa Implementação Faz

### Script: `generate_prompts.py`

1. **Geração de Prompts**:
   - ❌ **NÃO usa ChatGPT** para gerar descrições
   - ✅ Usa um **template fixo** para todos os ROIs:
     ```python
     prompt = (
         f"The brain region {clean} is associated with functional connectivity "
         f"in autism spectrum disorder."
     )
     ```
   - Exemplo gerado: `"The brain region Precentral L is associated with functional connectivity in autism spectrum disorder."`

2. **Text Encoder**:
   - ❌ Usa **SentenceTransformer** com modelo `all-MiniLM-L6-v2`
   - ❌ **NÃO usa** Llama-encoder-1.0B como especificado no paper

3. **Atlas e ROIs**:
   - ✅ Usa atlas AAL com 116 ROIs (correto)
   - ✅ Carrega labels do nilearn ou lista offline

---

## ⚠️ Diferenças Identificadas

| Aspecto | Paper Original | Nossa Implementação | Status |
|---------|----------------|---------------------|--------|
| **Geração de Texto** | ChatGPT com descrições detalhadas | Template fixo genérico | ❌ **DIFERENTE** |
| **Text Encoder** | Llama-encoder-1.0B (LLM2Vec) | all-MiniLM-L6-v2 (SentenceTransformer) | ❌ **DIFERENTE** |
| **Conteúdo do Prompt** | Descrições estruturais/funcionais específicas | Template genérico sobre ASD | ❌ **DIFERENTE** |
| **Atlas** | AAL116 (116 ROIs) | AAL116 (116 ROIs) | ✅ **CORRETO** |
| **Congelamento do Encoder** | Sim (congelado) | Sim (não treinado) | ✅ **CORRETO** |

---

## 🎯 Impacto das Diferenças

### 1. **Template Fixo vs. Descrições do ChatGPT**

**Impacto**: ⚠️ **ALTO**

- O paper usa descrições **específicas e detalhadas** de cada ROI (ex: "motor control and planning", "self-awareness and cognitive control")
- Nossa implementação usa um template **genérico** que não captura as características únicas de cada região
- Isso pode limitar a capacidade do modelo de distinguir entre diferentes ROIs baseado em conhecimento semântico

### 2. **Modelo de Encoder Diferente**

**Impacto**: ⚠️ **MÉDIO**

- `Llama-encoder-1.0B` é um modelo maior e mais poderoso (1 bilhão de parâmetros)
- `all-MiniLM-L6-v2` é um modelo menor e mais eficiente (22.7M parâmetros)
- A qualidade dos embeddings pode ser diferente, mas ambos são modelos de sentence embeddings válidos

### 3. **Falta de Descrições Específicas**

**Impacto**: ⚠️ **ALTO**

- As descrições do ChatGPT fornecem **conhecimento médico específico** sobre cada ROI
- O template genérico não aproveita esse conhecimento semântico rico
- Isso pode reduzir a capacidade do modelo de incorporar conhecimento externo (um dos objetivos principais do BrainPrompt)

---

## ✅ Recomendações para Alinhamento com o Paper

### Opção 1: Implementação Fiel ao Paper (Recomendado)

1. **Gerar descrições via ChatGPT/LLM**:
   - Criar um script que gera descrições específicas para cada ROI usando ChatGPT ou outro LLM
   - Salvar as descrições em um arquivo JSON/CSV
   - Exemplo de estrutura:
     ```json
     {
       "Precentral_L": "The left precentral gyrus, associated with motor control and planning.",
       "Precentral_R": "The right precentral gyrus, involved in executing voluntary motor movements.",
       ...
     }
     ```

2. **Usar Llama-encoder-1.0B**:
   - Substituir SentenceTransformer por Llama-encoder-1.0B
   - Instalar: `pip install llm2vec` ou usar diretamente do HuggingFace
   - Código sugerido:
     ```python
     from llm2vec import LLM2Vec
     model = LLM2Vec.from_pretrained("knowledgator/Llama-encoder-1.0B")
     ```

3. **Atualizar script de geração**:
   - Carregar descrições específicas de arquivo
   - Usar Llama-encoder-1.0B para codificar
   - Manter o resto da lógica igual

### Opção 2: Melhoria Incremental (Pragmática)

1. **Melhorar o template**:
   - Criar templates mais específicos baseados em conhecimento médico
   - Exemplo:
     ```python
     ROI_DESCRIPTIONS = {
         "Precentral_L": "motor control and planning",
         "Precentral_R": "executing voluntary motor movements",
         ...
     }
     prompt = f"The brain region {clean} is associated with {ROI_DESCRIPTIONS[label]} and functional connectivity in autism spectrum disorder."
     ```

2. **Manter SentenceTransformer** (mais prático):
   - `all-MiniLM-L6-v2` é mais leve e rápido
   - Pode ser suficiente se as descrições forem melhoradas

### Opção 3: Híbrida (Melhor dos Dois Mundos)

1. **Gerar descrições uma vez via ChatGPT/LLM** e salvar
2. **Usar SentenceTransformer** para codificar (mais eficiente)
3. **Atualizar o script** para carregar descrições específicas

---

## 📝 Exemplo de Implementação Sugerida

```python
"""
Gera embeddings textuais (LLM) para as 116 regiões do atlas AAL
Alinhado com o paper original BrainPrompt (2504.16096v1)
"""

import os
import torch
import json
from llm2vec import LLM2Vec  # ou usar transformers diretamente

# Descrições específicas geradas via ChatGPT (uma vez)
ROI_DESCRIPTIONS = {
    "Precentral_L": "The left precentral gyrus, associated with motor control and planning.",
    "Precentral_R": "The right precentral gyrus, involved in executing voluntary motor movements.",
    "Frontal_Sup_L": "The left superior frontal gyrus, plays a role in self-awareness and cognitive control.",
    # ... todas as 116 descrições
}

def generate_roi_embeddings(save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "roi_bert_embeddings.pt")
    
    # Carregar labels AAL
    labels = _aal_labels_offline()  # ou via nilearn
    
    # Construir prompts usando descrições específicas
    text_prompts = []
    for label in labels:
        if label in ROI_DESCRIPTIONS:
            prompt = ROI_DESCRIPTIONS[label]
        else:
            # Fallback para template genérico
            clean = label.replace("_", " ")
            prompt = f"The brain region {clean} is associated with functional connectivity."
        text_prompts.append(prompt)
    
    # Usar Llama-encoder-1.0B como no paper
    print("Codificando com Llama-encoder-1.0B (LLM2Vec)...")
    model = LLM2Vec.from_pretrained("knowledgator/Llama-encoder-1.0B")
    embeddings = model.encode(text_prompts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    torch.save(embeddings, save_path)
    print(f"Salvo em: {save_path}")
    return save_path
```

---

## 🔬 Próximos Passos

1. **Decisão**: Escolher entre Opção 1 (fiel ao paper), Opção 2 (pragmática) ou Opção 3 (híbrida)

2. **Se escolher Opção 1 ou 3**:
   - Gerar descrições via ChatGPT/LLM para todas as 116 ROIs
   - Salvar em arquivo JSON
   - Atualizar `generate_prompts.py`

3. **Se escolher usar Llama-encoder-1.0B**:
   - Instalar dependências: `pip install llm2vec` ou usar `transformers`
   - Atualizar código de encoding

4. **Testes**:
   - Regenerar embeddings com novo método
   - Comparar performance com embeddings atuais
   - Verificar se há melhoria na capacidade de distinguir ROIs

---

## 📚 Referências

- Paper: "BrainPrompt: Multi-Level Brain Prompt Enhancement for Neurological Condition Identification" (2504.16096v1)
- Seção 3.1: Message-Passing with ROI Prompt Enhancement
- Llama-encoder-1.0B: https://huggingface.co/knowledgator/Llama-encoder-1.0B
- LLM2Vec: https://github.com/McGill-NLP/llm2vec

---

**Data da Análise**: 2025-01-17  
**Status**: ⚠️ Implementação atual difere do paper original em aspectos importantes
