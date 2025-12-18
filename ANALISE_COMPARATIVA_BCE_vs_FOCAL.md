# Análise Comparativa: BCE Loss vs Focal Loss

## Resultados Observados com BCE Loss

### Métricas Finais (Fold 5)
- **Final Accuracy**: 61.49%
- **Melhor Época**: ~45 (Acc: 67.66%, Recall: 67.96%, F1: 0.6829)
- **Época de Early Stopping**: 65 (patience=10)

### Evolução das Métricas (Épocas 40-60)

| Época | Loss Train | Acc Train | Acc Test | Precision | Recall | F1 | AUC | Sensitivity | Specificity |
|-------|-----------|-----------|----------|-----------|--------|-----|-----|-------------|--------------|
| 40    | 0.0393    | 83.59%    | 65.17%   | 0.6988    | 0.5631 | 0.6237 | 0.654 | 0.5631 | 0.7449 |
| 45    | 0.0436    | 83.75%    | **67.66%** | 0.6863    | **0.6796** | **0.6829** | **0.6765** | **0.6796** | 0.6735 |
| 50    | 0.1198    | 86.84%    | 65.17%   | 0.7143    | 0.534  | 0.6111 | 0.6547 | 0.534  | 0.7755 |
| 55    | 0.0291    | 86.22%    | 64.18%   | 0.7067    | 0.5146 | 0.5955 | 0.645  | 0.5146 | 0.7755 |
| 60    | 0.0306    | 85.60%    | 62.19%   | 0.6901    | 0.4757 | 0.5632 | 0.6256 | 0.4757 | 0.7755 |

## Problemas Identificados com BCE Loss

### 1. **Overfitting Severo**
- **Gap Treino-Teste**: Aumenta de ~16% (época 40) para ~27% (época 60)
- **Acc Treino**: Cresce continuamente (83.59% → 89.32%)
- **Acc Teste**: Degrada após época 45 (67.66% → 62.19%)

### 2. **Degradação de Recall/Sensitivity**
- **Pico na Época 45**: 67.96% (melhor balanceamento)
- **Degradação Progressiva**: 67.96% → 47.57% (queda de 20 pontos percentuais)
- **Especificidade Alta**: Mantém ~77.55% (modelo tende a prever classe negativa)

### 3. **Instabilidade de Treinamento**
- **Loss Train Flutuante**: 0.0393 → 0.1198 → 0.0291 (variação grande)
- **Early Stopping Precoce**: Parou na época 65, mas melhor época foi ~45

## Comparação Esperada: Focal Loss vs BCE Loss

### Vantagens Esperadas do Focal Loss

1. **Melhor Generalização**
   - Focal Loss foca em exemplos difíceis, reduzindo overfitting
   - Menor gap entre treino e teste esperado

2. **Melhor Recall em Classes Minoritárias**
   - Focal Loss com `gamma > 0` dá mais peso a exemplos difíceis
   - Deve manter melhor balanceamento entre classes

3. **Treinamento Mais Estável**
   - Menos flutuações no loss durante treinamento
   - Convergência mais suave

4. **Melhor Performance Final**
   - Espera-se accuracy final > 65% com Focal Loss
   - Melhor F1 score e AUC

### Desvantagens Potenciais do Focal Loss

1. **Hiperparâmetros Adicionais**
   - Requer ajuste de `gamma` e `alpha`
   - Pode ser mais sensível a configurações

2. **Treinamento Mais Lento**
   - Foco em exemplos difíceis pode tornar treinamento mais lento

## Recomendações

### Para BCE Loss (Melhorias Necessárias)

1. **Aumentar Regularização**
   ```python
   --weight_decay 0.001  # Aumentar de 0.0
   --dropout 0.5  # Se ainda não estiver usando
   ```

2. **Ajustar Learning Rate**
   ```python
   --lr 0.0001  # Reduzir learning rate
   ```

3. **Early Stopping Mais Agressivo**
   ```python
   --early_stopping_patience 5  # Reduzir de 10 para 5
   ```

4. **Salvar Melhor Modelo**
   - Implementar checkpoint na melhor época (45)
   - Não usar modelo final, mas melhor validação

### Para Comparação Justa

1. **Mesmos Hiperparâmetros**
   - Usar mesma configuração para ambas as losses
   - Apenas variar `--loss_type`

2. **Múltiplas Execuções**
   - Executar 3-5 vezes cada loss
   - Comparar médias e desvios padrão

3. **Métricas de Interesse**
   - **Accuracy Final**: Focal Loss deve ser superior
   - **Recall/Sensitivity**: Focal Loss deve manter melhor
   - **F1 Score**: Focal Loss deve ser mais balanceado
   - **Estabilidade**: Focal Loss deve ter menor variância

## Próximos Passos

1. **Executar com Focal Loss** usando mesma configuração
2. **Comparar métricas finais** lado a lado
3. **Analisar curvas de treinamento** (loss, accuracy, recall)
4. **Decidir qual loss usar** baseado em:
   - Accuracy final
   - Balanceamento (Recall vs Precision)
   - Estabilidade do treinamento
   - Generalização (gap treino-teste)

## Conclusão Preliminar

Com base nos resultados observados, **BCE Loss está apresentando overfitting significativo** e **degradação de recall**. Espera-se que **Focal Loss**:
- Mantenha melhor generalização
- Preserve melhor o recall/sensitivity
- Produza accuracy final superior a 65%

**Recomendação**: Testar Focal Loss com mesma configuração e comparar resultados.
