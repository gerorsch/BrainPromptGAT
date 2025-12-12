"""
Script para modificar BrainPrompt/data_process.py e adicionar salvamento em formato matriz
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_process_path = os.path.join(base_dir, 'BrainPrompt', 'data_process.py')

def add_matrix_saving():
    """Adiciona código para salvar dados em formato matriz"""
    
    if not os.path.exists(data_process_path):
        print(f"Arquivo não encontrado: {data_process_path}")
        return False
    
    # Ler arquivo
    with open(data_process_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se já foi modificado
    if '_15_site_X_matrix.npy' in content:
        print("✓ O arquivo data_process.py já foi modificado para salvar formato matriz.")
        return True
    
    # Encontrar linha onde salva X1
    lines = content.split('\n')
    insert_index = None
    
    for i, line in enumerate(lines):
        if '_15_site_X1.npy' in line and 'np.save' in line:
            insert_index = i + 1
            break
    
    if insert_index is None:
        print("✗ Não foi possível encontrar onde inserir o código.")
        return False
    
    # Código para inserir
    insert_code = [
        "",
        "# Salvar também em formato matriz para BrainPromptGAT",
        "X_matrix = transformed_data  # Já está em formato (N, 25, 116, 116)",
        "np.save('./data/correlation/' + str(i) + '/' + str(i) + '_15_site_X_matrix.npy', X_matrix)",
        "print(f'Matrizes de correlação salvas em formato: {X_matrix.shape}')",
    ]
    
    # Inserir código
    for j, code_line in enumerate(insert_code):
        lines.insert(insert_index + j, code_line)
    
    # Salvar backup primeiro
    backup_path = data_process_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Backup criado: {backup_path}")
    
    # Salvar arquivo modificado
    new_content = '\n'.join(lines)
    with open(data_process_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✓ Arquivo modificado: {data_process_path}")
    print("  Agora ele salvará dados em formato matriz também.")
    return True


if __name__ == '__main__':
    print("Modificando BrainPrompt/data_process.py para salvar formato matriz...")
    print(f"Arquivo: {data_process_path}\n")
    
    if add_matrix_saving():
        print("\n✓ Modificação concluída!")
        print("\nPróximos passos:")
        print("1. Execute BrainPrompt/data_process.py para gerar os dados")
        print("2. Os dados serão salvos em ambos os formatos (vetor e matriz)")
        print("3. Execute BrainPromptGAT/main.py para treinar o modelo GAT")
    else:
        print("\n✗ Falha na modificação. Verifique os erros acima.")

