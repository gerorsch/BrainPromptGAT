"""
Script para preparar dados para BrainPromptGAT
Verifica se os dados existem e oferece opções para gerá-los
"""
import os
import sys
import numpy as np

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

def check_data_exists(site='NYU'):
    """Verifica se os dados existem para um site específico"""
    data_dir = os.path.join(base_dir, 'data', 'correlation', site)
    
    files_to_check = {
        'vector': os.path.join(data_dir, f'{site}_15_site_X1.npy'),
        'matrix': os.path.join(data_dir, f'{site}_15_site_X_matrix.npy'),
        'mask': os.path.join(data_dir, f'{site}_15_site_X1_mask.npy'),
        'label': os.path.join(data_dir, f'{site}_15_site_Y1.npy'),
    }
    
    print(f"\nVerificando dados para site: {site}")
    print(f"Diretório: {data_dir}\n")
    
    results = {}
    for key, path in files_to_check.items():
        exists = os.path.exists(path)
        results[key] = exists
        status = "✓ EXISTE" if exists else "✗ NÃO EXISTE"
        print(f"  {key:10s}: {status}")
        if exists:
            try:
                data = np.load(path)
                print(f"            Shape: {data.shape}")
            except Exception as e:
                print(f"            Erro ao carregar: {e}")
    
    return results, data_dir


def convert_vector_to_matrix(site='NYU'):
    """Converte dados de formato vetor para matriz"""
    # Importar do diretório atual
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from graph_utils import vector_to_matrix
    
    data_dir = os.path.join(base_dir, 'data', 'correlation', site)
    vector_path = os.path.join(data_dir, f'{site}_15_site_X1.npy')
    matrix_path = os.path.join(data_dir, f'{site}_15_site_X_matrix.npy')
    
    if not os.path.exists(vector_path):
        print(f"Arquivo vetor não encontrado: {vector_path}")
        return False
    
    print(f"\nConvertendo {vector_path} para formato matriz...")
    try:
        X_vector = np.load(vector_path)
        print(f"Shape do vetor: {X_vector.shape}")
        
        X_matrix = vector_to_matrix(X_vector)
        X_matrix_np = X_matrix.numpy() if hasattr(X_matrix, 'numpy') else X_matrix
        
        # Salvar em formato matriz
        os.makedirs(data_dir, exist_ok=True)
        np.save(matrix_path, X_matrix_np)
        print(f"✓ Matriz salva em: {matrix_path}")
        print(f"  Shape da matriz: {X_matrix_np.shape}")
        return True
    except Exception as e:
        print(f"✗ Erro na conversão: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preparar dados para BrainPromptGAT')
    parser.add_argument('--site', type=str, default='NYU', 
                       choices=['NYU', 'UCLA', 'UM', 'USM'],
                       help='Site para verificar/preparar')
    parser.add_argument('--convert', action='store_true',
                       help='Converter dados de vetor para matriz se necessário')
    
    args = parser.parse_args()
    
    # Verificar dados
    results, data_dir = check_data_exists(args.site)
    
    # Se dados vetor existem mas matriz não, oferecer conversão
    if results.get('vector') and not results.get('matrix'):
        print(f"\n⚠ Dados em formato vetor existem, mas formato matriz não.")
        if args.convert:
            print("Convertendo automaticamente...")
            convert_vector_to_matrix(args.site)
        else:
            print("\nPara converter automaticamente, execute:")
            print(f"  python prepare_data.py --site {args.site} --convert")
    
    # Se nenhum dado existe
    elif not results.get('vector') and not results.get('matrix'):
        print(f"\n⚠ Nenhum dado encontrado para o site {args.site}")
        print("\nPara gerar os dados, execute:")
        print(f"  cd BrainPrompt")
        print(f"  python data_process.py")
        print("\nOu modifique BrainPrompt/data_process.py para salvar também em formato matriz.")
        print("\nApós gerar os dados, você pode converter para matriz com:")
        print(f"  python prepare_data.py --site {args.site} --convert")
    
    # Se tudo existe
    elif results.get('matrix'):
        print(f"\n✓ Todos os dados necessários existem!")
        print("Você pode executar o treinamento agora.")


if __name__ == '__main__':
    main()

