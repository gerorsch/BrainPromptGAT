"""
Script para configurar e baixar dados do ABIDE
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brainprompt_dir = os.path.join(base_dir, 'BrainPrompt')

def check_abide_data():
    """Verifica se os dados do ABIDE foram baixados"""
    possible_paths = [
        os.path.join(base_dir, 'ABIDE-1035'),
        os.path.join(base_dir, 'ABIDE-871'),
        os.path.join(brainprompt_dir, 'ABIDE-1035'),
        os.path.join(brainprompt_dir, 'ABIDE-871'),
        os.path.join(brainprompt_dir, './/ABIDE-1035'),
    ]
    
    print("Verificando dados do ABIDE...\n")
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Dados encontrados em: {path}")
            # Verificar se tem os arquivos necessários
            pcp_path = os.path.join(path, 'ABIDE_pcp', 'cpac', 'filt_global')
            if os.path.exists(pcp_path):
                files = [f for f in os.listdir(pcp_path) if f.endswith('.1D')]
                print(f"  Arquivos .1D encontrados: {len(files)}")
                return True, path
            else:
                print(f"  ⚠ Diretório filt_global não encontrado")
    
    print("✗ Dados do ABIDE não foram encontrados")
    return False, None


def download_abide_data():
    """Baixa dados do ABIDE usando nilearn"""
    print("\n" + "="*60)
    print("BAIXANDO DADOS DO ABIDE")
    print("="*60)
    print("\nIsso pode levar bastante tempo (vários GB de dados)...")
    print("Por favor, aguarde...\n")
    
    try:
        from nilearn import datasets
        
        # Tentar baixar ABIDE-1035 (mais dados)
        print("Baixando ABIDE-1035 (1035 sujeitos)...")
        abide_dataset = datasets.fetch_abide_pcp(
            'ABIDE-1035/',
            derivatives=['rois_aal'],
            pipeline='cpac',
            band_pass_filtering=True,
            global_signal_regression=True,
            quality_checked=False
        )
        
        print("\n✓ Download concluído!")
        print(f"Dados salvos em: {abide_dataset['description']}")
        return True
        
    except Exception as e:
        print(f"\n✗ Erro no download: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Configurar dados do ABIDE')
    parser.add_argument('--download', action='store_true',
                       help='Baixar dados do ABIDE se não existirem')
    parser.add_argument('--check-only', action='store_true',
                       help='Apenas verificar, não baixar')
    
    args = parser.parse_args()
    
    # Verificar dados
    exists, path = check_abide_data()
    
    if exists:
        print(f"\n✓ Dados do ABIDE estão disponíveis!")
        print(f"Você pode executar BrainPrompt/data_process.py agora.")
        return
    
    # Dados não existem
    print("\n" + "="*60)
    print("DADOS DO ABIDE NÃO ENCONTRADOS")
    print("="*60)
    
    if args.check_only:
        print("\nPara baixar os dados, execute:")
        print("  python setup_data.py --download")
        print("\nOu execute manualmente:")
        print("  cd BrainPrompt")
        print("  python ABIDE_download.py")
        return
    
    if args.download:
        if download_abide_data():
            print("\n✓ Dados baixados com sucesso!")
            print("Agora você pode executar:")
            print("  cd BrainPrompt")
            print("  python data_process.py")
        else:
            print("\n✗ Falha no download. Tente executar manualmente:")
            print("  cd BrainPrompt")
            print("  python ABIDE_download.py")
    else:
        print("\nPara baixar os dados automaticamente, execute:")
        print("  python setup_data.py --download")
        print("\nOu execute manualmente:")
        print("  cd BrainPrompt")
        print("  python ABIDE_download.py")


if __name__ == '__main__':
    main()

