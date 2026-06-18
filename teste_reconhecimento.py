from deepface import DeepFace
import json

def verificar_faces(caminho_foto1, caminho_foto2):
    print("Analisando as imagens... (Na primeira execução, o download do modelo pode demorar um pouco)")
    
    try:
        # A função verify compara as duas fotos usando o modelo 'VGG-Face' por padrão
        resultado = DeepFace.verify(
            img1_path = caminho_foto1, 
            img2_path = caminho_foto2,
            enforce_detection = True # Garante que o código vai falhar se não achar um rosto
        )
        
        # O resultado é um dicionário com vários dados. Vamos formatar para ficar bonito.
        mesma_pessoa = resultado["verified"]
        distancia = resultado["distance"]
        limiar = resultado["threshold"]
        
        print("\n=== RESULTADO DA ANÁLISE ===")
        if mesma_pessoa:
            print(f"✅ É a mesma pessoa! (Confiança alta: distância de {distancia:.4f} contra um limite de {limiar})")
        else:
            print(f"❌ São pessoas diferentes. (Distância de {distancia:.4f}, o limite era {limiar})")
            
        # Exibe o JSON completo retornado pela biblioteca para você conhecer os dados
        print("\nDados detalhados do modelo:")
        print(json.dumps(resultado, indent=4))
        
    except ValueError as e:
        print(f"⚠️ Erro: Não foi possível detectar um rosto em uma das imagens. Detalhes: {e}")
    except Exception as e:
        print(f"💥 Ocorreu um erro inesperado: {e}")

# Executa o teste apontando para os seus arquivos locais
if __name__ == "__main__":
    # Substitua pelos nomes dos seus arquivos de teste
    verificar_faces("foto1.jpg", "foto2.jpg")