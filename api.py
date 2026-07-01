from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
import os
import shutil

app = FastAPI(title="API de Reconhecimento Facial para Flutter")

# Pasta temporária para salvar os uploads que o Flutter enviar
TMP_DIR = "temp_faces"
os.makedirs(TMP_DIR, exist_ok=True)

@app.post("/comparar")
async def comparar_faces(foto1: UploadFile = File(...), foto2: UploadFile = File(...)):
    # Definindo caminhos temporários para os arquivos
    path1 = os.path.join(TMP_DIR, foto1.filename)
    path2 = os.path.join(TMP_DIR, foto2.filename)
    
    try:
        # Salva os arquivos enviados na memória para o disco rígido temporariamente
        with open(path1, "wb") as buffer:
            shutil.copyfileobj(foto1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(foto2.file, buffer)
            
        # Executa o reconhecimento facial que você testou no script anterior
        resultado = DeepFace.verify(
            img1_path = path1, 
            img2_path = path2,
            enforce_detection = False # Evita travar caso o rosto tenha acessórios/máscara
        )
        
        # Monta a resposta limpa e simplificada para o Flutter consumir
        return {
            "mesma_pessoa": bool(resultado["verified"]),
            "distancia": float(resultado["distance"]),
            "limiar_maximo": float(resultado["threshold"]),
            "modelo_utilizado": resultado["model"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")
        
    finally:
        # Boa prática: Apaga as fotos temporárias após a análise para não lotar o servidor
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

# Comando para rodar localmente se executar o arquivo direto
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)