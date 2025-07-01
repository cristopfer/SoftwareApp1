from transformers import pipeline
import torch

class AnalizadorSentimientos:
    def __init__(self):
        # Cargar el modelo solo una vez al inicializar
        self.clasificador = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        self.traductor = pipeline(
            "translation_es_to_en",
            model="Helsinki-NLP/opus-mt-es-en",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def analizar(self, texto):
        try:
            resultado = self.clasificador(texto)[0]
            
            # Mapear las estrellas a sentimientos
            estrellas = int(resultado['label'].split()[0])
            confianza = resultado['score']
            
            if estrellas >= 4:
                sentimiento = "positivo"
            elif estrellas == 3:
                sentimiento = "neutral"
            else:
                sentimiento = "negativo"
            
            return {
                'texto': texto,
                'sentimiento': sentimiento,
                'estrellas': estrellas,
                'confianza': float(confianza),
                'error': None
            }
        except Exception as e:
            return {
                'error': str(e),
                'texto': texto
            }
    