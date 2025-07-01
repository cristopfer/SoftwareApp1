import random
import os
from src.algoritmoGen import GeneticAlgorithm
from flask import Flask, render_template, request, jsonify, send_file, url_for
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import chi2_contingency
import numpy as np
from datetime import datetime
import tempfile
from werkzeug.utils import secure_filename
from src.teoremaNaBa import calcular_probabilidades_bayes
from src.redesNeuronales import entrenar_red_neuronal
from src.computerVision import save_sample_images, train_and_save_model, predict_flower, load_and_preprocess_data
from src.lenguajeNatural import AnalizadorSentimientos

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.static_folder = 'static'

file_path = 'dataframe/dataset_teorema_bayes_extendido.csv' 
_, _, CLASS_NAMES = load_and_preprocess_data()

@app.route('/')
def home():
   return render_template('AlgoritmoGenetico.html')

@app.route('/TeoremaBayes.html')
def home1():
    df = pd.read_csv(file_path)
    df = df.drop(columns=["PacienteID", "Edad"])
    return render_template('TeoremaBayes.html', table=df.head().to_dict(orient='records'))

@app.route('/RedesNeuronales.html')
def home2():
   return render_template('RedesNeuronales.html')

@app.route('/ComputerVision.html')
def home3():
   return render_template('ComputerVision.html')

@app.route('/LenguajeNatural.html')
def home4():
   return render_template('LenguajeNatural.html')

@app.route('/algoritmoGen', methods=['POST'])
def calcular():
    population_size = int(request.form['poblacion'])
    chromosome_length = int(request.form['cromosoma'])
    max_generations = int(request.form['generaciones'])
    mutation_probability = float(request.form['mutacion'])
    a3 = float(request.form['a3'])
    a2 = float(request.form['a2'])
    a1 = float(request.form['a1'])
    a0 = float(request.form['a0'])

    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        max_generations=max_generations,
        mutation_probability=mutation_probability,
        a3=a3,
        a2=a2,
        a1=a1,
        a0=a0
    )

    best_solution, iteracion = ga.run()

    return jsonify({'valor': best_solution, 'iteracion': iteracion})

@app.route('/teoremaBayes', methods=['POST'])
def probabilidad():
    genero = request.form['genero']
    fumador = request.form['fumador']
    actividad = request.form['actividadFisica']
    enfermedad = request.form['tieneEnfermedad']
    
    resultados = calcular_probabilidades_bayes(genero, fumador, actividad, enfermedad, file_path)
    return jsonify(resultados)

@app.route('/entrenar-and', methods=['POST'])
def entrenar_and():
    try:
        n_inputs = int(request.form.get('entrada'))
        hidden_layers = list(map(int, request.form.get('capaOculta').split(',')))
        learning_rate = float(request.form.get('tasa'))
        epochs = int(request.form.get('epoca'))

        def generate_and_data(num_inputs):
            inputs = []
            outputs = []
            for i in range(2**num_inputs):
                binary = [int(x) for x in format(i, f'0{num_inputs}b')]
                inputs.append(binary)
                outputs.append([random.randint(0, 1)])
            return np.array(inputs), np.array(outputs)

        X, y = generate_and_data(n_inputs)

        resultados = entrenar_red_neuronal(n_inputs,hidden_layers,1,learning_rate,X,y,epochs)

        return jsonify({"resultados": resultados })

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error inesperado: {str(e)}"}), 500

@app.route('/get_flowers', methods=['GET'])
def get_flowers():
    try:
        images = save_sample_images(8)  
        return jsonify({"success": True, "images": images})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/train', methods=['GET'])
def train_model():
    try:
        class_names = train_and_save_model()
        global CLASS_NAMES  # Actualizar los nombres globales
        CLASS_NAMES = class_names
        return jsonify({
            "success": True,
            "message": "Model trained successfully!",
            "classes": class_names
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        predicted_class = predict_flower(filepath)
        return jsonify({
            "success": True,
            "predicted_class": int(predicted_class),
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "filename": filename,
            "all_classes": CLASS_NAMES
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/analizar', methods=['POST'])
def analizar_texto():
    analizador = AnalizadorSentimientos()
    data = request.get_json()
    texto = data.get('texto', '')
    
    if not texto:
        return jsonify({'error': 'No se proporcionó texto'}), 400
    
    resultado = analizador.analizar(texto)
    return jsonify(resultado)

if __name__ == '__main__':
   port = int(os.environ.get('PORT', 5000))
   app.run(host='0.0.0.0', port=port)