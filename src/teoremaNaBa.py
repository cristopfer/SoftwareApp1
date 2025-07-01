import pandas as pd

def calcular_probabilidades_bayes(genero_input, fumador_input, actividad_input, enfermedad_input, file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["PacienteID", "Edad"])

    total_registros = len(df)
    count_positivo = (df["ResultadoTest"] == "Positivo").sum()
    count_negativo = total_registros - count_positivo
    
    p_priori_positivo = count_positivo / total_registros
    p_priori_negativo = count_negativo / total_registros

    def p_condicional(feature, value, target_class, alpha=1):
        subgrupo = df[df["ResultadoTest"] == target_class]
        total_clase = len(subgrupo)
        if total_clase == 0:
            return 0.0
        count = (subgrupo[feature] == value).sum()
        unique_values = df[feature].nunique()
        return (count + alpha) / (total_clase + alpha * unique_values)

    def get_value(feature, input_value):
        if input_value == "Otro":
            return df[feature].mode()[0]
        return input_value
    
    genero = get_value("Genero", genero_input)
    fumador = get_value("Fumador", fumador_input)
    actividad = get_value("ActividadFisica", actividad_input)
    enfermedad = get_value("TieneEnfermedad", enfermedad_input)

    p_genero_pos = p_condicional("Genero", genero, "Positivo")
    p_fumador_pos = p_condicional("Fumador", fumador, "Positivo")
    p_actividad_pos = p_condicional("ActividadFisica", actividad, "Positivo")
    p_enfermedad_pos = p_condicional("TieneEnfermedad", enfermedad, "Positivo")

    p_genero_neg = p_condicional("Genero", genero, "Negativo")
    p_fumador_neg = p_condicional("Fumador", fumador, "Negativo")
    p_actividad_neg = p_condicional("ActividadFisica", actividad, "Negativo")
    p_enfermedad_neg = p_condicional("TieneEnfermedad", enfermedad, "Negativo")

    posterior_pos = p_priori_positivo * p_genero_pos * p_fumador_pos * p_actividad_pos * p_enfermedad_pos
    posterior_neg = p_priori_negativo * p_genero_neg * p_fumador_neg * p_actividad_neg * p_enfermedad_neg

    total = posterior_pos + posterior_neg
    if total == 0:
        return {"error": "No se puede calcular con los datos disponibles"}
    
    probabilidad_pos = posterior_pos / total
    probabilidad_neg = posterior_neg / total
    return {'resultado1': round(probabilidad_pos,4), 'resultado2': round(probabilidad_neg,4)}
   