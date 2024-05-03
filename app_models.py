from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)



@app.route('/')
def formulario():
    return render_template('models_predic.html')

# Endpoint para el formulario
@app.route('/predict', methods=['POST'])
def predecir():
    # Obtener los datos del formulario
    person_age = float(request.form['person_age'])
    person_income = float(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = float(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])
    cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])

    # Leer valores de las listas desplegables
    p_h_o_trans = float(request.form['p_h_o_trans'])
    loan_intent_trans = float(request.form['loan_intent_trans'])
    loan_grade_trans = float(request.form['loan_grade_trans'])
    cb_pdf_trans = float(request.form['cb_pdf_trans'])

    # Crear DataFrame con los datos
    df_d_n = pd.DataFrame({
        'person_age' : [person_age],
        'person_income' : [person_income],
        'person_emp_length' : [person_emp_length],
        'loan_amnt' : [loan_amnt],
        'loan_int_rate' : [loan_int_rate],
        'loan_percent_income' : [loan_percent_income],
        'cb_person_cred_hist_length' : [cb_person_cred_hist_length]
    })

        # Cargamos el encoder desde el archivo
    encoder_load = joblib.load('encoder_num.pkl')

    # Normalizamos nuevos datos
    datos_norm = encoder_load.transform(df_d_n)

    person_age = datos_norm[0, 0]
    person_income = datos_norm[0, 1]
    person_emp_length = datos_norm[0, 2]
    loan_amnt = datos_norm[0, 3]
    loan_int_rate = datos_norm[0, 4]
    loan_percent_income = datos_norm[0, 5]
    cb_person_cred_hist_length = datos_norm[0, 6]

    # Convertir los datos en una matriz de características
    dato_a_predecir = np.array([
        [person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, 
         loan_percent_income, cb_person_cred_hist_length, p_h_o_trans, 
         loan_intent_trans, loan_grade_trans, cb_pdf_trans]
    ])

    # Redirigir al endpoint correspondiente según el botón presionado
    model_index = request.form['model_index']
    if model_index == '0':
        return predecir_modelo_1(dato_a_predecir)
    elif model_index == '1':
        return predecir_modelo_2(dato_a_predecir)
    elif model_index == '2':
        return predecir_modelo_3(dato_a_predecir)
    else:
        return "Modelo no válido"

# Funciones para predecir con cada modelo
def predecir_modelo_1(dato_a_predecir):
    model_load = joblib.load('model_credit_dt.plk')
    prediccion = model_load.predict(dato_a_predecir)

    
    value = prediccion[0]
    if value == 1:
            cat_pre = 'Aprobado'
    else:
            cat_pre = 'Denegado'
    return render_template('resultado_dt.html', resultado=cat_pre)

def predecir_modelo_2(dato_a_predecir):
    model_load = joblib.load('model_credit_svc.plk')
    prediccion = model_load.predict(dato_a_predecir)
    
    value = prediccion[0]
    if value == 1:
        cat_pre = 'Aprobado'
    else:
        cat_pre = 'Denegado'
    return render_template('resultado_svc.html', resultado=cat_pre)

def predecir_modelo_3(dato_a_predecir):
    model_load = joblib.load('model_credit_rfc.plk')
    prediccion = model_load.predict(dato_a_predecir)

    
    value = prediccion[0]
    if value == 1:
        cat_pre = 'Aprobado'
    else:
        cat_pre = 'Denegado'
    return render_template('resultado_rfc.html', resultado=cat_pre)

if __name__ == '__main__':
    app.run(debug=True)
