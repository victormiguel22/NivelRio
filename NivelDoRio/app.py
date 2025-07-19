from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo e o scaler
try:
    model = joblib.load('modelo_previsao_rio.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Modelo e scaler carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo ou scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever', methods=['POST'])
def prever():
    try:
        chuva_itup = float(request.form['chuva_itup'])
        nivel_itup = float(request.form['nivel_itup'])
        chuva_taio = float(request.form['chuva_taio'])
        nivel_taio = float(request.form['nivel_taio'])
        # Reorder inputs to match training: NivelItuporanga, ChuvaItuporanga, NivelTaio, ChuvaTaio
        entrada = np.array([[nivel_itup, chuva_itup, nivel_taio, chuva_taio]])
        entrada_scaled = scaler.transform(entrada)
        previsao = model.predict(entrada_scaled)[0]
        return render_template('resultado.html', previsao=f'{previsao:.2f}')
    except Exception as e:
        return f"Erro na previsão: {e}"

if __name__ == '__main__':
    print("Iniciando o servidor Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)