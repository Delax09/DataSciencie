import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import os

# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
# Desactivar advertencias molestas de TensorFlow para mantener la consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración visual para consola (Igual que AnalisisHibrido.py)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

DIAS_ATRAS = 60  # La "memoria" de la IA (días pasados que mira para predecir)

# 2. FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------
def PrepararDatos(ticker):
    """
    Descarga datos, normaliza y crea las estructuras X (pasado) e y (futuro)
    para el entrenamiento de la LSTM.
    """
    try:
        # Descarga de datos
        df = yf.download(ticker, period='4y', interval='1d', progress=False, auto_adjust=True)
        
        # Corrección Bug MultiIndex de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        data = df.filter(['Close']).values
        
        # Validación de datos suficientes
        if len(data) < DIAS_ATRAS + 50:
            return None, None, None, None

        # Escalar datos (0 a 1) para la Red Neuronal
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Crear estructuras de entrenamiento
        x_train, y_train = [], []
        
        # Usamos toda la historia disponible para entrenar
        for i in range(DIAS_ATRAS, len(scaled_data)):
            x_train.append(scaled_data[i-DIAS_ATRAS:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # Reshape para LSTM [Muestras, Pasos de tiempo, Características]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train, scaler, scaled_data
        
    except Exception as e:
        print(f"Error al preparar datos de {ticker}: {e}")
        return None, None, None, None

def ConstruirModelo(input_shape):
    """
    Define y compila la arquitectura de la Red Neuronal LSTM.
    """
    model = Sequential()
    
    # Capa de entrada + Capa 1: LSTM + Dropout
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Capa 2: LSTM + Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Capas Densas (Salida)
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Predicción final del precio
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3. PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
def ScannerNeuronal():
    portafolio = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO']
    
    nombres_empresas = {
        'MSFT': 'Microsoft',
        'AAPL': 'Apple',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'GOOGL': 'Google',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
        'NFLX': 'Netflix',
        'INTC': 'Intel',
        'AMD': 'AMD',
        'KO': 'Coca-Cola'
    }
    
    informe = []
    
    print(f"INICIANDO RED NEURONAL (LSTM) \n")
    print(f"Analizando {len(portafolio)} empresas. Esto tomará unos momentos por el entrenamiento...\n")

    for ticker in portafolio:
        try:
            nombre_completo = nombres_empresas.get(ticker, ticker)
            print(f"Entrenando IA para: {nombre_completo}...")
            
            # 1. Preparar Datos
            x_train, y_train, scaler, scaled_data = PrepararDatos(ticker)
            
            if x_train is None:
                print(f"Datos insuficientes para {ticker}")
                continue

            # 2. Construir y Entrenar Modelo
            # verbose=0 silencia la barra de progreso de TensorFlow
            model = ConstruirModelo((x_train.shape[1], 1))
            model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=0) 
            
            # 3. Predecir Precio de Mañana
            # Tomamos los últimos 60 días reales para predecir el siguiente paso
            ultimo_fragmento = scaled_data[-DIAS_ATRAS:]
            ultimo_fragmento = ultimo_fragmento.reshape(1, DIAS_ATRAS, 1)
            
            prediccion_scaled = model.predict(ultimo_fragmento, verbose=0)
            prediccion_final = scaler.inverse_transform(prediccion_scaled)[0][0]
            
            # 4. Obtener precio real de hoy (último dato del dataset)
            precio_hoy_scaled = np.array([[scaled_data[-1][0]]])
            precio_hoy = scaler.inverse_transform(precio_hoy_scaled)[0][0]

            # 5. Calcular Potencial (Diferencia %)
            variacion = ((prediccion_final - precio_hoy) / precio_hoy) * 100
            
            # Determinar dirección visual
            direccion = "ALCISTA 📈" if variacion > 0 else "BAJISTA 📉"

            # 6. Guardar resultados
            informe.append({
                'Ticker': nombre_completo,
                'Precio Hoy': round(precio_hoy, 2),
                'Predicción IA': round(prediccion_final, 2),
                'Variación %': f"{variacion:+.2f}%",
                'Dirección': direccion,
                'Raw_Var': variacion # Columna oculta para ordenar
            })

        except Exception as e:
            print(f"Error crítico en {ticker}: {e}")

    # --- RESULTADOS FINALES ---
    df = pd.DataFrame(informe)
    
    # Ordenar por el potencial absoluto (las predicciones más fuertes primero, sea subida o bajada)
    # Si prefieres solo las subidas arriba, quita el .abs()
    df = df.sort_values(by='Raw_Var', ascending=False, key=abs).drop(columns=['Raw_Var'])
    
    print("\n" + "="*80)
    print("PREDICCIONES DE INTELIGENCIA ARTIFICIAL PARA MAÑANA ")
    print("="*80)
    
    if not df.empty:
        print(df)
    else:
        print("No se pudieron generar predicciones.")

if __name__ == "__main__":
    ScannerNeuronal()