import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import os

# 1. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Silenciar TensorFlow
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

DIAS_MEMORIA_IA = 60  # Días que la IA mira hacia atrás

# Lista de empresas
PORTAFOLIO = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO']
NOMBRES = {
    'MSFT': 'Microsoft', 'AAPL': 'Apple', 'TSLA': 'Tesla', 'AMZN': 'Amazon',
    'GOOGL': 'Google', 'NVDA': 'NVIDIA', 'META': 'Meta', 'NFLX': 'Netflix',
    'INTC': 'Intel', 'AMD': 'AMD', 'KO': 'Coca-Cola'
}

# 2. MÓDULO DE INTELIGENCIA ARTIFICIAL (LSTM)
# -----------------------------------------------------------------------------
def EntrenarYPredecir(ticker):
    """
    Entrena una Red Neuronal LSTM específica para el ticker y predice el % de cambio.
    Basado en tu archivo AnalisisRedNeuronal.py
    """
    try:
        # Descargamos 4 años para tener buen historial de entrenamiento
        df = yf.download(ticker, period='4y', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        data = df.filter(['Close']).values
        if len(data) < DIAS_MEMORIA_IA + 50: return None, None # Datos insuficientes

        # Escalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Preparar X e y
        x_train, y_train = [], []
        for i in range(DIAS_MEMORIA_IA, len(scaled_data)):
            x_train.append(scaled_data[i-DIAS_MEMORIA_IA:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Arquitectura LSTM (La misma de tu archivo)
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compilar y Entrenar (Rápido: 10 epochs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

        # Predecir Mañana
        ultimo_fragmento = scaled_data[-DIAS_MEMORIA_IA:].reshape(1, DIAS_MEMORIA_IA, 1)
        pred_scaled = model.predict(ultimo_fragmento, verbose=0)
        pred_final = scaler.inverse_transform(pred_scaled)[0][0]
        
        precio_hoy = df['Close'].iloc[-1]
        variacion_pct = ((pred_final - precio_hoy) / precio_hoy) * 100
        
        return variacion_pct, precio_hoy

    except Exception as e:
        return None, None

# 3. MÓDULO TÉCNICO Y FUNDAMENTAL
# -----------------------------------------------------------------------------
def AnalisisTecnicoYFundamental(ticker):
    """
    Calcula RSI, SMA 200, Cruces y descarga P/E.
    Basado en tu archivo AnalisisHibrido.py
    """
    try:
        # Descarga de datos técnicos
        df = yf.download(ticker, period='2y', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        if len(df) < 200: return None # Necesitamos 200 días para la media móvil

        # Indicadores
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        delta = df['Close'].diff()
        ganancia = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        perdida = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = ganancia / perdida
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Signal'] = df['RSI'].rolling(window=14).mean()

        # Datos de Hoy y Ayer
        hoy = df.iloc[-1]
        ayer = df.iloc[-2]

        # Lógica de Diagnóstico Técnico (Tu código Híbrido)
        rsi = hoy['RSI']
        tendencia_alcista = hoy['Close'] > hoy['SMA_200']
        cruce_alcista = (ayer['RSI'] < ayer['RSI_Signal']) and (hoy['RSI'] > hoy['RSI_Signal'])
        
        diagnostico_tec = "NEUTRAL"
        
        if rsi > 70:
            diagnostico_tec = "SOBRECOMPRA (Venta)"
        elif rsi < 35:
            if tendencia_alcista and cruce_alcista: diagnostico_tec = "COMPRA MAESTRA"
            elif cruce_alcista: diagnostico_tec = "COMPRA (Cruce)"
            elif tendencia_alcista: diagnostico_tec = "COMPRA (Tendencia)"
            else: diagnostico_tec = "REBOTE (Riesgo)"
        
        # Fundamentales
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', 999) # 999 si no hay dato
        except:
            pe = 999
            
        return diagnostico_tec, rsi, pe, hoy['SMA_200']

    except Exception:
        return None, None, None, None

# 4. MOTOR DE DECISIÓN (EL JUEZ)
# -----------------------------------------------------------------------------
def CalcularVeredicto(diag_tec, rsi, var_ia, pe):
    puntaje = 0
    razones = []

    # A. Evaluación Técnica
    if "COMPRA MAESTRA" in diag_tec:
        puntaje += 3
        razones.append("Técnico Perfecto")
    elif "COMPRA" in diag_tec:
        puntaje += 2
        razones.append("Señal Técnica Alcista")
    elif "REBOTE" in diag_tec:
        puntaje += 1
        razones.append("Posible Rebote")
    elif "SOBRECOMPRA" in diag_tec:
        puntaje -= 3
        razones.append("Técnicamente Cara")

    # B. Evaluación IA
    if var_ia > 1.5:
        puntaje += 2
        razones.append(f"IA muy alcista (+{var_ia:.1f}%)")
    elif var_ia > 0.3:
        puntaje += 1
    elif var_ia < -1.5:
        puntaje -= 2
        razones.append(f"IA muy bajista ({var_ia:.1f}%)")

    # C. Evaluación Fundamental (Seguridad)
    if pe > 100 and pe != 999:
        puntaje -= 2
        razones.append(f"P/E Peligroso ({pe:.0f})")
    
    # Resultado Final
    if puntaje >= 4: return "COMPRA FUERTE", puntaje, razones
    elif puntaje >= 2: return "COMPRA", puntaje, razones
    elif puntaje <= -2: return "VENTA", puntaje, razones
    else: return "NEUTRAL/ESPERAR", puntaje, razones

# 5. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def ScannerSupremo():
    print(f"INICIANDO PREDICCION  (Híbrido + Redes Neuronales LSTM)")
    print(f"Analizando {len(PORTAFOLIO)} empresas. Entrenando modelos de IA en tiempo real...\n")
    
    informe = []

    for ticker in PORTAFOLIO:
        nombre = NOMBRES.get(ticker, ticker)
        print(f"Procesando {nombre}...", end="\r")
        
        # 1. Ejecutar IA
        var_ia, precio = EntrenarYPredecir(ticker)
        if var_ia is None: continue

        # 2. Ejecutar Técnicos
        diag_tec, rsi, pe, sma200 = AnalisisTecnicoYFundamental(ticker)
        if diag_tec is None: continue

        # 3. Calcular Veredicto
        decision, score, razones = CalcularVeredicto(diag_tec, rsi, var_ia, pe)

        # Formato visual IA
        flecha_ia = "📈" if var_ia > 0 else "📉"
        
        informe.append({
            'Ticker': nombre,
            'Precio': precio,
            'RSI': round(rsi, 1),
            'IA Predicción': f"{var_ia:+.2f}% {flecha_ia}",
            'Diagnóstico Téc.': diag_tec,
            'P/E': round(pe, 1) if pe != 999 else "N/A",
            'RECOMENDACIÓN': decision,
            'Score': score # Oculto, solo para ordenar
        })

    # Mostrar Tabla Final
    df = pd.DataFrame(informe)
    
    # Ordenar: Primero las mejores compras (Score más alto), luego las ventas fuertes (Score más bajo)
    df = df.sort_values(by='Score', ascending=False).drop(columns=['Score'])

    print("\n" + "="*130)
    print(f"INFORME FINAL")
    print("="*130)
    print(df.to_string(index=False))

if __name__ == "__main__":
    ScannerSupremo()