import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from concurrent.futures import ProcessPoolExecutor # Importado para paralelismo
import os
import warnings
import logging
import gc 

"""
# Lista de empresas expandida
PORTAFOLIO = [
    'MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO',
    'JPM', 'V', 'JNJ', 'UNH', 'PG', 'WMT', 'XOM', 'CAT', 'DIS', 'SPY'
]

NOMBRES = {
    'MSFT': 'Microsoft', 'AAPL': 'Apple', 'TSLA': 'Tesla', 'AMZN': 'Amazon',
    'GOOGL': 'Google', 'NVDA': 'NVIDIA', 'META': 'Meta', 'NFLX': 'Netflix',
    'INTC': 'Intel', 'AMD': 'AMD', 'KO': 'Coca-Cola',
    'JPM': 'JPMorgan', 'V': 'Visa', 'JNJ': 'Johnson & Johnson', 
    'UNH': 'UnitedHealth', 'PG': 'Procter & Gamble', 'WMT': 'Walmart', 
    'XOM': 'Exxon Mobil', 'CAT': 'Caterpillar', 'DIS': 'Disney', 'SPY': 'S&P 500 ETF (SPY)'
}
"""
# 1. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DIAS_MEMORIA_IA = 60 
PORTAFOLIO = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO']
NOMBRES = {
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

# 2. MOTOR IA Y ANÁLISIS (FUNCIONES ATÓMICAS)
# -----------------------------------------------------------------------------
def EntrenarPredecir(df_hist, ticker):
    """Entrenamiento Multivariante optimizado"""
    try:
        df = df_hist.copy()
        
        # Aseguramos que el RSI esté calculado para que la IA lo use como característica
        delta = df['Close'].diff()
        ganancia = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        perdida = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = ganancia / perdida
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)

        # Seleccionamos las 3 características
        features = ['Close', 'Volume', 'RSI']
        data = df[features].values
        
        if len(data) < DIAS_MEMORIA_IA + 50: return None, None

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        x_train, y_train = [], []
        for i in range(DIAS_MEMORIA_IA, len(scaled_data)):
            # Tomamos todas las columnas (:) para las X
            x_train.append(scaled_data[i-DIAS_MEMORIA_IA:i, :])
            # Predecimos solo el Close (columna 0)
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        # La forma ahora es (Muestras, 60, 3)
        
        optimizador_personalizado = Adam(learning_rate=0.0005)
        parada_temprana = EarlyStopping(
            monitor='loss', 
            patience=5, 
            restore_best_weights=True, 
            verbose=0
            )

        model = Sequential()
        # Ajustamos el Input Shape a 3 características
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2]))) 
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer=optimizador_personalizado, loss='mean_squared_error')
        model.fit(
            x_train, 
            y_train, 
            batch_size=64, 
            epochs=50, 
            callbacks=[parada_temprana], 
            verbose=0)

        # Predecir Mañana con el último bloque de 3 columnas
        ultimo_fragmento = scaled_data[-DIAS_MEMORIA_IA:].reshape(1, DIAS_MEMORIA_IA, 3)
        pred_scaled = model.predict(ultimo_fragmento, verbose=0)
        
        # Para el inverse_transform, necesitamos una matriz con la misma forma original (3 columnas)
        matriz_auxiliar = np.zeros((1, 3))
        matriz_auxiliar[0, 0] = pred_scaled # Ponemos la predicción en la columna 'Close'
        pred_final = scaler.inverse_transform(matriz_auxiliar)[0, 0]
        
        precio_hoy = df['Close'].iloc[-1]
        variacion_pct = ((pred_final - precio_hoy) / precio_hoy) * 100
        
        del model 
        tf.keras.backend.clear_session()

        return variacion_pct, precio_hoy

    except Exception as e:
        return None, None

def AnalisisTecnicoYFundamental_Optimizado(df_hist, ticker):
    """Cálculos técnicos estándar"""
    try:
        if len(df_hist) < 200: return None
        df = df_hist.copy()
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

        # Lógica de Diagnóstico Técnico
        rsi = hoy['RSI']
        tendencia_alcista = hoy['Close'] > hoy['SMA_200']
        
        # Validación de seguridad para evitar errores si RSI_Signal es NaN
        if pd.isna(ayer['RSI_Signal']) or pd.isna(hoy['RSI_Signal']):
            cruce_alcista = False
        else:
            cruce_alcista = (ayer['RSI'] < ayer['RSI_Signal']) and (hoy['RSI'] > hoy['RSI_Signal'])
        diagnostico_tec = "NEUTRAL"
        if rsi > 70: diagnostico_tec = "SOBRECOMPRA (Venta)"
        elif rsi < 35:
            if tendencia_alcista and cruce_alcista: diagnostico_tec = "COMPRA MAESTRA"
            elif cruce_alcista: diagnostico_tec = "COMPRA (Cruce)"
            elif tendencia_alcista: diagnostico_tec = "COMPRA (Tendencia)"
            else: diagnostico_tec = "REBOTE (Riesgo)"
        
        # Fundamentales (Esto sigue siendo lento por naturaleza de Yahoo, pero necesario)
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', 999) 
            if pe is None: pe = 999
        except:
            pe = 999
        return diagnostico_tec, rsi, pe
    except Exception as e:
        return None, None, None

def CalcularVeredicto(diag_tec, rsi, var_ia, pe):
    """Motor de decisión"""
    puntaje = 0
    if "COMPRA" in diag_tec: puntaje += 2
    elif "VENTA" in diag_tec: puntaje -= 2
    if var_ia > 1.5: puntaje += 2
    elif var_ia < -1.5: puntaje -= 2
    
    if puntaje >= 3: return "COMPRA FUERTE", puntaje
    elif puntaje >= 1: return "COMPRA", puntaje
    elif puntaje <= -2: return "VENTA", puntaje
    return "NEUTRAL", puntaje

# FUNCIÓN DE TAREA (Para el bucle o paralelismo)
def EjecutarAnalisisIndividual(ticker, df_ticker):
    """Procesa una sola empresa y devuelve el resultado"""
    var_ia, precio = EntrenarPredecir(df_ticker, ticker)
    if var_ia is None: return None
    diag_tec, rsi, pe = AnalisisTecnicoYFundamental_Optimizado(df_ticker, ticker)
    if diag_tec is None: return None
    decision, score = CalcularVeredicto(diag_tec, rsi, var_ia, pe)
    return {
        'Ticker': NOMBRES.get(ticker, ticker), 
        'Precio': round(precio, 2),
        'RSI': round(rsi, 1), 
        'IA Predicción': f"{var_ia:+.2f}%",
        'Técnico': diag_tec, 
        'RECOMENDACIÓN': decision, 
        'Score': score
    }

# 3. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def Prediccion():
    print("INICIANDO SCANNER")
    print("-" * 80)
    print("1. Descargando datos masivos...")
    try:
        datos = yf.download(PORTAFOLIO, period='4y', interval='1d', group_by='ticker', progress=False, auto_adjust=True)
    except: return

    informe = []
    
    # --- OPCIÓN A: EJECUCIÓN SECUENCIAL (Lenta pero segura) ---
    print("2. Procesando empresas secuencialmente...")
    for ticker in PORTAFOLIO:
        df_t = datos[ticker].dropna()
        if not df_t.empty:
            res = EjecutarAnalisisIndividual(ticker, df_t)
            if res:
                print(f"Finalizado: {res['Ticker']}")
                informe.append(res)

    # --- OPCIÓN B: EJECUCIÓN PARALELA (Rápida - DESCOMENTAR PARA USAR) ---
    # print("2. Procesando empresas en PARALELO...")
    # with ProcessPoolExecutor(max_workers=2) as executor: # max_workers=2 por tus 8GB de RAM
    #     tareas = {executor.submit(EjecutarAnalisisIndividual, t, datos[t].dropna()): t for t in PORTAFOLIO}
    #     for future in tareas:
    #         res = future.result()
    #         if res:
    #             print(f"Finalizado: {res['Ticker']}")
    #             informe.append(res)

    if informe:
        df_f = pd.DataFrame(informe).sort_values(by='Score', ascending=False)
        print("\n" + "="*120)
        print(" INFORME FINAL DE ESTRATEGIA ")
        print("="*130)
        print(df_f.drop(columns=['Score']).to_string(index=False))

if __name__ == "__main__":
    Prediccion()