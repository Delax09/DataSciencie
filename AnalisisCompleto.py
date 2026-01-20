import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings

# 1. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Silenciar TensorFlow
warnings.filterwarnings("ignore") # Silenciar advertencias de pandas/yfinance
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

DIAS_MEMORIA_IA = 60  # Días que la IA mira hacia atrás

# Lista de empresas
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

# 2. MÓDULO DE INTELIGENCIA ARTIFICIAL (LSTM)
# -----------------------------------------------------------------------------
def EntrenarPredecir(df_hist, ticker):
    """
    Recibe el DataFrame ya descargado. Entrena la LSTM con Learning Rate 
    optimizado y Early Stopping, luego predice.
    """
    try:
        # Preparamos los datos (Solo Close)
        data = df_hist[['Close']].values
        
        if len(data) < DIAS_MEMORIA_IA + 50: return None, None

        # Escalar datos entre 0 y 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Preparar X e y basándose en los días de memoria configurados
        x_train, y_train = [], []
        for i in range(DIAS_MEMORIA_IA, len(scaled_data)):
            x_train.append(scaled_data[i-DIAS_MEMORIA_IA:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # --- OPTIMIZACIÓN: CONFIGURACIÓN DE APRENDIZAJE ---
        # Definimos una tasa de aprendizaje más conservadora para mejorar la convergencia
        optimizador_personalizado = Adam(learning_rate=0.0005)

        # Configuramos la parada temprana para detener el entrenamiento si el error no mejora
        parada_temprana = EarlyStopping(
            monitor='loss', 
            patience=5, #cantidad de eopocas sin mejora antes de parar
            restore_best_weights=True,
            verbose=0
        )

        # Arquitectura LSTM
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        #Compilar con el optimizador personalizado
        model.compile(optimizer=optimizador_personalizado, loss='mean_squared_error')
        
        #Entrenar con Early Stopping.
        #A más epocas mayor precisión, pero más tiempo en ejecución
        model.fit(
            x_train, 
            y_train, 
            batch_size=64, 
            epochs=50, 
            callbacks=[parada_temprana],
            verbose=0
        )

        # Predecir Mañana
        ultimo_fragmento = scaled_data[-DIAS_MEMORIA_IA:].reshape(1, DIAS_MEMORIA_IA, 1)
        pred_scaled = model.predict(ultimo_fragmento, verbose=0)
        pred_final = scaler.inverse_transform(pred_scaled)[0][0]
        
        precio_hoy = df_hist['Close'].iloc[-1]
        variacion_pct = ((pred_final - precio_hoy) / precio_hoy) * 100
        
        # Eliminar el modelo y limpiar sesión para liberar memoria
        del model 
        tf.keras.backend.clear_session()

        return variacion_pct, precio_hoy

    except Exception as e:
        # print(f"Error IA en {ticker}: {e}") 
        return None, None

# 3. MÓDULO TÉCNICO Y FUNDAMENTAL
# -----------------------------------------------------------------------------
def AnalisisTecnicoYFundamental_Optimizado(df_hist, ticker):
    """
    Usa el DataFrame ya descargado para cálculos técnicos.
    Descarga P/E ratio aparte (esto no se puede hacer en bulk fácilmente).
    """
    try:
        if len(df_hist) < 200: return None

        #Copiamos para no afectar el dataframe original
        df = df_hist.copy()

        # Indicadores Técnicos
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
        
        if rsi > 70:
            diagnostico_tec = "SOBRECOMPRA (Venta)"
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
        # print(f"Error Técnico en {ticker}: {e}")
        return None, None, None

# 4. MOTOR DE DECISIÓN
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

# 5. EJECUCIÓN PRINCIPAL OPTIMIZADA

def Prediccion():
    print(f"INICIANDO SCANNER OPTIMIZADO")
    print("-" * 80)
    
    # PASO 1: DESCARGA MASIVA (La optimización clave)
    print(f"1. Descargando historial de 4 años para {len(PORTAFOLIO)} empresas...")
    try:
        datos_globales = yf.download(PORTAFOLIO, period='4y', interval='1d', group_by='ticker', progress=True, auto_adjust=True)
    except Exception as e:
        print(f"Error crítico en descarga: {e}")
        return

    informe = []
    print(f"\n2. Entrenando Modelos de IA y Analizando Datos...\n")

    for ticker in PORTAFOLIO:
        nombre = NOMBRES.get(ticker, ticker)
        print(f" Procesando {nombre}...", end="\r")
        
        # Extraer el DataFrame específico de la empresa
        try:
            df_ticker = datos_globales[ticker].copy()
            # Limpieza básica
            df_ticker.dropna(inplace=True)
            
            if df_ticker.empty:
                print(f"Sin datos para {ticker}")
                continue
        except KeyError:
            print(f"Error accediendo a datos de {ticker}")
            continue
        
        # 1. Ejecutar IA (Usando el df ya descargado)
        var_ia, precio = EntrenarPredecir(df_ticker, ticker)
        if var_ia is None: continue

        # 2. Ejecutar Técnicos (Usando el MISMO df)
        diag_tec, rsi, pe = AnalisisTecnicoYFundamental_Optimizado(df_ticker, ticker)
        if diag_tec is None: continue

        # 3. Calcular Veredicto
        decision, score, razones = CalcularVeredicto(diag_tec, rsi, var_ia, pe)

        # Formato visual
        flecha_ia = "📈" if var_ia > 0 else "📉"
        
        informe.append({
            'Ticker': nombre,
            'Precio': precio,
            'RSI': round(rsi, 1),
            'IA Predicción': f"{var_ia:+.2f}% {flecha_ia}",
            'Diagnóstico Téc.': diag_tec,
            'P/E': round(pe, 1) if pe != 999 else "N/A",
            'RECOMENDACIÓN': decision,
            'Score': score # Oculto
        })

    # Mostrar Tabla Final
    if len(informe) > 0:
        df = pd.DataFrame(informe)
        df = df.sort_values(by='Score', ascending=False).drop(columns=['Score'])

        print("\n" + "="*130)
        print(f"INFORME FINAL DE ESTRATEGIA")
        print("="*130)
        print(df.to_string(index=False))
    else:
        print("\nNo se generaron resultados. Revisa tu conexión a internet.")

if __name__ == "__main__":
    Prediccion()