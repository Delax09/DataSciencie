import yfinance as yf
import pandas as pd
from datetime import datetime

# Configuración visual para consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

def calcular_tecnicos(datos):
    """
    Calcula RSI, Línea de Señal y Tendencia (SMA 200).
    Devuelve el DataFrame enriquecido.
    """
    # 1. RSI Estándar (14 periodos)
    delta = datos['Close'].diff()
    ganancia = (delta.where(delta > 0, 0))
    perdida = (-delta.where(delta < 0, 0))
    
    ewm_ganancia = ganancia.ewm(com=13, adjust=False).mean()
    ewm_perdida = perdida.ewm(com=13, adjust=False).mean()
    
    rs = ewm_ganancia / ewm_perdida
    datos['RSI'] = 100 - (100 / (1 + rs))

    # 2. NUEVO: Línea de Señal (Media del RSI de 14 dias)
    datos['RSI_Signal'] = datos['RSI'].rolling(window=14).mean()

    # 3. NUEVO: Tendencia de Largo Plazo (SMA 200)
    datos['SMA_200'] = datos['Close'].rolling(window=200).mean()
    
    return datos

def obtener_fundamentales(ticker):
    """
    Descarga P/E y Margen solo cuando es necesario.
    """
    try:
        empresa = yf.Ticker(ticker)
        info = empresa.info
        
        pe = info.get('trailingPE', 0)
        margen = info.get('profitMargins', 0)
        
        pe_fmt = round(pe, 2) if pe else "N/A"
        margen_fmt = f"{round(margen * 100, 2)}%" if margen else "N/A"
        
        return pe_fmt, margen_fmt
    except:
        return "Error", "Error"

def interpretar_senales(row, prev_row):
    """
    Aplica la lógica avanzada para determinar el diagnóstico.
    row: Datos de hoy
    prev_row: Datos de ayer (para ver cruces)
    """
    rsi = row['RSI']
    rsi_signal = row['RSI_Signal']
    precio = row['Close']
    sma_200 = row['SMA_200']
    
    # Detección de Cruce Alcista (RSI cruza hacia arriba su señal)
    cruce_alcista = (prev_row['RSI'] < prev_row['RSI_Signal']) and (rsi > rsi_signal)
    
    # Detección de Tendencia
    tendencia_alcista = precio > sma_200
    
    diagnostico = "NEUTRAL"
    prioridad = 0 # Para ordenar los resultados después

    # --- LÓGICA DE DECISIÓN PROFUNDA ---
    
    # 1. Señales de VENTA
    if rsi > 70:
        diagnostico = "VENDER (Sobrecompra)"
        prioridad = 1
        
    # 2. Señales de COMPRA
    elif rsi < 35:
        if tendencia_alcista and cruce_alcista:
            diagnostico = "COMPRA MAESTRA (Tendencia + Cruce)"
            prioridad = 5 # Máxima prioridad
        elif cruce_alcista:
            diagnostico = "COMPRA (Cruce confirmado)"
            prioridad = 4
        elif tendencia_alcista:
            diagnostico = "COMPRA (En Tendencia)"
            prioridad = 3
        else:
            diagnostico = "REBOTE (Riesgoso / Contra-tendencia)"
            prioridad = 2
    return diagnostico, prioridad

def scanner_profundo():
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
    
    print(f"INICIANDO ANÁLISIS PROFUNDO \n")

    for ticker in portafolio:
        try:
            # Descarga extendida a 2 años para poder calcular la SMA_200 correctamente
            #Aquí esta el periodo
            datos = yf.download(ticker, period='2y', interval='1d', progress=False, threads=True) 
            
            # Corrección Bug MultiIndex
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            
            if len(datos) < 200: 
                print(f"{ticker}: Datos insuficientes para SMA 200")
                continue

            # 1. Calcular Indicadores
            datos = calcular_tecnicos(datos)
            
            # 2. Obtener últimas 2 filas (Hoy y Ayer) para comparar
            hoy = datos.iloc[-1]
            ayer = datos.iloc[-2]
            
            # 3. Interpretar Señal
            diagnostico, prioridad = interpretar_senales(hoy, ayer)
            
            # 4. Si hay señal interesante, buscar fundamentales
            pe_ratio, profit_margin = "---", "---"
            
            if diagnostico != "NEUTRAL":
                print(f"Profundizando en {nombres_empresas[ticker]} ({diagnostico})...")
                pe_ratio, profit_margin = obtener_fundamentales(ticker)

            # 5. Guardar
            informe.append({
                'Ticker': nombres_empresas[ticker],
                'Precio': round(hoy['Close'], 2),
                'RSI': round(hoy['RSI'], 2),
                'Tendencia': 'Alcista 📈' if hoy['Close'] > hoy['SMA_200'] else 'Bajista 📉',
                'Diagnóstico': diagnostico,
                'P/E': pe_ratio,
                'Margen': profit_margin,
                'Prioridad': prioridad # Columna oculta para ordenar
            })

        except Exception as e:
            print(f"Error en {ticker}: {e}")

    # --- RESULTADOS FINALES ---
    df = pd.DataFrame(informe)
    
    # Ordenar por prioridad (Las mejores oportunidades arriba)
    df = df.sort_values(by='Prioridad', ascending=False).drop(columns=['Prioridad'])
    
    # Filtrar solo lo que no sea Neutral para mostrar
    df_mostrar = df[df['Diagnóstico'] != 'NEUTRAL']

    print("\n" + "="*80)
    print(" RESUMEN ESTRATÉGICO DE OPORTUNIDADES ")
    print("="*80)
    
    if not df_mostrar.empty:
        print(df_mostrar)
    else:
        print("El mercado está indeciso. No hay señales claras de alta probabilidad hoy.")
        
    # Exportar (Opcional, descomentar si deseas guardar)
    # df.to_csv(f"Reporte_Profundo_{datetime.now().strftime('%Y-%m-%d')}.csv", index=False)

if __name__ == "__main__":
    scanner_profundo()