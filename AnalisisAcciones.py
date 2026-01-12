import yfinance as yf
import pandas as pd

# Configuración visual para que Pandas muestre todas las columnas en la consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

def scanner_detallado():
    portafolio = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META']
    
    print(f"INICIANDO ANÁLISIS DETALLADO (Lógica RSI paso a paso)\n")

    for ticker in portafolio:
        print("=" * 100)
        print(f" ANALIZANDO: {ticker} ")
        print("=" * 100)
        
        try:
            # 1. Descargar datos
            datos = yf.download(ticker, period='1y', interval='1d', progress=False)
            
            # Corrección de formato (Bug MultiIndex de yfinance)
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)

            if len(datos) < 15:
                continue

            # ---------------------------------------------------------
            # PASO A PASO: Agregamos columnas con los cálculos intermedios
            # ---------------------------------------------------------
            
            # Paso 1: Diferencia (Precio hoy - Precio ayer)
            datos['Diff'] = datos['Close'].diff()

            # Paso 2: Separar Ganancias y Pérdidas
            # Si Diff > 0 es ganancia, si no, 0
            datos['Ganancia'] = datos['Diff'].where(datos['Diff'] > 0, 0)
            # Si Diff < 0 es pérdida (le quitamos el signo negativo con el -), si no, 0
            datos['Perdida'] = -datos['Diff'].where(datos['Diff'] < 0, 0)

            # Paso 3: Calcular Promedios Exponenciales (Suavizado de 14 días)
            # Esto es lo que usa el RSI internamente para no saltar bruscamente
            periodos = 14
            datos['Avg_Ganancia'] = datos['Ganancia'].ewm(com=periodos-1, adjust=False).mean()
            datos['Avg_Perdida'] = datos['Perdida'].ewm(com=periodos-1, adjust=False).mean()

            # Paso 4: Calcular RS (Relative Strength)
            # Evitamos división por cero
            datos['RS'] = datos['Avg_Ganancia'] / datos['Avg_Perdida']

            # Paso 5: Calcular RSI Final (0 - 100)
            datos['RSI'] = 100 - (100 / (1 + datos['RS']))

            # ---------------------------------------------------------
            # VISUALIZACIÓN
            # ---------------------------------------------------------
            
            # Seleccionamos solo las columnas de cálculo para mostrar
            columnas_a_mostrar = [
                'Close', 'Diff', 'Ganancia', 'Perdida', 
                'Avg_Ganancia', 'Avg_Perdida', 'RSI'
            ]
            
            # Mostramos los últimos 10 días formateados
            # apply(lambda x...) se usa aquí solo para formatear los decimales bonitos
            tabla_final = datos[columnas_a_mostrar].tail(10)
            
            print(tabla_final.applymap(lambda x: f"{x:.2f}"))
            
            # Diagnóstico final
            ultimo_rsi = datos['RSI'].iloc[-1]
            estado = "NEUTRAL"
            if ultimo_rsi < 30: estado = "COMPRA (Sobrevendido)"
            elif ultimo_rsi > 70: estado = "VENTA (Sobrecomprado)"
            
            print(f"\n---> CONCLUSIÓN PARA {ticker}: RSI actual es {ultimo_rsi:.2f} ({estado})\n\n")

        except Exception as e:
            print(f"Error en {ticker}: {e}")

if __name__ == "__main__":
    scanner_detallado()