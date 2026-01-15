import yfinance as yf
import pandas as pd
from datetime import datetime

def AnalisisHibrido():
    # Tu lista combinada de KPIs.py y AnalisisAcciones.py
    portafolio = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO']
    
    informe = []
    
    print(f"🚀 INICIANDO SUPER SCANNER (Técnico + Fundamental)...\n")

    for ticker in portafolio:
        try:
            # 1. ANÁLISIS TÉCNICO (Rápido)
            # ----------------------------------------------------
            datos = yf.download(ticker, period='1y', interval='1d', progress=False)
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            
            if len(datos) < 15: continue

            # Cálculo de RSI (Tu lógica de AnalisisAcciones.py)
            delta = datos['Close'].diff()
            ganancia = (delta.where(delta > 0, 0))
            perdida = (-delta.where(delta < 0, 0))
            ewm_ganancia = ganancia.ewm(com=13, adjust=False).mean()
            ewm_perdida = perdida.ewm(com=13, adjust=False).mean()
            rs = ewm_ganancia / ewm_perdida
            rsi = 100 - (100 / (1 + rs))
            
            ultimo_rsi = rsi.iloc[-1]
            precio_actual = datos['Close'].iloc[-1]

            # Solo nos interesa si está en extremos (Filtrado)
            senal = "NEUTRAL"
            if ultimo_rsi < 35: senal = "OPORTUNIDAD DE COMPRA"
            elif ultimo_rsi > 70: senal = "ALERTA DE VENTA"
            
            # 2. ANÁLISIS FUNDAMENTAL (Solo si hay señal o es interesante)
            # ----------------------------------------------------
            # Hacemos esto para no perder tiempo descargando info de acciones que no vamos a operar
            pe_ratio = "N/A"
            profit_margin = "N/A"
            
            if senal != "NEUTRAL":
                print(f"🔎 Analizando a fondo {ticker} por señal {senal}...")
                empresa = yf.Ticker(ticker)
                info = empresa.info
                
                pe = info.get('trailingPE', 0)
                margen = info.get('profitMargins', 0)
                
                pe_ratio = round(pe, 2) if pe else "N/A"
                profit_margin = f"{round(margen * 100, 2)}%" if margen else "N/A"

            # 3. GUARDAR DATOS
            informe.append({
                'Ticker': ticker,
                'Precio': round(precio_actual, 2),
                'RSI': round(ultimo_rsi, 2),
                'Señal': senal,
                'P/E Ratio': pe_ratio,
                'Margen Ganancia': profit_margin
            })

        except Exception as e:
            print(f"Error en {ticker}: {e}")

    # 4. EXPORTAR RESULTADOS
    df = pd.DataFrame(informe)
    
    # Mostrar en pantalla las oportunidades primero
    print("\n--- RESUMEN DE OPORTUNIDADES ---")
    print(df[df['Señal'] != 'NEUTRAL'])
    
    # Guardar en Excel (requiere tener openpyxl instalado: pip install openpyxl)
"""    nombre_archivo = f"Reporte_Bursatil_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(nombre_archivo, index=False)
    print(f"\nReporte completo guardado en: {nombre_archivo}")"""

if __name__ == "__main__":
    AnalisisHibrido()