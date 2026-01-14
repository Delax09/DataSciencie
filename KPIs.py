import yfinance as yf
import pandas as pd

def AnalisisKPI():
    # Lista de empresas
    portafolio = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'INTC', 'AMD', 'KO']

    nombre_empresas = {
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
    
    print(f"ESCANEANDO DATOS FUNDAMENTALES (KPIs)...\n")
    
    # Encabezados de la tabla
    print(f"{'TICKER':<8} | {'P/E RATIO':<10} | {'DIVIDEND %':<10} | {'BETA':<6} | {'MARGEN DE GANANCIA'}")
    print("-" * 75)

    for simbolo in portafolio:
        try:
            # 1. Creamos el objeto Ticker (OJO: No usamos .download aquí)
            empresa = yf.Ticker(simbolo)
            
            # 2. Obtenemos el diccionario de información
            # Esto puede tardar un poco porque descarga muchos datos financieros
            info = empresa.info 
            
            # 3. Extraemos los KPIs usando .get() para evitar errores si falta el dato
            
            # P/E Ratio (Precio vs Beneficio): ¿La acción está cara o barata?
            pe_ratio = info.get('trailingPE', 0) 
            
            # Dividend Yield: ¿Cuánto paga de dividendo anual?
            dividendo = info.get('dividendRate', 0)
            precio_actual = info.get('currentPrice', 1)
            # A veces el yield viene calculado, a veces hay que calcularlo
            if precio_actual:
                div_yield = (dividendo / precio_actual) * 100
            else:
                div_yield = 0
            
            # Beta: ¿Qué tan volátil es respecto al mercado?
            beta = info.get('beta', 0)
            
            # Margen de Beneficio: De cada dólar que venden, ¿cuánto les queda limpio?
            margen = info.get('profitMargins', 0) * 100

            # 4. Formateo para que se vea limpio (Si es None o 0, mostramos N/A)
            txt_pe = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            txt_div = f"{div_yield:.2f}%" 
            txt_beta = f"{beta:.2f}" if beta else "N/A"
            txt_margen = f"{margen:.2f}%"

            print(f"{simbolo:<8} | {txt_pe:<10} | {txt_div:<10} | {txt_beta:<6} | {txt_margen}")

        except Exception as e:
            print(f"{simbolo:<8} | Error al obtener datos: {e}")

    print("-" * 75)
    print("\nAnálisis finalizado.")

if __name__ == "__main__":
    AnalisisKPI()