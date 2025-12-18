# Reactive Power Buy Decision Tool

A Streamlit app that computes hourly BUY/NO BUY decisions for reactive power based on network demand/generation forecasts and day-ahead market prices.

## Setup

1. Install Python 3.10 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload two files:
   - **Norwegian_case.xlsx**: Contains demand and generation forecast data (sheets: "demand", "gen_pforecast")
   - **Piyasa_Takas_Fiyati-YYYYMMDD-YYYYMMDD.csv**: Contains EPİAŞ day-ahead market price data (semicolon-separated)

3. Adjust algorithm parameters in the sidebar:
   - `Q_target`: Target reactive power (default: 0.0 MVar)
   - `PF_min`: Minimum power factor (default: 0.90)
   - `Pmin_frac`: Minimum PV power fraction (default: 0.10)
   - `S_rated multiplier`: Inverter rating multiplier (default: 1.1)
   - `Coverage ratio`: Fraction of Q gap to cover (default: 0.8)

4. Click "Run Decision Algorithm" to compute decisions

5. View results:
   - Interactive plots showing power profiles, reactive demand, prices, and decisions
   - Decision table with all computed values
   - Download results as CSV

## Algorithm

The app implements a heuristic that:
1. Computes hourly Q need (target - demand)
2. Determines inverter Q capacity based on PV output and constraints (S-limit and PF-limit)
3. Calculates residual Q gap that needs to be purchased
4. Selects hours to BUY reactive power based on price (cheapest hours first) up to the coverage ratio

## Output

For each hour (0-23), the app outputs:
- `P_total`: Total active demand (MW)
- `Q_demand`: Reactive demand (MVar)
- `P_PV_max`: Maximum PV generation (MW)
- `Qmax`: Maximum Q capacity from inverter (MVar)
- `Q_from_inv`: Q supplied by inverter (MVar)
- `Q_gap`: Residual Q gap requiring purchase (MVar)
- `price_TL_per_kWh`: Market price (TL/kWh)
- `decision`: BUY or NO BUY



# Reactive-Power-Purchase-Decision-Tool
