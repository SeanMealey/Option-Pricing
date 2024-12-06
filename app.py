from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime, timedelta
import markdown
from mdx_math import MathExtension
import os


app = Flask(__name__)

@app.route('/strategies', methods=['GET', 'POST'])
def strategies():
    return render_template('strategies.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/submit-contact', methods=['GET', 'POST'])
def contact():
    return render_template('thanks.html')


@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    # Read markdown content from file
    print('MARKDOWN')
    with open('templates/iv-explanation.md', 'r') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML with LaTeX support
    html_content = markdown.markdown(
        markdown_content,
        extensions=['extra', 'codehilite', MathExtension(enable_dollar_delimiter=True)]
    )

    if request.method == 'POST':
        result = None
        volatility_smile_plot = None
        try:
            ticker = request.form['ticker']
            strike_price = float(request.form['strike_price'])
            dte = float(request.form['dte'])
            
            # Get available expiration dates
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            
            # Calculate target date
            target_date = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')
            
            # Find closest expiration
            closest_date = min(available_dates, key=lambda x: 
                abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target_date, '%Y-%m-%d')))
            
            # Get available strikes for this expiration
            chain = stock.option_chain(closest_date)
            available_strikes = sorted(chain.calls['strike'].unique())
            
            # Find closest strike
            closest_strike = min(available_strikes, key=lambda x: abs(x - strike_price))
            
            result = get_implied_volatility(ticker, closest_strike, closest_date)
            iv = result['implied_volatility']

            volatility_smile_plot = generate_volatility_smile(available_strikes, iv, ticker, closest_date)

            
            return render_template('stocks.html', result=result, iv=iv, volatility_smile_plot=volatility_smile_plot, html_content=html_content)
                                 
        except Exception as e:
            return render_template('stocks.html',
                                 result={'error': f'Error: {str(e)}'},
                                 form_data=request.form)

    try:
        with open('static/plots/volatility_smile_NVDA_2025-01-03.png', 'rb') as f:
            nvda_default_plot = base64.b64encode(f.read()).decode('utf-8')
        return render_template('stocks.html', 
                            html_content=html_content,
                            nvda_default_plot=nvda_default_plot)
    except FileNotFoundError:
        # If file doesn't exist, render template without plot
        return render_template('stocks.html', html_content=html_content)
    # Return template with both markdown content and any other necessary variables
    return render_template('stocks.html', html_content=html_content)

def generate_volatility_smile(available_strikes, iv, ticker, closest_date):
    iv_values = []
    valid_strikes = []
    
    # Get stock data
    stock = yf.Ticker(ticker)
    
    # Get available expiration dates
    available_dates = stock.options
    
    # Calculate target date (30 days from now)
    target_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Find closest expiration to 30 days
    thirty_day_expiry = min(available_dates, key=lambda x: 
        abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target_date, '%Y-%m-%d')))
    
    # Get chain for 30-day expiry
    chain = stock.option_chain(thirty_day_expiry)
    available_strikes = sorted(chain.calls['strike'].unique())
    
    for strike in available_strikes:
        result = get_implied_volatility(ticker, strike, thirty_day_expiry)
        # Only append values if IV is >= 1
        if result.get('implied_volatility', 0) >= 1:
            iv_values.append(result['implied_volatility'])
            valid_strikes.append(strike)

    # print(iv_values)
    # print(valid_strikes)
    # Only create plot if we have data
    if iv_values and valid_strikes:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_strikes, iv_values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility (%)')
        plt.title(f'30-Day Volatility Smile (Expiry: {thirty_day_expiry})')
        plt.grid(True)

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')

        # # Save to file system
        # os.makedirs('static/plots', exist_ok=True)
        # filename = f'volatility_smile_{ticker}_{thirty_day_expiry}.png'
        # plt.savefig(f'static/plots/{filename}', bbox_inches='tight')


        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_data
    
    return None


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period='1d')
    return {
        'underlying_price': stock_info['Close'].iloc[-1],
        'last': stock_info['Close'].iloc[-1],
        'bid': stock_info['Close'].iloc[-1],  # Substitute with real bid if available
        'ask': stock_info['Close'].iloc[-1],  # Substitute with real ask if available
    }

@app.route('/exotic', methods=['GET', 'POST'])
def exotic():
    result = None
    calendar_spread_data = None
    
    
    # Default values
    form_data = {
        'asset_price': 100,
        'strike_price': 100,
        'time_to_maturity': 1,
        'volatility': 0.2,
        'risk_free_rate': float(yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]/100),
        'option_type': 'call',
        'exotic_type': 'american',
        'average_price': None,
        'payout': None,
        'max_price': None,
        'min_price': None,
        'barrier_price': None,
        'barrier_type': None
    }

    if request.method == 'POST':
        try:
            # Update form_data with POST values
            form_data = {
                'asset_price': float(request.form['asset_price']),
                'strike_price': float(request.form['strike_price']),
                'time_to_maturity': float(request.form['time_to_maturity']),
                'volatility': float(request.form['volatility']),
                'risk_free_rate': float(request.form['risk_free_rate']),
                'option_type': request.form['option_type'],
                'exotic_type': request.form['exotic_type'],
                'average_price': float(request.form['average_price']) if 'average_price' in request.form else None,
                'payout': float(request.form['payout']) if 'payout' in request.form else None,
                'max_price': float(request.form['max_price']) if 'max_price' in request.form else None,
                'min_price': float(request.form['min_price']) if 'min_price' in request.form else None,
                'barrier_price': float(request.form['barrier_price']) if 'barrier_price' in request.form else None,
                'barrier_type': request.form['barrier_type'] if 'barrier_type' in request.form else None
            }
        except ValueError:
            return render_template('exotic.html', error="Error: Invalid input. Please enter valid numbers.", form_data=form_data)

    # Calculate result for both GET and POST requests
    try:
        if form_data['exotic_type'] == 'asian':
            result = asian_option_pricing(
                form_data['asset_price'],
                form_data['strike_price'],
                form_data['time_to_maturity'],
                form_data['volatility'],
                form_data['risk_free_rate'],
                form_data['option_type'],
                form_data['average_price']
            )
        elif form_data['exotic_type'] == 'digital':
            result = digital_option_pricing(
                form_data['asset_price'],
                form_data['strike_price'],
                form_data['time_to_maturity'],
                form_data['volatility'],
                form_data['risk_free_rate'],
                form_data['option_type'],
                form_data['payout']
            )
        elif form_data['exotic_type'] == 'lookback':
            result = lookback_option_pricing(
                form_data['asset_price'],
                form_data['strike_price'],
                form_data['time_to_maturity'],
                form_data['volatility'],
                form_data['risk_free_rate'],
                form_data['option_type'],
                form_data['max_price'],
                form_data['min_price']
            )
        elif form_data['exotic_type'] == 'barrier':
            result = barrier_option_pricing(
                form_data['asset_price'],
                form_data['strike_price'],
                form_data['time_to_maturity'],
                form_data['volatility'],
                form_data['risk_free_rate'],
                form_data['option_type'],
                form_data['barrier_price'],
                form_data['barrier_type']
            )
        elif form_data['exotic_type'] == 'american':
            result = binomial_tree_american(
                form_data['asset_price'],
                form_data['strike_price'],
                form_data['time_to_maturity'],
                form_data['volatility'],
                form_data['risk_free_rate'],
                form_data['option_type']
            )
    except Exception as e:
        return render_template('exotic.html', error=f"Error in calculation: {str(e)}", form_data=form_data)

    return render_template('exotic.html', result=result, calendar_spread_data=calendar_spread_data, form_data=form_data)


def load_ticker_data_logic(ticker):
    """Core logic separated from the route handler"""
    try:
        if not ticker:
            return {'error': 'No ticker provided'}

        # Get stock data
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Get options chain for the nearest expiration
        expirations = stock.options
        if not expirations:
            return {'error': 'No options data available for this ticker'}
        
        nearest_expiration = expirations[0]
        chain = stock.option_chain(nearest_expiration)
        
        # Find ATM strike price
        strikes = chain.calls['strike'].values
        strike_price = strikes[abs(strikes - current_price).argmin()]
        
        # Calculate DTE
        expiry_date = datetime.strptime(nearest_expiration, '%Y-%m-%d')
        dte = (expiry_date - datetime.now()).days
        
        # Get historical volatility
        hist_data = stock.history(period='1mo')
        hist_volatility = hist_data['Close'].pct_change().std() * (252 ** 0.5) * 100
        
        return {
            'asset_price': float(current_price),
            'strike_price': float(strike_price),
            'time_to_maturity': dte/365,  # Convert to years
            'risk_free_rate': yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]/100,
            'volatility': round(float(hist_volatility), 2)
        }

    except Exception as e:
        return {'error': f'Error loading data: {str(e)}'}


#option varieties
def barrier_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, barrier, barrier_type):
    """Price barrier options (Knock-In, Knock-Out) using analytical formulas."""
    # Calculate standard Black-Scholes parameters
    sigma_sqt = volatility * np.sqrt(time_to_maturity)
    d1 = (np.log(asset_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / sigma_sqt
    d2 = d1 - sigma_sqt
    
    # Calculate barrier-specific parameters
    mu = risk_free_rate - 0.5 * volatility**2
    lambda_param = (mu + 0.5 * volatility**2) / volatility**2
    y = np.log(barrier**2 / (asset_price * strike_price)) / (volatility * np.sqrt(time_to_maturity))
    
    # Standard Black-Scholes prices
    if option_type == 'call':
        vanilla_price = (asset_price * stats.norm.cdf(d1) - 
                        strike_price * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(d2))
    else:  # put
        vanilla_price = (strike_price * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(-d2) - 
                        asset_price * stats.norm.cdf(-d1))
    
    # Barrier adjustment
    barrier_factor = ((barrier / asset_price)**(2 * lambda_param) * 
                     stats.norm.cdf(y - lambda_param * sigma_sqt))
    
    if barrier_type == 'knock-out':
        return vanilla_price * (1 - barrier_factor)
    else:  # knock-in
        return vanilla_price * barrier_factor
    
def binomial_tree_american(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, steps=500):
    """Binomial tree model for pricing American options."""
    dt = time_to_maturity / int(steps)
    u = np.exp(volatility * np.sqrt(dt))  # Up factor
    d = 1 / u                            # Down factor
    p = (np.exp(risk_free_rate * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset price tree
    price_tree = np.zeros((int(steps) + 1, int(steps) + 1))
    for i in range(int(steps) + 1):
        for j in range(i + 1):
            price_tree[j, i] = asset_price * (u**j) * (d**(i - j))

    # Initialize option value tree
    option_tree = np.zeros_like(price_tree)
    for j in range(int(steps) + 1):
        if option_type == 'call':
            option_tree[j, int(steps)] = max(0, price_tree[j, int(steps)] - strike_price)
        elif option_type == 'put':
            option_tree[j, int(steps)] = max(0, strike_price - price_tree[j, int(steps)])

    # Backward induction
    for i in range(int(steps) - 1, -1, -1):
        for j in range(i + 1):
            continuation = (p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1]) * np.exp(-risk_free_rate * dt)
            if option_type == 'call':
                option_tree[j, i] = max(price_tree[j, i] - strike_price, continuation)
            elif option_type == 'put':
                option_tree[j, i] = max(strike_price - price_tree[j, i], continuation)

    return option_tree[0, 0]

def digital_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, payout):
    """Price digital options with a fixed payout."""
    d2 = (np.log(asset_price / strike_price) + (risk_free_rate - 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))

    if option_type == 'call':
        return payout * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(d2)
    elif option_type == 'put':
        return payout * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(-d2)

def lookback_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, max_price, min_price):
    """Price lookback options."""
    if option_type == 'call':
        intrinsic_value = max(max_price - strike_price, 0)
    elif option_type == 'put':
        intrinsic_value = max(strike_price - min_price, 0)
    
    return intrinsic_value * np.exp(-risk_free_rate * time_to_maturity)

def asian_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, average_price):
    """Price Asian options using arithmetic average."""
    adjusted_price = (asset_price + average_price) / 2
    d1 = (np.log(adjusted_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    if option_type == 'call':
        return adjusted_price * stats.norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(d2)
    elif option_type == 'put':
        return strike_price * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(-d2) - adjusted_price * stats.norm.cdf(-d1)


def black_scholes_pricing_and_greeks(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate):
    # Calculate d1 and d2
    d1 = (np.log(asset_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    # Common values
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    # Call and Put prices
    call_price = (asset_price * N_d1 
                  - strike_price * np.exp(-risk_free_rate * time_to_maturity) * N_d2)
    put_price = (strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) 
                 - asset_price * norm.cdf(-d1))

    # Greeks
    delta_call = N_d1
    delta_put = N_d1 - 1
    gamma = n_d1 / (asset_price * volatility * np.sqrt(time_to_maturity))
    theta_call = (- (asset_price * n_d1 * volatility) / (2 * np.sqrt(time_to_maturity))
                  - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * N_d2)
    theta_put = (- (asset_price * n_d1 * volatility) / (2 * np.sqrt(time_to_maturity))
                 + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2))
    vega = asset_price * n_d1 * np.sqrt(time_to_maturity)
    rho_call = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * N_d2
    rho_put = -strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)

    return {
        'Call Price': call_price,
        'Put Price': put_price,
        'Call Greeks': {
            'Delta': delta_call,
            'Gamma': gamma,
            'Theta': theta_call,
            'Vega': vega,
            'Rho': rho_call,
        },
        'Put Greeks': {
            'Delta': delta_put,
            'Gamma': gamma,
            'Theta': theta_put,
            'Vega': vega,
            'Rho': rho_put,
        }
    }

def calculate_option_data(form_data):
    """Calculate all option prices and chart data based on input parameters"""
    result = black_scholes_pricing_and_greeks(
        form_data['asset_price'], 
        form_data['strike_price'], 
        form_data['time_to_maturity'], 
        form_data['volatility'], 
        form_data['risk_free_rate']
    )

    # Generate stock price range for charts
    stock_prices = np.linspace(form_data['asset_price'] * 0.5, form_data['asset_price'] * 1.5, 100)
    
    # Calculate all Greeks data in a single loop
    call_deltas, put_deltas, vegas, gammas = [], [], [], []
    for price in stock_prices:
        greeks = black_scholes_pricing_and_greeks(
            price,
            form_data['strike_price'],
            form_data['time_to_maturity'],
            form_data['volatility'],
            form_data['risk_free_rate']
        )
        call_deltas.append(greeks['Call Greeks']['Delta'])
        put_deltas.append(greeks['Put Greeks']['Delta'])
        vegas.append(greeks['Call Greeks']['Vega'])
        gammas.append(greeks['Call Greeks']['Gamma'])  # Add gamma calculation

    # Calculate time-based Greeks
    times = np.linspace(0.01, form_data['time_to_maturity'] * 2, 100)
    call_thetas, put_thetas = [], []
    for t in times:
        greeks = black_scholes_pricing_and_greeks(
            form_data['asset_price'],
            form_data['strike_price'],
            t,
            form_data['volatility'],
            form_data['risk_free_rate']
        )
        call_thetas.append(greeks['Call Greeks']['Theta'])
        put_thetas.append(greeks['Put Greeks']['Theta'])

    # Calculate interest rate sensitivity
    interest_rates = np.linspace(0.0, 0.10, 100)
    call_rhos, put_rhos = [], []
    for rate in interest_rates:
        greeks = black_scholes_pricing_and_greeks(
            form_data['asset_price'],
            form_data['strike_price'],
            form_data['time_to_maturity'],
            form_data['volatility'],
            rate
        )
        call_rhos.append(greeks['Call Greeks']['Rho'])
        put_rhos.append(greeks['Put Greeks']['Rho'])

    return {
        'result': result,
        'delta_chart_data': {
            'stock_prices': stock_prices.tolist(),
            'call_deltas': call_deltas,
            'put_deltas': put_deltas
        },
        'gamma_chart_data': {  # Add gamma data to the return
            'stock_prices': stock_prices.tolist(),
            'gammas': gammas
        },
        'vega_chart_data': {
            'stock_prices': stock_prices.tolist(),
            'vegas': vegas
        },
        'theta_chart_data': {
            'times': times.tolist(),
            'call_thetas': call_thetas,
            'put_thetas': put_thetas
        },
        'rho_chart_data': {
            'interest_rates': interest_rates.tolist(),
            'call_rhos': call_rhos,
            'put_rhos': put_rhos
        }
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values
    form_data = {
        'asset_price': 100,
        'strike_price': 100,
        'time_to_maturity': 1,
        'volatility': 0.2,
        'risk_free_rate': float(yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]/100)  # Convert to float immediately
    }

    if request.method == 'POST':
        try:
            form_data = {
                'asset_price': float(request.form['asset_price']),
                'strike_price': float(request.form['strike_price']),
                'time_to_maturity': float(request.form['time_to_maturity']),
                'volatility': float(request.form['volatility']),
                'risk_free_rate': float(request.form['risk_free_rate'])  # Convert to float
            }
        except ValueError:
            return render_template('index.html', 
                                error="Error: Invalid input. Please enter valid numbers.",
                                form_data=form_data)

    # Calculate all data (for both GET and POST)
    option_data = calculate_option_data(form_data)
    
    return render_template('index.html',
                         result=option_data['result'],
                         delta_chart_data=option_data['delta_chart_data'],
                         gamma_chart_data=option_data['gamma_chart_data'],
                         theta_chart_data=option_data['theta_chart_data'],
                         rho_chart_data=option_data['rho_chart_data'],
                         vega_chart_data=option_data['vega_chart_data'],
                         form_data=form_data)

def get_option_market_price(ticker, strike_price, expiration_date, option_type='call'):
    """Get market price for an option using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiration_date)
        
        # Select calls or puts based on option_type
        chain = options.calls if option_type.lower() == 'call' else options.puts
        
        # Find options with the exact strike price
        matching_options = chain[chain['strike'] == strike_price]
        
        if matching_options.empty:
            # If no exact match, find the closest available strike
            available_strikes = chain['strike'].values
            closest_strike = min(available_strikes, key=lambda x: abs(x - strike_price))
            matching_options = chain[chain['strike'] == closest_strike]
            
        if matching_options.empty:
            return None
            
        option = matching_options.iloc[0]
        
        # Check if we have valid bid/ask prices
        if option['bid'] == 0 and option['ask'] == 0:
            return None
            
        return (option['bid'] + option['ask']) / 2
        
    except Exception as e:
        print(f"Error fetching option market price: {e}")
        return None

def calculate_implied_volatility(market_price, asset_price, strike_price, time_to_maturity, 
                               risk_free_rate, option_type='call', tolerance=0.1):
    """
    Calculate implied volatility using binary search method.
    Returns the volatility that produces an option price matching the market price.
    """
    max_iterations = 500
    vol_low = 0.001
    vol_high = 5.0  # 500% volatility as upper bound
    
    for i in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2
        
        # Calculate option price using Black-Scholes
        price_results = black_scholes_pricing_and_greeks(
            asset_price, strike_price, time_to_maturity, vol_mid, risk_free_rate
        )
        
        calculated_price = price_results['Call Price'] if option_type.lower() == 'call' else price_results['Put Price']
        
        price_diff = calculated_price - market_price
        
        # Check if we're within tolerance
        if abs(price_diff) < tolerance:
            return vol_mid
        
        # Adjust bounds based on price difference
        if price_diff > 0:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
            
        # # If we can't converge, return None
        # if vol_high - vol_low < tolerance:
        #     return None
    
    return (vol_low + vol_high) / 2

def get_implied_volatility(ticker, strike_price, expiration_date):
    try:
        # Get the stock data
        stock = yf.Ticker(ticker)

        # Get the option chain
        options = stock.option_chain(expiration_date)
        if options is None:
            return {'error': f'No option chain found for {ticker} on {expiration_date}'}
            
        # Get calls data
        calls = options.calls
        
        
        # Find the option with matching strike
        option = calls[calls['strike'] == strike_price]
        if option.empty:
            return {'error': f'No option found for strike price {strike_price}. Available strikes: {sorted(calls["strike"].unique())}'}
            
        # Get the market price and current stock price
        market_price = option['lastPrice'].iloc[0]
        if market_price <= 0:
            return {'error': f'Invalid market price {market_price}'}
        
        yf_implied_vol = float(option['impliedVolatility'].iloc[0])
        # print(f"\nYFinance Implied Volatility: {round(yf_implied_vol * 100, 2)}%")
            
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Calculate time to expiration in years
        expiry_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        time_to_maturity = days_to_expiry / 365.0
        
        # Use 5% as the risk-free rate (you might want to fetch this from a reliable source)
        risk_free_rate = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]/100
        
        # Calculate implied volatility
        try:
            implied_vol = calculate_implied_volatility(
                market_price=market_price,
                asset_price=current_price,
                strike_price=strike_price,
                time_to_maturity=time_to_maturity,
                risk_free_rate=risk_free_rate,
                option_type='call'
            )
            
            if implied_vol is None:
                return {'error': 'Could not converge on implied volatility'}
            
            # print(implied_vol)
                
            # Convert to percentage and round to 2 decimal places
            implied_vol_percentage = round(implied_vol * 100, 2)
            
            return {
                'implied_volatility': implied_vol_percentage,
                'market_price': market_price,
                'current_price': current_price,
                'days_to_expiry': days_to_expiry
            }
            
        except Exception as e:
            return {'error': f'Error in IV calculation: {str(e)}'}
            
    except Exception as e:
        return {'error': f'Error calculating IV: {str(e)}'}

@app.route('/load_ticker_data', methods=['POST'])
def load_ticker_data():
    try:
        ticker = request.json.get('ticker')
        if not ticker:
            return jsonify({'error': 'No ticker provided'})

        # Get stock data
        stock = yf.Ticker(ticker)
        
        # Get current price
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Get options chain for the nearest expiration
        expirations = stock.options
        if not expirations:
            return jsonify({'error': 'No options data available for this ticker'})
        
        nearest_expiration = expirations[0]
        chain = stock.option_chain(nearest_expiration)
        
        # Find ATM strike price (closest to current price)
        strikes = chain.calls['strike'].values
        strike_price = strikes[abs(strikes - current_price).argmin()]
        
        # Calculate DTE
        expiry_date = datetime.strptime(nearest_expiration, '%Y-%m-%d')
        dte = (expiry_date - datetime.now()).days
        
        # Get historical volatility (1 month)
        hist_data = stock.history(period='1mo')
        hist_volatility = hist_data['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized
        
        # Get current risk-free rate from ^TNX (10-year Treasury yield)
        risk_free_rate = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]

        return jsonify({
            'asset_price': float(current_price),
            'strike_price': int(strike_price),
            'dte': dte,
            'risk_free_rate': float(risk_free_rate),
            'volatility': round(float(hist_volatility), 2)
        })

    except Exception as e:
        return jsonify({'error': f'Error loading data: {str(e)}'})

@app.route('/calculate_exotic', methods=['POST'])
def calculate_exotic():
    try:
        data = request.get_json()

        # Extract form data
        exotic_type = data.get('exotic_type', 'american')
        option_type = data.get('option_type', 'call')

        asset_price = float(data.get('asset_price'))
        strike_price = float(data.get('strike_price'))
        time_to_maturity = float(data.get('time_to_maturity'))
        volatility = float(data.get('volatility'))
        risk_free_rate = float(data.get('risk_free_rate'))

        # Handle additional exotic parameters
        extra = {}
        if exotic_type == 'asian':
            average_price = float(data.get('average_price', asset_price))
        elif exotic_type == 'digital':
            payout = float(data.get('payout', 1))
        elif exotic_type == 'lookback':
            max_price = float(data.get('max_price', asset_price))
            min_price = float(data.get('min_price', asset_price))
        elif exotic_type == 'barrier':
            barrier_price = float(data.get('barrier_price', asset_price))
            barrier_type = data.get('barrier_type', 'knock-in')

        # Calculate option price based on exotic type
        if exotic_type == 'american':
            option_price = binomial_tree_american(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type)
        elif exotic_type == 'barrier':
            if barrier_type not in ['knock-in', 'knock-out']:
                return jsonify({'error': 'Invalid barrier type'}), 400
            option_price = barrier_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, barrier_price, barrier_type)
            extra['Barrier'] = f"${barrier_price} ({barrier_type.title()})"
        elif exotic_type == 'asian':
            option_price = asian_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, average_price)
            extra['Average Price'] = f"${average_price}"
        elif exotic_type == 'digital':
            option_price = digital_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, payout)
            extra['Payout'] = f"${payout}"
        elif exotic_type == 'lookback':
            option_price = lookback_option_pricing(asset_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type, max_price, min_price)
            extra['Max Price'] = f"${max_price}"
            extra['Min Price'] = f"${min_price}"
        else:
            return jsonify({'error': 'Invalid exotic type provided'}), 400

        return jsonify({
            'result': option_price,
            'exotic_type': exotic_type,
            'option_type': option_type,
            'asset_price': asset_price,
            'strike_price': strike_price,
            'time_to_maturity': time_to_maturity,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'extra': extra
        })

    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)