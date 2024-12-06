### Implied Volatility Calculation Tool

This tool allows you to calculate the **implied volatility** (IV) of an 
option from a stock ticker and generate a volatility smile. Implied 
volatility is the market's estimate of the future volatility of the 
underlying asset, as inferred from option prices. It reflects the market’s 
expectations of future price fluctuations, influencing option pricing 
through the Black-Scholes model.

<br> </br>

#### How Implied Volatility is Calculated

Implied volatility is found by solving the Black-Scholes pricing model for 
volatility, given a market price for the option. The process involves a 
**binary search** algorithm, which iteratively adjusts the volatility 
input until the calculated option price matches the observed market price.

<br> </br>

1. **Black-Scholes Formula**:
   The price of a European call option \(C\) under the Black-Scholes model 
is given by: <br> </br>

   \[
   C(S, K, T, r, \sigma) = S \Phi(d_1) - K e^{-rT} \Phi(d_2)
   \]

   where:
   - \(S\) = Current stock price
   - \(K\) = Strike price
   - \(T\) = Time to expiration (in years)
   - \(r\) = Risk-free interest rate
   - \(\sigma\) = Volatility (the value we are solving for)
   - \(\Phi\) = Cumulative distribution function (CDF) of the standard 
normal distribution

   The terms \(d_1\) and \(d_2\) are defined as:

   \[
   d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r + 
\frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}
   \]
   \[
   d_2 = d_1 - \sigma \sqrt{T}
   \]

2. **Market Price of Option**:
   Let \(C_{market}\) be the observed market price of the option. The goal 
is to find the volatility \(\sigma_{implied}\) such that the Black-Scholes 
price \(C(S, K, T, r, \sigma_{implied})\) is equal to \(C_{market}\).

3. **Binary Search**:
   The binary search algorithm is applied to find \(\sigma_{implied}\) 
that satisfies the equation:

   \[
   C(S, K, T, r, \sigma_{implied}) = C_{market}
   \]


   

   The steps of the binary search are as follows:

   - **Step 1**: Initialize a range for \(\sigma\) (e.g., between 0 and 2, 
where 0 represents no volatility and 2 is very high volatility).
   - **Step 2**: Calculate the option price using the Black-Scholes 
formula at the midpoint of the current volatility range.
   - **Step 3**: If the calculated price is too high (i.e., \(C_{calc} > 
C_{market}\)), decrease the volatility range.
   - **Step 4**: If the calculated price is too low (i.e., \(C_{calc} < 
C_{market}\)), increase the volatility range.
   - **Step 5**: Repeat the process until the calculated option price 
\(C_{calc}\) is sufficiently close to the market price \(C_{market}\), 
within a small tolerance.

4. **Volatility Smile**:
   Once implied volatilities are calculated for various strike prices or 
expiration dates, the volatility smile can be plotted by graphing the 
implied volatility as a function of the strike price (or moneyness). This 
curve typically shows that implied volatility tends to increase for 
strikes far from the current stock price, particularly for deep in- or 
out-of-the-money options.

By performing this binary search, the tool determines the implied 
volatility that best matches the market's perception of future volatility, 
and the resulting volatility smile gives insights into the market’s view 
of volatility across different strikes.
