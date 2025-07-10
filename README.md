# Mathematical Finance

Comprehensive course materials and implementations from MATH 176: Mathematical Finance at UC San Diego.

## Repository Structure

```
Math_of_Finance/
├── textbook_notes/      # Chapter-by-chapter course notes
├── discussion_sections/ # Weekly discussion materials
├── exams/              # Practice tests and review materials
├── implementations/     # Python implementations
└── projects/           # Advanced projects
```

## Course Content

### Textbook Notes

Complete chapter coverage with theory and examples:

1. **Chapter 1**: Interest rates and time value of money
2. **Chapter 2**: Fixed income securities and bond pricing
3. **Chapter 3**: Term structure of interest rates
4. **Chapter 4**: Portfolio theory and mean-variance optimization
5. **Chapters 5-6**: Options fundamentals and arbitrage pricing
6. **Chapter 7**: Black-Scholes model and Greeks
7. **Chapter 8**: American options and early exercise
8. **Chapter 9**: Exotic options and structured products

### Discussion Sections

Weekly discussion materials with problem sets and solutions:
- Week 5: Introduction to options pricing
- Week 6: Black-Scholes applications
- Week 9: Advanced portfolio optimization

### Exam Materials

**Practice Tests:**
- Midterm 1 with solutions
- Practice problems for all major topics

**Review Sessions:**
- Midterm 1: Bond pricing and portfolio theory
- Midterm 2: Stochastic calculus and Ito's lemma
- Final Review: Options pricing models
- Final Review: Betting strategies and probability
- Final Review: Comprehensive options theory

## Python Implementations

The `implementations/` directory contains modular Python code:

### Core Modules

- **finance.py**: Financial calculations
  - Present/future value computations
  - Bond pricing and yield calculations
  - Duration and convexity measures
  
- **option_hedging.py**: Options analytics
  - Black-Scholes pricing formulas
  - Greeks calculations (Delta, Gamma, Vega, Theta, Rho)
  - Hedging strategies implementation

- **Sports_Betting.py**: Optimal betting
  - Kelly criterion implementation
  - Risk management for betting strategies
  - Expected value calculations

- **Math_Functions.py**: Mathematical utilities
  - Probability distributions
  - Numerical integration methods
  - Statistical functions

- **plot.py**: Visualization tools
  - Price path simulations
  - Greeks visualization
  - Portfolio efficient frontier plots

## Projects

### Deep Hedging

Neural network approach to option hedging that learns optimal hedging strategies directly from market data, going beyond traditional Black-Scholes delta hedging.

## Key Topics Covered

1. **Fixed Income Mathematics**
   - Bond pricing and yield curves
   - Duration and convexity
   - Term structure modeling

2. **Portfolio Theory**
   - Markowitz mean-variance optimization
   - Capital Asset Pricing Model (CAPM)
   - Risk measures and performance metrics

3. **Stochastic Calculus**
   - Brownian motion and Ito processes
   - Stochastic differential equations
   - Ito's lemma applications

4. **Options Pricing**
   - Binomial tree models
   - Black-Scholes-Merton framework
   - Monte Carlo simulation methods
   - American option pricing

5. **Risk Management**
   - Value at Risk (VaR)
   - Greeks and sensitivity analysis
   - Dynamic hedging strategies

## Usage Examples

```python
from implementations.finance import bond_price, duration
from implementations.option_hedging import black_scholes_call, calculate_greeks

# Bond pricing
price = bond_price(face_value=1000, coupon_rate=0.05, yield_rate=0.04, maturity=10)
dur = duration(face_value=1000, coupon_rate=0.05, yield_rate=0.04, maturity=10)

# Options pricing
option_price = black_scholes_call(S=100, K=110, r=0.05, sigma=0.2, T=1)
greeks = calculate_greeks(S=100, K=110, r=0.05, sigma=0.2, T=1)
```

## Requirements

```bash
pip install numpy scipy matplotlib pandas
```

## Course Information

MATH 176 covers the mathematical foundations of modern finance, including:
- Discrete and continuous-time models
- Risk-neutral valuation
- Complete and incomplete markets
- Computational methods in finance

The materials span two academic years of teaching, providing both theoretical depth and practical implementation experience.