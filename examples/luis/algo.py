# Markowitz Portfolio Construction 


from pytz import timezone
from datetime import datetime, timedelta  
#from zipline.utils.tradingcalendar import get_early_closes
from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
import numpy as np
import scipy.optimize
import math
import random
from pylivetrader.api import(    symbols,
                            order_target_percent,
                            schedule_function,
                            date_rules,
                            time_rules,
                            get_datetime,
                            history,
                            record,
                            get_open_orders,
                            cancel_order,
                            set_max_position_size,
                            set_max_order_size,
                            set_max_order_count,
                            set_long_only,
                            symbol,
                            get_order,
                            order_target_value,
                            order_target,
                            order_percent,
                            order_value,
                            order,
                       )
def initialize(context):
    # INPUTS:
    # 1) Enter the risk tolerance on a scale of 1 to 20.
    #    1 is the lowest risk tolerance (lowest risk portfolio)
    #    20 is the highest risk tolerance (highest risk portfolio)
    # 2) Stock listing (manually or from set_universe)
    #    NOTE: To meet the 50 second requirement the number of 
    #    securities must be not more than 120 or about 1.5 % of the 
    #    universe.
    # 3) Do or do not use margin
    #    1 is no margin
    #    2 is double cash
    # 4) Frequency to rebalance

    # Configuration parameters
    context.risk_tolerance = 10
    #set_universe(universe.DollarVolumeUniverse(99.0, 100))
    context.securities =symbols( 'AAPL','gs', 'NVDA', 'AMZN')
    context.allowableMargin        = 1.0  # 
    context.requiredMargin         = 0.0  # In Dollars
    context.usedMargin             = 0.0  # In Dollars
    context.lastPortfolioUpdate    = datetime(1989,1,30,9,15,0,tzinfo=timezone('US/Eastern')) # randomly chosen init value
    context.trailingStopPortfolioValue  = 18000 # Based on starting cash of $20,000
    context.disableMarginPortfolioValue = 19000 # Based on starting cash of $20,000
    context.enableMarginPortfolioValue  = 22000 # Based on starting cash of $20,000
    context.rebalanceFrequency = 1        # Number of weeks
    
def handle_data(context, data):    
    # Morning Margin Check (UTC timezone) - LONG ONLY
    if get_datetime().hour == 14 and get_datetime().minute == 35:
        context.requiredMargin = marginRequirements(context.portfolio) * 3.0
        if context.portfolio.cash < 0.:
            context.usedMargin = abs(context.portfolio.cash)
        else:
            context.usedMargin = 0.
        if context.requiredMargin < context.usedMargin:
                log.warn('MARGIN REQUIREMENTS EXCEEDED. ' +\
                         'Used Margin = ' + str(context.usedMargin) +\
                         ' Allowable Margin = ' + str(context.requiredMargin))
        # Liquidate if total value falls 10% or more (disable margin use after 5% loss)
        if 0.9*(context.portfolio.positions_value+context.portfolio.cash) > context.trailingStopPortfolioValue:
            context.trailingStopPortfolioValue  = 0.90*(context.portfolio.positions_value+context.portfolio.cash)
            context.disableMarginPortfolioValue = 0.95*(context.portfolio.positions_value+context.portfolio.cash)
        if (context.portfolio.positions_value+context.portfolio.cash) < context.trailingStopPortfolioValue:
            log.warn('*** L I Q U I D A T E ***')
            liquidate(context.portfolio)
            context.trailingStopPortfolioValue  = 0.90*(context.portfolio.positions_value+context.portfolio.cash)
        if (context.portfolio.positions_value+context.portfolio.cash) < context.disableMarginPortfolioValue:
            log.info('*** MARGIN USE DISABLED ***')
            context.allowableMargin = 1.
            context.enableMarginPortfolioValue = 1.10*(context.portfolio.positions_value+context.portfolio.cash)
        elif (context.portfolio.positions_value+context.portfolio.cash) > context.enableMarginPortfolioValue:
            log.info('*** MARGIN USE ENABLED ***')
            context.allowableMargin = 2.
    
    # End of Day
    if get_datetime().hour == 20 and get_datetime().minute == 55:
        for stock in list(data.keys()):
            closeAnyOpenOrders(stock)
    
    #if loc_dt.month != context.previous_month:
###    if (get_datetime() - context.lastPortfolioUpdate) >= timedelta(weeks=context.rebalanceFrequency):
###        context.lastPortfolioUpdate = get_datetime()
###        log.debug('Number of secruities to be considered: ' + str(len(data.keys())))
    if get_datetime().hour == 14 and get_datetime().minute == 35:
        all_prices = history(250, '1d', 'price')
        daily_returns = all_prices.pct_change().dropna()
                
        dr = np.array(daily_returns)
        (rr,cc) = dr.shape
        
        expreturns, covars = assets_meanvar(dr, list(data.keys()))
        R = expreturns
        C = covars
        rf = 0.015
        expreturns = np.array(expreturns)
        
        frontier_mean, frontier_var, frontier_weights = solve_frontier(R, C, rf,context)
        
        f_w = array(frontier_weights)          
        (row_1, col_1) = f_w.shape         
  
        # Choose an allocation along the efficient frontier
        wts = frontier_weights[context.risk_tolerance]
        new_weights = wts  
            
        # Set leverage to 1
        leverage = sum(abs(new_weights))
        portfolio_value = (context.portfolio.positions_value + context.portfolio.cash)/leverage
        record(PV=portfolio_value)
        record(Cash=context.portfolio.cash)
    
        # Reweight portfolio 
        i = 0
        for sec in list(data.keys()):
            if wts[i] < 0.01:
                wts[i] = 0.0
            if wts[i] < 0.01 and context.portfolio.positions[sec].amount == 0:
                i = i+1
                continue
            order_target_percent(sec, wts[i],None,None)
            log.info('Adjusting ' + str(sec) + ' to ' + str(wts[i]*100.0) + '%')
            i=i+1

#################################################################################
#
# HELPER FUNCTIONS ( This algorithm's math functions)
#
#################################################################################                     

# Compute expected return on portfolio.
def compute_mean(W,R):
    return sum(R*W)

# Compute the variance of the portfolio.
def compute_var(W,C):
    return dot(dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation. 
def compute_mean_var(W, R, C):
    return compute_mean(W, R), compute_var(W, C)

def fitness(W, R, C, r):
    # For given level of return r, find weights which minimizes portfolio variance.
    mean_1, var = compute_mean_var(W, R, C)
    # Penalty for not meeting stated portfolio return effectively serves as optimization constraint
    # Here, r is the 'target' return
    penalty = (1/100)*abs(mean_1-r)
    return var + penalty
    
# Given risk-free rate, asset returns, and covariances, this function
# calculates the weights of the tangency portfolio with regard to Sharpe
# ratio maximization.

def fitness_sharpe(W, R, C, rf):
    mean_1, var = compute_mean_var(W, R, C)
    utility = (mean_1 - rf)/sqrt(var)
    return 1/utility

# Solves for the optimal portfolio weights using the Sharpe ratio 
# maximization objective function.

def solve_weights(R, C, rf,context):
    n = len(R)
    W = ones([n])/n # Start optimization with equal weights
    b_ = [(0.,1.) for i in range(n)] # Bounds for decision variables
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-context.allowableMargin }) 
    # Constraints - weights must sum to 1
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x    

# Solve for the efficient frontier using the variance + penalty minimization
# function fitness. 

def solve_frontier(R, C, rf,context):
    frontier_mean, frontier_var, frontier_weights = [], [], []
    n = len(R)      # Number of assets in the portfolio
    for r in linspace(max(min(R), rf), max(R), num=20): # Iterate through the range of returns on Y axis
        W = ones([n])/n # Set initial guess for weights
        b_ = [(0,1) for i in range(n)] # Set bounds on weights
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-context.allowableMargin }) # Set constraints
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)  
        #if not optimized.success:
        #    raise BaseException(optimized.message)s
        # Add point to the efficient frontier
        frontier_mean.append(r)
        frontier_var.append(compute_var(optimized.x, C))   # Min-variance based on optimized weights
        frontier_weights.append(optimized.x)
    return array(frontier_mean), array(frontier_var), frontier_weights
        
# Weights - array of asset weights (derived from market capitalizations)
# Expreturns - expected returns based on historical data
# Covars - covariance matrix of asset returns based on historical data

def assets_meanvar(daily_returns, stockList):    
    
    # Calculate expected returns
    expreturns = array([])
    daily_returns = daily_returns.transpose()
    
    for i in range(0, len(stockList)):
        expreturns = append(expreturns, mean(daily_returns[i,:]))
    
    # Compute covariance matrix
    covars = cov(daily_returns)
    # Annualize expected returns and covariances
    # Assumes 255 trading days per year    
    expreturns = (1+expreturns)**255-1
    covars = covars * 255
    expreturns = np.array(expreturns)
    
    return expreturns, covars
            
#################################################################################
#
# HELPER FUNCTIONS ( Portfolio / Stock Management)
#
#################################################################################         

def closeAnyOpenOrders(stock):
    orders = get_open_orders(stock)
    if orders:
        for order in orders:
             message = 'Canceling order for {amount} shares in {stock}'  
             message = message.format(amount=order.amount, stock=stock)  
             #log.debug(message)
             cancel_order(order)
def hasOrders(stock):
    hasOrders = False
    orders = get_open_orders(stock)
    if orders:
        for order in orders:
             message = 'Open order for {amount} shares in {stock}'  
             message = message.format(amount=order.amount, stock=stock)  
             log.info(message)
             hasOrders = True
    return hasOrders   
def liquidate(portfolio):
    for stock in portfolio.positions:
        closeAnyOpenOrders(stock)
        if portfolio.positions[stock].amount > 0:
         order(stock,-portfolio.positions[stock].amount,None,None)
         log.info('Sold stake in ' + str(stock) +
                  ' @ $' + str(portfolio.positions[stock].last_sale_price) +
                  ' Cost Basis: ' + str(portfolio.positions[stock].cost_basis) )
def marginRequirements(portfolio):  
    req = 0  
    for stock in portfolio.positions:  
        amount = portfolio.positions[stock].amount  
        last_price = portfolio.positions[stock].last_sale_price  
        if amount > 0:  
            req += .25 * amount * last_price  
        elif amount < 0:  
            if last_price < 5:  
                req += max(2.5 * amount, abs(amount * last_price))  
            else:  
                req += max(5 * amount, abs(0.3 * amount * last_price))  
    return req
