from pylivetrader.api import order_target, record, symbol


def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    # Here are any algorithm 'constants' we'll be using
    context.target_leverage = 1.0
    
    # Here are the ETFs we want to trade along with the weights 
    # Ensure they add to 1.00
    context.etfs = {
        symbol('TYD'): 0.1, # Daily 7-10 Year Treasury Bull 3X Shares
        symbol('TMF'): 0.2, # Daily 20+ Year Treasury Bull 3X Shares
        symbol('EDZ'): 0.2, # Daily MSCI Emerging Markets Bear 3X Shares
        symbol('SPXL'): 0.5, # Daily S&P 500 Bull 3X Shares

    }
    
    # Set commision model for Robinhood
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
 
    # Rebalance our portfolio to maintain target weights
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes = 35))


def rebalance(context, data):
    for stock, weight in context.etfs.items():
        order_target_percent(stock, weight*context.target_leverage)
