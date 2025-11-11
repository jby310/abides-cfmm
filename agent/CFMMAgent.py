import pandas as pd
import numpy as np
from agent.ExchangeAgent import ExchangeAgent
from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import log_print
from copy import deepcopy

class CFMMAgent(ExchangeAgent):
    """
    Constant Function Market Maker Agent implementing AMM functionality
    """
    
    def __init__(self, id, name, type, mkt_open, mkt_close, symbol, 
                 initial_k=10000000, fee=0.003, 
                 reset_threshold=0.1, pipeline_delay=40000,
                 computation_delay=1, log_orders=False, random_state=None):
        
        super().__init__(id, name, type, mkt_open, mkt_close, [symbol], 
                        pipeline_delay=pipeline_delay, computation_delay=computation_delay,
                        log_orders=log_orders, random_state=random_state)
        
        # CFMM specific parameters
        self.symbol = symbol
        self.fee = fee  # Trading fee (0.3%)
        self.reset_threshold = reset_threshold  # Price deviation threshold for reset
        
        # Track initial values for reset
        self.initial_x = np.sqrt(initial_k)
        self.initial_y = np.sqrt(initial_k)
        self.initial_k = initial_k
        
        self.x = self.initial_x  # Reserve of asset X (e.g., ETH)
        self.y = self.initial_y  # Reserve of asset Y (e.g., USDT)
        self.k = self.initial_k  # Constant product

        # Transaction history for volume calculation
        self.transaction_history = []
        
        # Market data subscribers (similar to CLOB)
        self.subscription_dict = {}
        
    def get_pool_price(self):
        """Calculate current pool price (y/x)"""
        if self.x == 0:
            return float('inf')
        return self.y / self.x
    
    def calculate_bid_ask(self, phi=0.99):
        """
        Calculate bid and ask prices with spread factor phi
        phi: spread factor (0 < phi < 1), smaller phi means larger spread
        """
        pool_price = self.get_pool_price()
        bid_price = pool_price * phi  # Buy from traders
        ask_price = pool_price / phi  # Sell to traders
        return bid_price, ask_price
    
    def calculate_trade_output(self, input_amount, is_buy_order):
        """
        Calculate output amount for a given input amount
        is_buy_order: True if buying X (paying Y), False if selling X (receiving Y)
        """
        if is_buy_order:
            # Buying X with Y: input is Δy, output is Δx
            if self.x == 0:
                return 0
            fee_adjusted_input = input_amount * (1 - self.fee)
            delta_x = self.x - (self.k / (self.y + fee_adjusted_input))
            return max(0, delta_x)
        else:
            # Selling X for Y: input is Δx, output is Δy
            if self.y == 0:
                return 0
            fee_adjusted_input = input_amount * (1 - self.fee)
            delta_y = self.y - (self.k / (self.x + fee_adjusted_input))
            return max(0, delta_y)
    
    def calculate_price_impact(self, trade_amount, is_buy_order):
        """
        Calculate price impact for a given trade size
        Returns effective price and price impact percentage
        """
        if trade_amount <= 0:
            return self.get_pool_price(), 0
        
        if is_buy_order:
            output = self.calculate_trade_output(trade_amount, True)
            if output == 0:
                return float('inf'), 1.0
            effective_price = trade_amount / output
        else:
            output = self.calculate_trade_output(trade_amount, False)
            if output == 0:
                return 0, 1.0
            effective_price = output / trade_amount
        
        pool_price = self.get_pool_price()
        price_impact = abs(effective_price - pool_price) / pool_price
        
        return effective_price, price_impact
    
    def needs_reset(self):
        """Check if pool needs reset based on price deviation"""
        current_price = self.get_pool_price()
        initial_price = self.initial_y / self.initial_x
        price_deviation = abs(current_price - initial_price) / initial_price
        return price_deviation > self.reset_threshold
    
    def reset_pool(self):
        """Reset pool to initial values"""
        log_print(f"CFMM {self.symbol}: Resetting pool from ({self.x}, {self.y}) to ({self.initial_x}, {self.initial_y})")
        self.x = self.initial_x
        self.y = self.initial_y
        self.k = self.initial_k
    
    def execute_trade(self, agent_id, quantity, is_buy_order, limit_price=None):
        """
        Execute a trade in the CFMM pool
        Returns: (executed_quantity, execution_price, fee_paid)
        """
        # Check if pool needs reset before trade
        if self.needs_reset():
            self.reset_pool()
        
        if is_buy_order:
            # Buying X with Y
            if quantity <= 0 or self.x == 0:
                return 0, 0, 0
            
            # Calculate output and update reserves
            delta_x = self.calculate_trade_output(quantity, True)
            if delta_x <= 0:
                return 0, 0, 0
            
            # Apply fee
            fee_amount = quantity * self.fee
            actual_input = quantity - fee_amount
            
            # Update reserves
            self.y += actual_input
            self.x -= delta_x
            self.k = self.x * self.y  # k changes due to fees
            
            execution_price = quantity / delta_x
            log_print(f"CFMM {self.symbol}: BUY {delta_x:.4f} X for {quantity:.2f} Y, price: {execution_price:.4f}")
            
            return delta_x, execution_price, fee_amount
            
        else:
            # Selling X for Y
            if quantity <= 0 or self.y == 0:
                return 0, 0, 0
            
            # Calculate output and update reserves
            delta_y = self.calculate_trade_output(quantity, False)
            if delta_y <= 0:
                return 0, 0, 0
            
            # Apply fee
            fee_amount = delta_y * self.fee
            actual_output = delta_y - fee_amount
            
            # Update reserves
            self.x += quantity
            self.y -= actual_output
            self.k = self.x * self.y  # k changes due to fees
            
            execution_price = delta_y / quantity
            log_print(f"CFMM {self.symbol}: SELL {quantity:.4f} X for {delta_y:.2f} Y, price: {execution_price:.4f}")
            
            return quantity, execution_price, fee_amount
    
    def depth_at_1pct(self):
        """Depth: 把价格推离 1 % 所需交易量（无手续费理论值）"""
        if self.x == 0:
            return 0.0
        dx_bid = self.x * (1.0 / 0.99 - 1.0)   # 压价 1 %
        dx_ask = self.x * (1.0 - 1.0 / 1.01)   # 抬价 1 % (这里选择x，不选y是为了保持单位一致)
        return min(dx_bid, dx_ask)
    
    def get_market_data(self, levels=1):
        """Get current market data (similar to CLOB but for CFMM)"""
        bid_price, ask_price = self.calculate_bid_ask()
        
        # # For CFMM, depth is theoretical based on price impact
        # bid_depth = self.calculate_trade_output(10000, False)  # Depth for 10k units
        # ask_depth = self.calculate_trade_output(10000, True)   # Depth for 10k units
        bid_depth = ask_depth = self.depth_at_1pct()
        
        return {
            'bids': [(bid_price, bid_depth)] if levels > 0 else [],
            'asks': [(ask_price, ask_depth)] if levels > 0 else [],
            'last_trade': self.get_pool_price(),
            'pool_reserves': (self.x, self.y)
        }
    
    def receiveMessage(self, currentTime, msg):
        """Handle messages specific to CFMM"""
        if msg.body['msg'] == "CFMM_TRADE_REQUEST":
            self.handleCFMMTrade(currentTime, msg)
        elif msg.body['msg'] == "CFMM_MARKET_DATA_REQUEST":
            self.handleCFMMMarketData(currentTime, msg)
        elif msg.body['msg'] == "CFMM_SUBSCRIPTION_REQUEST":
            self.handleCFMMSubscription(currentTime, msg)
    
    def handleCFMMTrade(self, currentTime, msg):
        """Handle trade requests from MarketOnlyAgent"""
        agent_id = msg.body['sender']
        symbol = msg.body['symbol']
        quantity = msg.body['quantity']
        is_buy_order = msg.body['is_buy_order']
        max_slippage = msg.body.get('max_slippage', 0.05)  # Default 5% slippage
        
        if symbol != self.symbol:
            log_print(f"CFMM {self.symbol}: Trade request discarded for wrong symbol: {symbol}")
            return
        
        # Calculate price impact to check slippage
        effective_price, price_impact = self.calculate_price_impact(quantity, is_buy_order)
        
        if price_impact > max_slippage:
            log_print(f"CFMM {self.symbol}: Trade rejected due to slippage {price_impact:.2%} > {max_slippage:.2%}")
            self.sendMessage(agent_id, Message({
                "msg": "CFMM_TRADE_REJECTED",
                "symbol": symbol,
                "reason": "slippage_exceeded",
                "slippage": price_impact
            }))
            return
        
        # Execute trade
        executed_qty, execution_price, fee = self.execute_trade(agent_id, quantity, is_buy_order)
        
        # Record transaction
        self.transaction_history.append({
            'time': currentTime,
            'agent_id': agent_id,
            'symbol': symbol,
            'side': 'BUY' if is_buy_order else 'SELL',
            'quantity': executed_qty,
            'price': execution_price,
            'fee': fee
        })
        
        # Send confirmation
        self.sendMessage(agent_id, Message({
            "msg": "CFMM_TRADE_EXECUTED",
            "symbol": symbol,
            "quantity": executed_qty,
            "price": execution_price,
            "fee": fee,
            "slippage": price_impact
        }))
        
        # Notify subscribers of market data update
        self.publishCFMMData()
    
    def handleCFMMMarketData(self, currentTime, msg):
        """Handle market data requests"""
        agent_id = msg.body['sender']
        symbol = msg.body['symbol']
        levels = msg.body.get('levels', 1)
        
        market_data = self.get_market_data(levels)
        
        self.sendMessage(agent_id, Message({
            "msg": "CFMM_MARKET_DATA",
            "symbol": symbol,
            "data": market_data
        }))
    
    def handleCFMMSubscription(self, currentTime, msg):
        """Handle market data subscription requests"""
        agent_id = msg.body['sender']
        symbol = msg.body['symbol']
        freq = msg.body.get('freq', 0)  # Default: all updates
        
        if agent_id not in self.subscription_dict:
            self.subscription_dict[agent_id] = {}
        
        self.subscription_dict[agent_id][symbol] = {
            'freq': freq,
            'last_update': currentTime
        }
    
    def publishCFMMData(self):
        """Publish market data to subscribers"""
        market_data = self.get_market_data()
        
        for agent_id, subscriptions in self.subscription_dict.items():
            for symbol, params in subscriptions.items():
                freq = params['freq']
                last_update = params['last_update']
                
                if freq == 0 or (self.currentTime - last_update).delta >= freq:
                    self.sendMessage(agent_id, Message({
                        "msg": "CFMM_MARKET_DATA",
                        "symbol": symbol,
                        "data": market_data
                    }))
                    self.subscription_dict[agent_id][symbol]['last_update'] = self.currentTime
    
    def get_transacted_volume(self, lookback_period='10min'):
        """Calculate transacted volume for given lookback period"""
        lookback_delta = pd.Timedelta(lookback_period)
        cutoff_time = self.currentTime - lookback_delta
        
        volume = sum(tx['quantity'] for tx in self.transaction_history 
                    if tx['time'] >= cutoff_time)
        
        return volume
