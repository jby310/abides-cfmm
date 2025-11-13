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
    Strictly follows Document 3 specifications
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
        
        # Track initial values for reset as per Document 3
        self.initial_x = np.sqrt(initial_k)
        self.initial_y = np.sqrt(initial_k)
        self.initial_k = initial_k
        self.initial_price = self.initial_y / self.initial_x

        self.x = self.initial_x  # Reserve of asset X (e.g., ETH)
        self.y = self.initial_y  # Reserve of asset Y (e.g., USDT)
        self.k = self.initial_k  # Constant product

        # Transaction history for volume calculation
        self.transaction_history = []
        
        # Market data subscribers
        self.subscription_dict = {}
        
    def get_pool_price(self):
        """Calculate current pool price (y/x) as per Document 3"""
        if self.x == 0:
            return float('inf')
        return self.y / self.x
    
    def calculate_bid_ask(self):
        """
        Calculate bid and ask prices following Document 3 formulas
        phi = 1 - fee_rate
        bid_price = pool_price * phi
        ask_price = pool_price / phi
        """
        pool_price = self.get_pool_price()
        phi = 1 - self.fee
        
        bid_price = pool_price * phi  # Buy from traders
        ask_price = pool_price / phi  # Sell to traders
        
        return bid_price, ask_price
    
    def calculate_trade_output(self, input_amount, is_buy_order):
        """
        Calculate output amount for a given input amount following Document 3 formulas
        is_buy_order: True if buying X (paying Y), False if selling X (receiving Y)
        """
        if is_buy_order:
            # Buying X with Y: input is Δy, output is Δx
            # Formula: (x - Δx) * (y + φ*Δy) = k
            if self.x == 0 or input_amount <= 0:
                return 0
            
            phi = 1 - self.fee
            fee_adjusted_input = input_amount * phi
            
            # Solve for Δx: Δx = x - k/(y + φ*Δy)
            delta_x = self.x - (self.k / (self.y + fee_adjusted_input))
            return max(0, delta_x)
        else:
            # Selling X for Y: input is Δx, output is Δy
            # Formula: (x + φ*Δx) * (y - Δy) = k
            if self.y == 0 or input_amount <= 0:
                return 0
            
            phi = 1 - self.fee
            fee_adjusted_input = input_amount * phi
            
            # Solve for Δy: Δy = y - k/(x + φ*Δx)
            delta_y = self.y - (self.k / (self.x + fee_adjusted_input))
            return max(0, delta_y)
    
    def calculate_effective_price(self, trade_amount, is_buy_order):
        """
        Calculate effective price for a given trade size following Document 3
        Returns effective price and price impact percentage
        """
        if trade_amount <= 0:
            pool_price = self.get_pool_price()
            return pool_price, 0
        
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
        if pool_price == 0:
            return effective_price, 1.0
            
        price_impact = abs(effective_price - pool_price) / pool_price
        
        return effective_price, price_impact
    
    def needs_reset(self):
        """Check if pool needs reset based on price deviation as per Document 3"""
        current_price = self.get_pool_price()
        if current_price == 0 or self.initial_price == 0:
            return False
            
        price_deviation = abs(current_price - self.initial_price) / self.initial_price
        return price_deviation > self.reset_threshold
    
    def reset_pool(self):
        """Reset pool to initial values as per Document 3"""
        log_print(f"CFMM {self.symbol}: Resetting pool from ({self.x:.2f}, {self.y:.2f}) to ({self.initial_x:.2f}, {self.initial_y:.2f})")
        self.x = self.initial_x
        self.y = self.initial_y
        self.k = self.initial_k
    
    def execute_trade(self, agent_id, quantity, is_buy_order):
        """
        Execute a trade in the CFMM pool following Document 3 formulas
        Returns: (executed_quantity, execution_price, fee_paid)
        """
        # Check if pool needs reset before trade as per Document 3
        if self.needs_reset():
            self.reset_pool()
        
        if is_buy_order:
            # Buying X with Y: paying Δy, receiving Δx
            if quantity <= 0 or self.x == 0:
                return 0, 0, 0
            
            # Calculate output using Document 3 formula
            delta_x = self.calculate_trade_output(quantity, True)
            if delta_x <= 0:
                return 0, 0, 0
            
            # Apply fee and update reserves
            phi = 1 - self.fee
            fee_amount = quantity * self.fee
            actual_input = quantity * phi
            
            # Update reserves: (x - Δx) * (y + φ*Δy) = k
            new_x = self.x - delta_x
            new_y = self.y + actual_input
            new_k = new_x * new_y
            
            # Update state
            self.x = new_x
            self.y = new_y
            self.k = new_k
            
            execution_price = quantity / delta_x
            log_print(f"CFMM {self.symbol}: BUY {delta_x:.4f} X for {quantity:.2f} Y, price: {execution_price:.4f}, fee: {fee_amount:.2f}")
            
            return delta_x, execution_price, fee_amount
            
        else:
            # Selling X for Y: paying Δx, receiving Δy
            if quantity <= 0 or self.y == 0:
                return 0, 0, 0
            
            # Calculate output using Document 3 formula
            delta_y = self.calculate_trade_output(quantity, False)
            if delta_y <= 0:
                return 0, 0, 0
            
            # Apply fee and update reserves
            phi = 1 - self.fee
            fee_amount = delta_y * self.fee
            actual_output = delta_y * phi
            
            # Update reserves: (x + φ*Δx) * (y - Δy) = k
            new_x = self.x + (quantity * phi)
            new_y = self.y - delta_y
            new_k = new_x * new_y
            
            # Update state
            self.x = new_x
            self.y = new_y
            self.k = new_k
            
            execution_price = delta_y / quantity
            log_print(f"CFMM {self.symbol}: SELL {quantity:.4f} X for {delta_y:.2f} Y, price: {execution_price:.4f}, fee: {fee_amount:.2f}")
            
            return quantity, execution_price, fee_amount
    
    def depth_at_1pct(self):
        """
        Calculate depth at 1% price movement as per Document 3
        Depth: the trade volume required to move price by 1%
        """
        if self.x == 0 or self.y == 0:
            return 0.0
        
        # For buying X (price increases by 1%)
        # To increase price by 1%, we need to calculate Δx needed
        current_price = self.get_pool_price()
        target_price = current_price * 1.01
        
        # Using the formula: new_y / new_x = target_price
        # And new_x * new_y = k (approximately, ignoring fees for depth calculation)
        # We can solve for the required trade
        new_x = np.sqrt(self.k / target_price)
        delta_x_buy = self.x - new_x
        
        # For selling X (price decreases by 1%)
        target_price = current_price * 0.99
        new_x = np.sqrt(self.k / target_price)
        delta_x_sell = new_x - self.x
        
        # Take the minimum of both directions as depth
        depth = min(abs(delta_x_buy), abs(delta_x_sell))
        
        return max(0, depth)
    
    def get_market_data(self, levels=1):
        """
        Get current market data following Document 3 specifications
        """
        bid_price, ask_price = self.calculate_bid_ask()
        depth = self.depth_at_1pct()
        
        # Create depth levels
        bids = [(bid_price, depth)] if levels > 0 else []
        asks = [(ask_price, depth)] if levels > 0 else []
        
        return {
            'bids': bids,
            'asks': asks,
            'last_trade': self.get_pool_price(),
            'pool_reserves': (self.x, self.y),
            'fee_rate': self.fee,  # Include fee rate as per Document 3
            'depth_1pct': depth
        }
    
    def receiveMessage(self, currentTime, msg):
        """Handle messages specific to CFMM"""
        super().receiveMessage(currentTime, msg)
        
        if msg.body['msg'] == "CFMM_TRADE_REQUEST":
            self.handleCFMMTrade(currentTime, msg)
        elif msg.body['msg'] == "CFMM_MARKET_DATA_REQUEST":
            self.handleCFMMMarketData(currentTime, msg)
        elif msg.body['msg'] == "CFMM_SUBSCRIPTION_REQUEST":
            self.handleCFMMSubscription(currentTime, msg)
    
    def handleCFMMTrade(self, currentTime, msg):
        """
        Handle trade requests from MarketOnlyAgent following Document 3
        """
        agent_id = msg.body['sender']
        symbol = msg.body['symbol']
        quantity = msg.body['quantity']  # This is the trade amount in quote currency (Y)
        is_buy_order = msg.body['is_buy_order']
        max_slippage = msg.body.get('max_slippage', 0.05)
        
        if symbol != self.symbol:
            log_print(f"CFMM {self.symbol}: Trade request discarded for wrong symbol: {symbol}")
            return
        
        # Convert quote currency amount to base currency if needed
        # For CFMM, we need to handle the trade based on the pool mechanics
        if is_buy_order:
            # Buying X with Y: quantity is in Y (quote currency)
            # We'll execute the trade directly with the Y amount
            trade_amount_y = quantity
        else:
            # Selling X for Y: quantity is the value in Y we want to get
            # We need to calculate how much X to sell to get this Y amount
            current_price = self.get_pool_price()
            if current_price > 0:
                trade_amount_x = quantity / current_price
            else:
                trade_amount_x = 0
            trade_amount_y = quantity
        
        # Calculate price impact to check slippage
        effective_price, price_impact = self.calculate_effective_price(
            trade_amount_y if is_buy_order else trade_amount_x, 
            is_buy_order
        )
        
        if price_impact > max_slippage:
            log_print(f"CFMM {self.symbol}: Trade rejected due to slippage {price_impact:.2%} > {max_slippage:.2%}")
            self.sendMessage(agent_id, Message({
                "msg": "CFMM_TRADE_REJECTED",
                "symbol": symbol,
                "reason": "slippage_exceeded",
                "slippage": price_impact
            }))
            return
        
        # Execute trade - using the Y amount for both buy and sell
        # For sells, we calculate the equivalent X amount inside execute_trade
        executed_qty, execution_price, fee = self.execute_trade(
            agent_id, 
            trade_amount_y,  # Always use Y amount for consistency
            is_buy_order
        )
        
        if executed_qty == 0:
            log_print(f"CFMM {self.symbol}: Trade execution failed")
            self.sendMessage(agent_id, Message({
                "msg": "CFMM_TRADE_REJECTED",
                "symbol": symbol,
                "reason": "execution_failed"
            }))
            return
        
        # Record transaction
        self.transaction_history.append({
            'time': currentTime,
            'agent_id': agent_id,
            'symbol': symbol,
            'side': 'BUY' if is_buy_order else 'SELL',
            'quantity': executed_qty,
            'price': execution_price,
            'fee': fee,
            'input_amount': quantity
        })
        
        # Send confirmation
        self.sendMessage(agent_id, Message({
            "msg": "CFMM_TRADE_EXECUTED",
            "symbol": symbol,
            'is_buy_order': is_buy_order,
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
    
    def get_pool_statistics(self):
        """Get comprehensive pool statistics"""
        return {
            'x_reserve': self.x,
            'y_reserve': self.y,
            'pool_price': self.get_pool_price(),
            'constant_product': self.k,
            'fee_rate': self.fee,
            'reset_threshold': self.reset_threshold,
            'transaction_count': len(self.transaction_history),
            'total_volume': sum(tx['quantity'] for tx in self.transaction_history)
        }