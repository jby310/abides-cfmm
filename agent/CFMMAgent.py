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
    Modified logic: is_buy_order=True uses Y to buy X, False sells X to get Y
    """
    
    # Class-level registry for static access
    _cfmm_instances = {}
    
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
        self.exchange_rate = 3500
        self.initial_x = np.sqrt(initial_k / self.exchange_rate)
        self.initial_y = self.initial_x * self.exchange_rate
        self.initial_k = initial_k
        self.initial_price = self.initial_y / self.initial_x

        self.x = self.initial_x  # Reserve of asset X (e.g., ETH)
        self.y = self.initial_y  # Reserve of asset Y (e.g., USDT)
        self.k = self.initial_k  # Constant product

        # Register instance for static access
        CFMMAgent._cfmm_instances[symbol] = self

        self.trade_history = []
    
    @classmethod
    def get_cfmm_instance(cls, symbol):
        """Static method to get CFMM instance by symbol"""
        return cls._cfmm_instances.get(symbol)
    
    @classmethod
    def get_cfmm_market_data(cls, symbol, levels=1):
        """
        Static interface to get CFMM market data
        Returns None if CFMM instance not found
        """
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None
        return cfmm_instance.get_market_data(levels)
    
    @classmethod
    def get_cfmm_pool_price(cls, symbol):
        """Static interface to get CFMM pool price"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None
        return cfmm_instance.get_pool_price()
    
    @classmethod
    def get_cfmm_reserves(cls, symbol):
        """Static interface to get CFMM reserves"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None
        return (cfmm_instance.x, cfmm_instance.y)
    
    @classmethod
    def get_cfmm_fee_rate(cls, symbol):
        """Static interface to get CFMM fee rate"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None
        return cfmm_instance.fee
    
    @classmethod
    def calculate_cfmm_trade_output(cls, symbol, input_amount, is_buy_order):
        """Static interface to calculate trade output"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return 0
        return cfmm_instance.calculate_trade_output(input_amount, is_buy_order)
    
    @classmethod
    def calculate_cfmm_effective_price(cls, symbol, trade_amount, is_buy_order):
        """Static interface to calculate effective price"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None, 1.0
        return cfmm_instance.calculate_effective_price(trade_amount, is_buy_order)
    
    @classmethod
    def get_cfmm_bid_ask_prices(cls, symbol):
        """Static interface to get CFMM bid and ask prices"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return None, None
        return cfmm_instance.calculate_bid_ask()
    
    @classmethod
    def get_cfmm_depth(cls, symbol):
        """Static interface to get CFMM depth"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return 0.0
        return cfmm_instance.depth_at_1pct()
    
    @classmethod
    def check_cfmm_reset_needed(cls, symbol):
        """Static interface to check if CFMM needs reset"""
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return False
        return cfmm_instance.needs_reset()
    
    @classmethod
    def execute_cfmm_trade_static(cls, symbol, agent_id, quantity, is_buy_order, current_time=None):
        """
        Static interface to execute CFMM trade
        Returns (executed_quantity, execution_price, fee_paid) or (0, 0, 0) if failed
        """
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return 0, 0, 0
        
        # Use instance method but add static recording
        return cfmm_instance.execute_trade(agent_id, quantity, is_buy_order, current_time)
    
    @classmethod
    def get_transacted_volume_static(cls, symbol, lookback_period='10min'):
        """ Used by any trading agent subclass to query the total transacted volume in a given lookback period """
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return False

        return cfmm_instance.get_transacted_volume(lookback_period)


    def get_transacted_volume(self, lookback_period='1s'):
        volume = 0
    
        # 将lookback_period转换为pandas Timedelta
        lookback_td = pd.Timedelta(lookback_period)
        current_time = pd.Timestamp.now()
        
        # 遍历交易历史，统计指定时间段内的交易量
        if self.trade_history is None:
            for trade in self.trade_history:
                if current_time - trade['timestamp'] <= lookback_td:
                    if self.symbol is None or trade['symbol'] == self.symbol:
                        volume += trade['amount']
            
            return volume
        else:
            return 0


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
        修改逻辑：is_buy_order=True 用Y买X，False 卖X得Y
        """
        if is_buy_order:
            # 用Y买X：输入Δy，输出Δx
            if self.x == 0 or input_amount <= 0:
                return 0
            
            phi = 1 - self.fee
            fee_adjusted_input = input_amount * phi
            
            # 解方程: Δx = x - k/(y + φ*Δy)
            delta_x = self.x - (self.k / (self.y + fee_adjusted_input))
            return max(0, delta_x)
        else:
            # 卖X得Y：输入Δx，输出Δy  
            if self.y == 0 or input_amount <= 0:
                return 0
            
            phi = 1 - self.fee
            fee_adjusted_input = input_amount * phi
            
            # 解方程: Δy = y - k/(x + φ*Δx)
            delta_y = self.y - (self.k / (self.x + fee_adjusted_input))
            return max(0, delta_y)
    
    def calculate_effective_price(self, trade_amount, is_buy_order):
        """
        Calculate effective price for a given trade size
        Returns effective price and price impact percentage
        """
        if trade_amount <= 0:
            pool_price = self.get_pool_price()
            return pool_price, 0
        
        if is_buy_order:
            # Buying X with Y: output is Δx, input is Δy
            output = self.calculate_trade_output(trade_amount, True)
            if output == 0:
                return float('inf'), 1.0
            effective_price = trade_amount / output  # Δy / Δx
        else:
            # Selling X for Y: output is Δy, input is Δx  
            output = self.calculate_trade_output(trade_amount, False)
            if output == 0:
                return 0, 1.0
            effective_price = output / trade_amount  # Δy / Δx
        
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

    @classmethod
    def reset_cfmm_pool(cls, symbol, reset_price):
        ''' Static interface to reset CFMM pool'''
        cfmm_instance = cls.get_cfmm_instance(symbol)
        if cfmm_instance is None:
            return
        cfmm_instance.reset_pool(reset_price)

    def reset_pool(self, reset_price=None):
        """Reset pool to CLOB Price"""
        log_print(f"CFMM {self.symbol}: Resetting pool from ({self.x:.2f}, {self.y:.2f}) to ({self.initial_x:.2f}, {self.initial_y:.2f})")
        if reset_price is not None:
            self.x = np.sqrt(self.k / reset_price)
            self.y = self.k / self.x
        else:
            self.x = self.initial_x
            self.y = self.initial_y
    
    def execute_trade(self, agent_id, quantity, is_buy_order, current_time):
        """
        Execute a trade in the CFMM pool with modified logic
        Returns: executed_quantity
        """
        # # Check if pool needs reset before trade as per Document 3
        # if self.needs_reset():
        #     self.reset_pool()
        
        if is_buy_order:
            # Buying X with Y: 对于CFMM来说是流出X，收入Y
            if quantity <= 0 or self.x == 0:
                return 0
            
            # Calculate output using modified formula
            delta_x = self.calculate_trade_output(quantity, True)
            if delta_x <= 0:
                return 0
            
            new_x = self.x - delta_x
            new_y = self.k / new_x
            
            # Update state
            self.x = new_x
            self.y = new_y
            
            log_print(f"CFMM {self.symbol}: BUY {delta_x:.4f} X for {quantity:.2f} Y")
            self.trade_history.append({
                'trade_time': current_time,
                'symbol': self.symbol,
                'amount': delta_x,
            })
            
            return delta_x
            
        else:
            # Selling X for Y: 对于CFMM来说是收入X，流出Y
            if quantity <= 0 or self.y == 0:
                return 0
            
            # Calculate output using modified formula
            delta_y = self.calculate_trade_output(quantity, False)
            if delta_y <= 0:
                return 0
            
            # Update reserves: (x + φ*Δx) * (y - Δy) = k
            new_y = self.y - delta_y
            new_x = self.k / new_y
            delta_x = new_x - self.x
            self.trade_history.append({
                'trade_time': current_time,
                'symbol': self.symbol,
                'amount': delta_x,
            })
            
            # Update state
            self.x = new_x
            self.y = new_y
            
            log_print(f"CFMM {self.symbol}: SELL {quantity:.4f} X for {delta_y:.2f} Y")
            
            return delta_y
    
    def depth_at_1pct(self):
        """Calculate depth at 1% price change"""
        if self.x == 0 or self.y == 0:
            return 0.0
        
        current_price = self.get_pool_price()
        
        # 压价 1% 需要买入的 Δx (用Y买X)
        delta_x_buy = self.x * (1/0.99 - 1)
        
        # 抬价 1% 需要卖出的 Δx (卖X得Y)  
        delta_x_sell = self.x * (1.01 - 1)
        
        # 取双向最小作为深度
        depth = min(delta_x_buy, delta_x_sell)
        
        return max(0, depth)
    
    def get_market_data(self, levels=1):
        """
        Get current market data
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
            'fee_rate': self.fee,
            'depth_1pct': depth
        }
    
    def receiveMessage(self, currentTime, msg):
        """Handle messages specific to CFMM"""
        super().receiveMessage(currentTime, msg)
        
        if msg.body['msg'] == "CFMM_TRADE_REQUEST":
            self.handleCFMMTrade(currentTime, msg)
        elif msg.body['msg'] == "CFMM_MARKET_DATA_REQUEST":
            self.handleCFMMMarketData(currentTime, msg)
    
    def handleCFMMTrade(self, currentTime, msg):
        """
        Handle trade requests from MarketOnlyAgent with modified logic
        """
        agent_id = msg.body['sender']
        symbol = msg.body['symbol']
        quantity = msg.body['quantity']  # Trade amount in the currency being spent
        is_buy_order = msg.body['is_buy_order']
        max_slippage = msg.body.get('max_slippage', 0.05)
        
        if symbol != self.symbol:
            log_print(f"CFMM {self.symbol}: Trade request discarded for wrong symbol: {symbol}")
            return
        
        # For CFMM trades, quantity is always in the currency being spent
        # is_buy_order=True: spending Y to buy X, quantity is in Y
        # is_buy_order=False: spending X to get Y, quantity is in X
        
        # Calculate price impact to check slippage
        effective_price, price_impact = self.calculate_effective_price(quantity, is_buy_order)
        
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
        
        if executed_qty == 0:
            log_print(f"CFMM {self.symbol}: Trade execution failed")
            self.sendMessage(agent_id, Message({
                "msg": "CFMM_TRADE_REJECTED",
                "symbol": symbol,
                "reason": "execution_failed"
            }))
            return
        
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