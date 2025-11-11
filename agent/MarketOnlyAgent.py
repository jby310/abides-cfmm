from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import log_print
import pandas as pd
import numpy as np
import random

class MarketOnlyAgent(TradingAgent):
    """
    Market Only Agent that trades based on amount rather than price/quantity
    Routes orders intelligently between CLOB and CFMM venues
    """
    
    def __init__(self, id, name, type, symbol, starting_cash=100000, 
                 max_slippage=0.05, wake_up_freq='60s', min_trade_size=100,
                 log_orders=False, random_state=None):
        
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state)
        
        self.symbol = symbol
        self.max_slippage = max_slippage  # Maximum acceptable slippage (e.g., 5%)
        self.wake_up_freq = wake_up_freq
        self.min_trade_size = min_trade_size  # Minimum trade size to avoid dust
        
        # Venue identifiers
        self.clob_exchange_id = None
        self.cfmm_exchange_id = None
        
        # State tracking
        self.state = 'AWAITING_WAKEUP'
        self.pending_queries = 0
        self.current_strategy = None
        
        # Price data from both venues
        self.clob_data = None
        self.cfmm_data = None
        
        # Trading statistics
        self.trade_history = []
        self.venue_preference = {'CLOB': 0, 'CFMM': 0}
        
        # 可以在这里设置交易金额的随机范围，比如1到100
        self.min_trade_amount = 1
        self.max_trade_amount = 10000
        self.trade_amount = self.get_random_trade_amount()
        
    def get_random_trade_amount(self):
        # 生成指定范围内的随机整数作为交易金额
        return random.randint(self.min_trade_amount, self.max_trade_amount)
        
    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        
        # Find both CLOB and CFMM exchanges
        from agent.ExchangeAgent import ExchangeAgent
        from agent.CFMMAgent import CFMMAgent  # Assuming CFMMAgent is in cfmm_agent.py
        
        self.clob_exchange_id = self.kernel.findAgentByType(ExchangeAgent)
        self.cfmm_exchange_id = self.kernel.findAgentByType(CFMMAgent)
        
        log_print(f"MarketOnlyAgent {self.id}: CLOB ID: {self.clob_exchange_id}, CFMM ID: {self.cfmm_exchange_id}")

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        
        if not can_trade:
            return
            
        self.state = 'QUERYING_MARKETS'
        self.pending_queries = 0
        
        # Reset price data
        self.clob_data = None
        self.cfmm_data = None
        
        self.setWakeup(currentTime + self.getWakeFrequency())
        
        # Query both markets
        if self.clob_exchange_id is not None:
            self.getCurrentSpread(self.symbol, depth=5)  # Get multiple levels for depth analysis
            self.pending_queries += 1
            
        if self.cfmm_exchange_id is not None:
            self.queryCFMMMarketData()
            self.pending_queries += 1
            
        if self.pending_queries == 0:
            log_print(f"MarketOnlyAgent {self.id}: No venues available")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())

    def queryCFMMMarketData(self):
        """Query CFMM for current market data"""
        self.sendMessage(self.cfmm_exchange_id,  Message({
            "msg": "CFMM_MARKET_DATA_REQUEST",
            "sender": self.id,
            "symbol": self.symbol,
            "levels": 5  # Get multiple depth levels
        }))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        if self.state == 'QUERYING_MARKETS':
            if msg.body['msg'] == 'QUERY_SPREAD':
                self.handleCLOBData(currentTime, msg)
            elif msg.body['msg'] == 'CFMM_MARKET_DATA':
                self.handleCFMMData(currentTime, msg)
                
            self.checkAllDataReceived(currentTime)
            
        elif self.state == 'EXECUTING_TRADE':
            if msg.body['msg'] in ['ORDER_EXECUTED', 'CFMM_TRADE_EXECUTED']:
                self.handleTradeExecution(currentTime, msg)
            elif msg.body['msg'] in ['ORDER_REJECTED', 'CFMM_TRADE_REJECTED']:
                self.handleTradeRejection(currentTime, msg)

    def handleCLOBData(self, currentTime, msg):
        """Process CLOB market data"""
        self.clob_data = {
            'bids': msg.body['bids'],  # List of (price, quantity) tuples
            'asks': msg.body['asks'],  # List of (price, quantity) tuples
            'timestamp': currentTime,
            'last_trade': msg.body['data']
        }
        
        log_print(f"MarketOnlyAgent {self.id}: CLOB data received - "
                 f"Best Bid: {self.clob_data['bids'][0] if self.clob_data['bids'] else 'N/A'}, "
                 f"Best Ask: {self.clob_data['asks'][0] if self.clob_data['asks'] else 'N/A'}")

    def handleCFMMData(self, currentTime, msg):
        """Process CFMM market data"""
        data = msg.body['data']
        self.cfmm_data = {
            'bids': data['bids'],  # List of (price, quantity) tuples
            'asks': data['asks'],  # List of (price, quantity) tuples
            'timestamp': currentTime,
            'pool_reserves': data.get('pool_reserves', (0, 0)),
            'last_trade': data.get('last_trade', 0)
        }
        
        log_print(f"MarketOnlyAgent {self.id}: CFMM data received - "
                 f"Pool Price: {self.cfmm_data['last_trade']:.4f}, "
                 f"Reserves: {self.cfmm_data['pool_reserves']}")

    def checkAllDataReceived(self, currentTime):
        """Check if we have data from all venues and proceed to trading"""
        self.pending_queries -= 1
        
        if self.pending_queries > 0:
            return
            
        # Both queries completed, analyze and trade
        self.analyzeMarketsAndTrade(currentTime)

    def analyzeMarketsAndTrade(self, currentTime):
        """Analyze both markets and execute optimal trade"""
        if not self.clob_data and not self.cfmm_data:
            log_print(f"MarketOnlyAgent {self.id}: No market data available")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())
            return
            
        # Determine trading direction (random for demonstration)
        # In real implementation, this could be based on strategy
        is_buy_order = self.random_state.choice([True, False])
        action = "BUY" if is_buy_order else "SELL"
        
        log_print(f"MarketOnlyAgent {self.id}: Planning to {action} {self.trade_amount} worth of {self.symbol}")
        
        # Compare venues and execute trade
        venue_analysis = self.compareVenues(is_buy_order)
        
        if venue_analysis['best_venue']:
            self.executeTrade(currentTime, is_buy_order, venue_analysis)
        else:
            log_print(f"MarketOnlyAgent {self.id}: No suitable venue found")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())

    def compareVenues(self, is_buy_order):
        """Compare CLOB and CFMM for best execution"""
        analysis = {
            'clob_effective_price': None,
            'cfmm_effective_price': None,
            'clob_max_qty': 0,
            'cfmm_max_qty': 0,
            'clob_slippage': 0,
            'cfmm_slippage': 0,
            'best_venue': None,
            'split_trade': False,
            'split_ratios': None
        }
        
        if is_buy_order:
            # Buying X with Y
            analysis.update(self.analyzeBuyOpportunities())
        else:
            # Selling X for Y
            analysis.update(self.analyzeSellOpportunities())
            
        return analysis

    def analyzeBuyOpportunities(self):
        """Analyze opportunities for buying asset X"""
        analysis = {}
        
        # Analyze CLOB
        if self.clob_data and self.clob_data['asks']:
            clob_prices = [ask[0] for ask in self.clob_data['asks']]
            clob_quantities = [ask[1] for ask in self.clob_data['asks']]
            
            # Calculate effective price for our trade amount
            clob_effective_price, clob_max_qty = self.calculateCLOBEffectivePrice(
                clob_prices, clob_quantities, self.trade_amount, is_buy=True
            )
            analysis['clob_effective_price'] = clob_effective_price
            analysis['clob_max_qty'] = clob_max_qty
            analysis['clob_slippage'] = self.calculateSlippage(
                clob_prices[0], clob_effective_price
            )
        
        # Analyze CFMM
        if self.cfmm_data:
            cfmm_ask_price = self.cfmm_data['asks'][0][0] if self.cfmm_data['asks'] else float('inf')
            cfmm_depth = self.cfmm_data['asks'][0][1] if self.cfmm_data['asks'] else 0
            
            # Calculate CFMM effective price considering slippage
            cfmm_effective_price, cfmm_max_qty = self.calculateCFMMEffectivePrice(
                self.trade_amount, is_buy=True
            )
            analysis['cfmm_effective_price'] = cfmm_effective_price
            analysis['cfmm_max_qty'] = cfmm_max_qty
            analysis['cfmm_slippage'] = self.calculateSlippage(
                cfmm_ask_price, cfmm_effective_price
            )
        
        # Determine best venue
        analysis.update(self.determineBestVenue(analysis, is_buy=True))
        
        return analysis

    def analyzeSellOpportunities(self):
        """Analyze opportunities for selling asset X"""
        analysis = {}
        
        # Analyze CLOB
        if self.clob_data and self.clob_data['bids']:
            clob_prices = [bid[0] for bid in self.clob_data['bids']]
            clob_quantities = [bid[1] for bid in self.clob_data['bids']]
            
            # Calculate effective price for our trade quantity
            # For sells, we need to estimate how much we can sell for our target amount
            estimated_sell_qty = self.trade_amount / clob_prices[0] if clob_prices[0] > 0 else 0
            
            clob_effective_price, clob_max_qty = self.calculateCLOBEffectivePrice(
                clob_prices, clob_quantities, estimated_sell_qty, is_buy=False
            )
            analysis['clob_effective_price'] = clob_effective_price
            analysis['clob_max_qty'] = clob_max_qty
            analysis['clob_slippage'] = self.calculateSlippage(
                clob_prices[0], clob_effective_price
            )
        
        # Analyze CFMM
        if self.cfmm_data:
            cfmm_bid_price = self.cfmm_data['bids'][0][0] if self.cfmm_data['bids'] else 0
            cfmm_depth = self.cfmm_data['bids'][0][1] if self.cfmm_data['bids'] else 0
            
            # Calculate CFMM effective price for selling
            cfmm_effective_price, cfmm_max_qty = self.calculateCFMMEffectivePrice(
                self.trade_amount, is_buy=False
            )
            analysis['cfmm_effective_price'] = cfmm_effective_price
            analysis['cfmm_max_qty'] = cfmm_max_qty
            analysis['cfmm_slippage'] = self.calculateSlippage(
                cfmm_bid_price, cfmm_effective_price
            )
        
        # Determine best venue
        analysis.update(self.determineBestVenue(analysis, is_buy=False))
        
        return analysis

    def calculateCLOBEffectivePrice(self, prices, quantities, target_amount, is_buy):
        """Calculate effective price for trading target_amount on CLOB"""
        if not prices or target_amount <= 0:
            return float('inf') if is_buy else 0, 0
        
        total_cost = 0
        remaining_amount = target_amount
        total_quantity = 0
        
        for i, (price, qty) in enumerate(zip(prices, quantities)):
            if remaining_amount <= 0:
                break
                
            if is_buy:
                # Buying: amount is in quote currency
                available_value = qty * price
                if available_value <= remaining_amount:
                    # Take entire level
                    total_cost += available_value
                    total_quantity += qty
                    remaining_amount -= available_value
                else:
                    # Partial fill at this level
                    fill_qty = remaining_amount / price
                    total_cost += remaining_amount
                    total_quantity += fill_qty
                    remaining_amount = 0
            else:
                # Selling: amount is in base currency
                if qty <= remaining_amount:
                    # Take entire level
                    total_cost += qty * price
                    total_quantity += qty
                    remaining_amount -= qty
                else:
                    # Partial fill at this level
                    total_cost += remaining_amount * price
                    total_quantity += remaining_amount
                    remaining_amount = 0
        
        if total_quantity > 0:
            effective_price = total_cost / total_quantity if is_buy else total_cost / target_amount
            max_qty = total_quantity
        else:
            effective_price = prices[0] if prices else float('inf')
            max_qty = 0
            
        return effective_price, max_qty

    def calculateCFMMEffectivePrice(self, trade_amount, is_buy):
        """Calculate effective price for trading on CFMM"""
        if not self.cfmm_data:
            return float('inf') if is_buy else 0, 0
        
        # Query CFMM for precise calculation
        # This would require a more sophisticated CFMM interface
        # For now, use simplified calculation based on pool reserves
        
        x_reserve, y_reserve = self.cfmm_data['pool_reserves']
        if x_reserve == 0 or y_reserve == 0:
            return float('inf') if is_buy else 0, 0
        
        pool_price = y_reserve / x_reserve
        
        if is_buy:
            # Buying X with Y: price impact increases with trade size
            # Simplified: effective price = pool_price * (1 + slippage_estimate)
            slippage_estimate = min(0.1, trade_amount / (y_reserve * 10))  # Conservative estimate
            effective_price = pool_price * (1 + slippage_estimate)
            max_qty = x_reserve * 0.1  # Don't trade more than 10% of pool
        else:
            # Selling X for Y
            slippage_estimate = min(0.1, (trade_amount / pool_price) / (x_reserve * 10))
            effective_price = pool_price * (1 - slippage_estimate)
            max_qty = x_reserve * 0.1
            
        return effective_price, max_qty

    def calculateSlippage(self, best_price, effective_price):
        """Calculate slippage percentage"""
        if best_price == 0 or effective_price == 0:
            return float('inf')
        return abs(effective_price - best_price) / best_price

    def determineBestVenue(self, analysis, is_buy):
        """Determine the best venue for execution"""
        result = {
            'best_venue': None,
            'split_trade': False,
            'split_ratios': None
        }
        
        clob_price = analysis.get('clob_effective_price')
        cfmm_price = analysis.get('cfmm_effective_price')
        
        if clob_price is None and cfmm_price is None:
            return result
            
        if clob_price is None:
            result['best_venue'] = 'CFMM'
            return result
            
        if cfmm_price is None:
            result['best_venue'] = 'CLOB'
            return result
        
        # Compare prices (considering slippage constraints)
        if is_buy:
            best_price = min(clob_price, cfmm_price)
            if best_price > 0 and best_price != float('inf'):
                if clob_price == best_price and analysis['clob_slippage'] <= self.max_slippage:
                    result['best_venue'] = 'CLOB'
                elif cfmm_price == best_price and analysis['cfmm_slippage'] <= self.max_slippage:
                    result['best_venue'] = 'CFMM'
        else:
            best_price = max(clob_price, cfmm_price)
            if best_price > 0 and best_price != float('inf'):
                if clob_price == best_price and analysis['clob_slippage'] <= self.max_slippage:
                    result['best_venue'] = 'CLOB'
                elif cfmm_price == best_price and analysis['cfmm_slippage'] <= self.max_slippage:
                    result['best_venue'] = 'CFMM'
        
        # Consider split trading if both venues are good
        if (result['best_venue'] and 
            analysis['clob_slippage'] <= self.max_slippage and 
            analysis['cfmm_slippage'] <= self.max_slippage and
            abs(clob_price - cfmm_price) / min(clob_price, cfmm_price) < 0.01):  # Within 1%
            
            result['split_trade'] = True
            result['split_ratios'] = self.calculateOptimalSplit(analysis, is_buy)
            
        return result

    def calculateOptimalSplit(self, analysis, is_buy):
        """Calculate optimal split between CLOB and CFMM"""
        # Simple proportional split based on depth
        clob_depth_value = analysis.get('clob_max_qty', 0) * (
            analysis['clob_effective_price'] if is_buy else 1/analysis['clob_effective_price']
        )
        cfmm_depth_value = analysis.get('cfmm_max_qty', 0) * (
            analysis['cfmm_effective_price'] if is_buy else 1/analysis['cfmm_effective_price']
        )
        
        total_depth = clob_depth_value + cfmm_depth_value
        if total_depth == 0:
            return {'CLOB': 0.5, 'CFMM': 0.5}
            
        clob_ratio = clob_depth_value / total_depth
        cfmm_ratio = cfmm_depth_value / total_depth
        
        return {'CLOB': clob_ratio, 'CFMM': cfmm_ratio}

    def executeTrade(self, currentTime, is_buy_order, venue_analysis):
        """Execute trade based on venue analysis"""
        self.state = 'EXECUTING_TRADE'
        self.current_strategy = venue_analysis
        
        if venue_analysis['split_trade']:
            self.executeSplitTrade(currentTime, is_buy_order, venue_analysis)
        else:
            self.executeSingleVenueTrade(currentTime, is_buy_order, venue_analysis)

    def executeSingleVenueTrade(self, currentTime, is_buy_order, venue_analysis):
        """Execute trade on a single venue"""
        venue = venue_analysis['best_venue']
        
        if venue == 'CLOB':
            self.executeCLOBTrade(currentTime, is_buy_order)
        elif venue == 'CFMM':
            self.executeCFMMTrade(currentTime, is_buy_order)

    def executeSplitTrade(self, currentTime, is_buy_order, venue_analysis):
        """Execute split trade across both venues"""
        ratios = venue_analysis['split_ratios']
        clob_amount = self.trade_amount * ratios['CLOB']
        cfmm_amount = self.trade_amount * ratios['CFMM']
        
        log_print(f"MarketOnlyAgent {self.id}: Splitting trade - CLOB: {clob_amount:.0f}, CFMM: {cfmm_amount:.0f}")
        
        # Execute both trades (simplified - in reality would need to handle partial fills)
        if clob_amount >= self.min_trade_size:
            self.executeCLOBTrade(currentTime, is_buy_order, clob_amount)
            
        if cfmm_amount >= self.min_trade_size:
            self.executeCFMMTrade(currentTime, is_buy_order, cfmm_amount)

    def executeCLOBTrade(self, currentTime, is_buy_order, amount=None):
        """Execute trade on CLOB"""
        if amount is None:
            amount = self.trade_amount
            
        if is_buy_order:
            # For CLOB buy, we need to convert amount to quantity
            best_ask = self.clob_data['asks'][0][0] if self.clob_data['asks'] else 0
            if best_ask > 0:
                quantity = int(amount / best_ask)
                if quantity > 0:
                    # Use market order (limit order at slightly above best ask)
                    limit_price = best_ask * 1.01  # 1% above to ensure execution
                    self.placeLimitOrder(self.symbol, quantity, True, limit_price)
                    log_print(f"MarketOnlyAgent {self.id}: CLOB BUY order placed - {quantity} @ {limit_price:.4f}")
        else:
            # For CLOB sell, amount is the value we want to get
            best_bid = self.clob_data['bids'][0][0] if self.clob_data['bids'] else 0
            if best_bid > 0:
                quantity = int(amount / best_bid)
                if quantity > 0:
                    limit_price = best_bid * 0.99  # 1% below to ensure execution
                    self.placeLimitOrder(self.symbol, quantity, False, limit_price)
                    log_print(f"MarketOnlyAgent {self.id}: CLOB SELL order placed - {quantity} @ {limit_price:.4f}")

    def executeCFMMTrade(self, currentTime, is_buy_order, amount=None):
        """Execute trade on CFMM"""
        if amount is None:
            amount = self.trade_amount
            
        if self.cfmm_exchange_id:
            self.sendMessage(self.cfmm_exchange_id, Message({
                "msg": "CFMM_TRADE_REQUEST",
                "sender": self.id,
                "symbol": self.symbol,
                "quantity": amount,
                "is_buy_order": is_buy_order,
                "max_slippage": self.max_slippage
            }))
            log_print(f"MarketOnlyAgent {self.id}: CFMM trade request sent - {amount} for {'BUY' if is_buy_order else 'SELL'}")

    def handleTradeExecution(self, currentTime, msg):
        """Handle successful trade execution"""
        if msg.body['msg'] == 'ORDER_EXECUTED':
            venue = 'CLOB'
            order = msg.body['order']
            quantity = order.quantity
            price = order.fill_price
        else:  # CFMM_TRADE_EXECUTED
            venue = 'CFMM'
            quantity = msg.body['quantity']
            price = msg.body['price']
            
        # Record trade
        self.trade_history.append({
            'timestamp': currentTime,
            'venue': venue,
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'direction': 'BUY' if ('is_buy_order' in msg.body and msg.body['is_buy_order']) else 'SELL'
        })
        
        self.venue_preference[venue] += 1
        
        log_print(f"MarketOnlyAgent {self.id}: Trade executed on {venue} - {quantity} @ {price:.4f}")
        
        self.completeTradingCycle(currentTime)

    def handleTradeRejection(self, currentTime, msg):
        """Handle trade rejection"""
        venue = 'CLOB' if msg.body['msg'] == 'ORDER_REJECTED' else 'CFMM'
        reason = msg.body.get('reason', 'unknown')
        
        log_print(f"MarketOnlyAgent {self.id}: Trade rejected by {venue} - Reason: {reason}")
        
        # Fallback to other venue if available
        if venue == 'CLOB' and self.cfmm_exchange_id:
            log_print(f"MarketOnlyAgent {self.id}: Falling back to CFMM")
            self.executeCFMMTrade(currentTime, True)  # Assuming buy order for fallback
        elif venue == 'CFMM' and self.clob_exchange_id:
            log_print(f"MarketOnlyAgent {self.id}: Falling back to CLOB")
            self.executeCLOBTrade(currentTime, True)  # Assuming buy order for fallback
        else:
            self.completeTradingCycle(currentTime)

    def completeTradingCycle(self, currentTime):
        """Complete the trading cycle and schedule next wakeup"""
        self.state = 'AWAITING_WAKEUP'
        self.setWakeup(currentTime + self.getWakeFrequency())
        
        # Log trading statistics
        total_trades = len(self.trade_history)
        if total_trades > 0:
            clob_trades = sum(1 for t in self.trade_history if t['venue'] == 'CLOB')
            cfmm_trades = sum(1 for t in self.trade_history if t['venue'] == 'CFMM')
            log_print(f"MarketOnlyAgent {self.id}: Statistics - Total: {total_trades}, CLOB: {clob_trades}, CFMM: {cfmm_trades}")

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def getTradingStatistics(self):
        """Get comprehensive trading statistics"""
        if not self.trade_history:
            return {}
            
        df = pd.DataFrame(self.trade_history)
        stats = {
            'total_trades': len(df),
            'clob_trades': len(df[df['venue'] == 'CLOB']),
            'cfmm_trades': len(df[df['venue'] == 'CFMM']),
            'avg_trade_size': df['quantity'].mean(),
            'avg_trade_price': df['price'].mean(),
            'venue_preference': self.venue_preference
        }
        
        return stats