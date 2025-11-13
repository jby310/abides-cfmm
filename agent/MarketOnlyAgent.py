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
    Strictly follows Document 3 specifications
    """
    
    def __init__(self, id, name, type, symbol, starting_cash=100000, 
                 max_slippage=0.05, wake_up_freq='60s', min_trade_size=100,
                 log_orders=False, random_state=None, trade_direction=None):
        
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state)
        
        self.symbol = symbol
        self.max_slippage = max_slippage  # Maximum acceptable slippage (e.g., 5%)
        self.wake_up_freq = wake_up_freq
        self.min_trade_size = min_trade_size  # Minimum trade size to avoid dust
        
        # Trade direction is fixed (not random) as per Document 3
        self.trade_direction = trade_direction  # True for buy, False for sell
        
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
        self.cfmm_fee = None  # Fee rate from CFMM
        
        # Trading statistics
        self.trade_history = []
        self.venue_preference = {'CLOB': 0, 'CFMM': 0}
        
        # Trade amount as per Document 3 - fixed amount M
        self.trade_amount = starting_cash  # Fixed amount M as per Document 3
        
    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        
        # Find both CLOB and CFMM exchanges
        from agent.ExchangeAgent import ExchangeAgent
        from agent.CFMMAgent import CFMMAgent
        
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
        self.cfmm_fee = None
        
        self.setWakeup(currentTime + self.getWakeFrequency())
        
        # Query both markets
        if self.clob_exchange_id is not None:
            self.getCurrentSpread(self.symbol, depth=5)
            self.pending_queries += 1
            
        if self.cfmm_exchange_id is not None:
            self.queryCFMMMarketData()
            self.pending_queries += 1
            
        if self.pending_queries == 0:
            log_print(f"MarketOnlyAgent {self.id}: No venues available")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())

    def queryCFMMMarketData(self):
        """Query CFMM for current market data including fee rate"""
        self.sendMessage(self.cfmm_exchange_id,  Message({
            "msg": "CFMM_MARKET_DATA_REQUEST",
            "sender": self.id,
            "symbol": self.symbol,
            "levels": 5
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
            'bids': msg.body['bids'],
            'asks': msg.body['asks'],
            'timestamp': currentTime,
            'last_trade': msg.body['data']
        }
        
        log_print(f"MarketOnlyAgent {self.id}: CLOB data received - "
                 f"Best Bid: {self.clob_data['bids'][0] if self.clob_data['bids'] else 'N/A'}, "
                 f"Best Ask: {self.clob_data['asks'][0] if self.clob_data['asks'] else 'N/A'}")

    def handleCFMMData(self, currentTime, msg):
        """Process CFMM market data including fee rate"""
        data = msg.body['data']
        self.cfmm_data = {
            'bids': data['bids'],
            'asks': data['asks'],
            'timestamp': currentTime,
            'pool_reserves': data.get('pool_reserves', (0, 0)),
            'last_trade': data.get('last_trade', 0),
            'fee_rate': data.get('fee_rate', 0.003)  # Get fee rate from CFMM
        }
        self.cfmm_fee = self.cfmm_data['fee_rate']
        
        log_print(f"MarketOnlyAgent {self.id}: CFMM data received - "
                 f"Pool Price: {self.cfmm_data['last_trade']:.4f}, "
                 f"Fee Rate: {self.cfmm_fee:.3f}, "
                 f"Reserves: {self.cfmm_data['pool_reserves']}")

    def checkAllDataReceived(self, currentTime):
        """Check if we have data from all venues and proceed to trading"""
        self.pending_queries -= 1
        
        if self.pending_queries > 0:
            return
            
        # Both queries completed, analyze and trade
        self.analyzeMarketsAndTrade(currentTime)

    def analyzeMarketsAndTrade(self, currentTime):
        """Analyze both markets and execute optimal trade following Document 3 flow chart"""
        if not self.clob_data and not self.cfmm_data:
            log_print(f"MarketOnlyAgent {self.id}: No market data available")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())
            return
            
        is_buy_order = self.random_state.choice([True, False])
        action = "BUY" if is_buy_order else "SELL"
        
        log_print(f"MarketOnlyAgent {self.id}: Planning to {action} {self.trade_amount} worth of {self.symbol}")
        
        # Execute trade following the detailed flow chart logic
        self.executeTradeFlowChart(currentTime, is_buy_order)

    def executeTradeFlowChart(self, currentTime, is_buy_order):
        """Execute trade following the detailed flow chart from experimental design"""
        self.state = 'EXECUTING_TRADE'
        
        remaining_amount = self.trade_amount
        max_slippage_price = None
        
        # Get initial best prices
        clob_price, cfmm_price = self.getCurrentBestPrices(is_buy_order)
        
        # Calculate slippage price based on best available price
        if is_buy_order:
            best_price = min([p for p in [clob_price, cfmm_price] if p is not None])
            max_slippage_price = best_price * (1 + self.max_slippage)
        else:
            best_price = max([p for p in [clob_price, cfmm_price] if p is not None])
            max_slippage_price = best_price * (1 - self.max_slippage)
        
        log_print(f"MarketOnlyAgent {self.id}: Best price: {best_price:.4f}, Slippage limit: {max_slippage_price:.4f}")
        
        # Main trading loop - continue until all amount is traded or no venues available
        while remaining_amount > self.min_trade_size:
            # Get current best prices (they may change after partial execution)
            clob_price, cfmm_price = self.getCurrentBestPrices(is_buy_order)
            
            # Check which venues are available and within slippage limits
            available_venues = []
            
            if clob_price is not None:
                if is_buy_order:
                    clob_acceptable = clob_price <= max_slippage_price
                else:
                    clob_acceptable = clob_price >= max_slippage_price
                if clob_acceptable:
                    available_venues.append(('CLOB', clob_price))
            
            if cfmm_price is not None:
                if is_buy_order:
                    cfmm_acceptable = cfmm_price <= max_slippage_price
                else:
                    cfmm_acceptable = cfmm_price >= max_slippage_price
                if cfmm_acceptable:
                    available_venues.append(('CFMM', cfmm_price))
            
            if not available_venues:
                log_print(f"MarketOnlyAgent {self.id}: No venues available within slippage limits")
                break
            
            # Find the best venue (lowest price for buy, highest for sell)
            if is_buy_order:
                best_venue, best_price = min(available_venues, key=lambda x: x[1])
            else:
                best_venue, best_price = max(available_venues, key=lambda x: x[1])
            
            # Calculate how much to trade on this venue
            trade_amount = self.calculateTradeAmount(best_venue, is_buy_order, best_price, remaining_amount)
            
            if trade_amount < self.min_trade_size:
                log_print(f"MarketOnlyAgent {self.id}: Trade amount {trade_amount} below minimum, skipping")
                break
            
            # Execute trade on the best venue
            if best_venue == 'CLOB':
                executed = self.executeCLOBTradeFlowChart(currentTime, is_buy_order, trade_amount, best_price)
            else:  # CFMM
                executed = self.executeCFMMTradeFlowChart(currentTime, is_buy_order, trade_amount)
            
            if executed:
                remaining_amount -= trade_amount
                log_print(f"MarketOnlyAgent {self.id}: Trade executed on {best_venue}, remaining: {remaining_amount:.2f}")
                
                # Update prices after trade (simulate market impact)
                # For CLOB, we assume the best level is consumed and need to refresh
                # For CFMM, the pool price changes automatically
                if best_venue == 'CLOB':
                    # Refresh CLOB data by querying again
                    self.getCurrentSpread(self.symbol, depth=5)
            else:
                log_print(f"MarketOnlyAgent {self.id}: Trade failed on {best_venue}")
                # Remove this venue from consideration for the next iteration
                if best_venue == 'CLOB':
                    self.clob_data = None
                else:
                    self.cfmm_data = None
        
        if remaining_amount < self.trade_amount:
            log_print(f"MarketOnlyAgent {self.id}: Trading completed. Total executed: {self.trade_amount - remaining_amount:.2f}, Remaining: {remaining_amount:.2f}")
        else:
            log_print(f"MarketOnlyAgent {self.id}: No trades executed")
        
        self.state = 'AWAITING_WAKEUP'
        self.setWakeup(currentTime + self.getWakeFrequency())

    def getCurrentBestPrices(self, is_buy_order):
        """Get current best prices from both venues"""
        clob_price = None
        cfmm_price = None
        
        # Get CLOB price
        if self.clob_data:
            if is_buy_order and self.clob_data['asks']:
                clob_price = self.clob_data['asks'][0][0]
            elif not is_buy_order and self.clob_data['bids']:
                clob_price = self.clob_data['bids'][0][0]
        
        # Get CFMM price
        if self.cfmm_data:
            x_reserve, y_reserve = self.cfmm_data['pool_reserves']
            if x_reserve > 0 and y_reserve > 0:
                pool_price = y_reserve / x_reserve
                phi = 1 - self.cfmm_fee
                
                if is_buy_order:
                    cfmm_price = pool_price / phi  # Ask price
                else:
                    cfmm_price = pool_price * phi  # Bid price
        
        return clob_price, cfmm_price

    def calculateTradeAmount(self, venue, is_buy_order, price, remaining_amount):
        """Calculate how much to trade on a given venue following proper market structure"""
        if venue == 'CLOB':
            # For CLOB, trade through order book levels until price exceeds slippage limit
            if is_buy_order:
                return self.calculateCLOBBuyAmount(remaining_amount)
            else:
                return self.calculateCLOBSellAmount(remaining_amount)
        
        else:  # CFMM
            # For CFMM, calculate maximum tradable amount based on pool reserves and slippage
            return self.calculateCFMMAmount(is_buy_order, remaining_amount)

    def calculateCLOBBuyAmount(self, remaining_amount):
        """Calculate how much to buy from CLOB order book within slippage limits"""
        if not self.clob_data or not self.clob_data['asks']:
            return 0
        
        total_tradable = 0
        current_remaining = remaining_amount
        slippage_limit = self.clob_data['asks'][0][0] * (1 + self.max_slippage)
        
        for price, quantity in self.clob_data['asks']:
            # Stop if price exceeds slippage limit
            if price > slippage_limit:
                break
                
            # Calculate how much we can buy at this price level
            level_value = price * quantity
            if level_value <= current_remaining:
                # Can buy entire level
                total_tradable += level_value
                current_remaining -= level_value
            else:
                # Can only buy part of this level
                tradable_at_level = current_remaining
                total_tradable += tradable_at_level
                current_remaining = 0
                break
                
            if current_remaining <= 0:
                break
        
        return total_tradable

    def calculateCLOBSellAmount(self, remaining_amount):
        """Calculate how much to sell to CLOB order book within slippage limits"""
        if not self.clob_data or not self.clob_data['bids']:
            return 0
        
        total_tradable = 0
        current_remaining = remaining_amount
        slippage_limit = self.clob_data['bids'][0][0] * (1 - self.max_slippage)
        
        for price, quantity in self.clob_data['bids']:
            # Stop if price falls below slippage limit
            if price < slippage_limit:
                break
                
            # Calculate how much we can sell at this price level
            level_value = price * quantity
            if level_value <= current_remaining:
                # Can sell entire level
                total_tradable += level_value
                current_remaining -= level_value
            else:
                # Can only sell part of this level
                tradable_at_level = current_remaining
                total_tradable += tradable_at_level
                current_remaining = 0
                break
                
            if current_remaining <= 0:
                break
        
        return total_tradable

    def executeCLOBTradeFlowChart(self, currentTime, is_buy_order, amount, best_price):
        """Execute CLOB trade by walking through the order book"""
        if is_buy_order:
            return self.executeCLOBBuyOrder(currentTime, amount)
        else:
            return self.executeCLOBSellOrder(currentTime, amount)

    def executeCLOBBuyOrder(self, currentTime, amount):
        """Execute CLOB buy order by walking through ask levels"""
        if not self.clob_data or not self.clob_data['asks']:
            return False
        
        remaining_amount = amount
        slippage_limit = self.clob_data['asks'][0][0] * (1 + self.max_slippage)
        total_quantity = 0
        
        for price, quantity in self.clob_data['asks']:
            if price > slippage_limit:
                break
                
            level_value = price * quantity
            if level_value <= remaining_amount:
                # Buy entire level
                self.placeLimitOrder(self.symbol, quantity, True, price)
                total_quantity += quantity
                remaining_amount -= level_value
            else:
                # Buy part of this level
                partial_quantity = int(remaining_amount / price)
                if partial_quantity > 0:
                    self.placeLimitOrder(self.symbol, partial_quantity, True, price)
                    total_quantity += partial_quantity
                    remaining_amount = 0
                break
                
            if remaining_amount <= 0:
                break
        
        if total_quantity > 0:
            log_print(f"MarketOnlyAgent {self.id}: CLOB BUY executed - {total_quantity} shares, {amount - remaining_amount:.2f} value")
            return True
        return False

    def executeCLOBSellOrder(self, currentTime, amount):
        """Execute CLOB sell order by walking through bid levels"""
        if not self.clob_data or not self.clob_data['bids']:
            return False
        
        remaining_amount = amount
        slippage_limit = self.clob_data['bids'][0][0] * (1 - self.max_slippage)
        total_quantity = 0
        
        for price, quantity in self.clob_data['bids']:
            if price < slippage_limit:
                break
                
            level_value = price * quantity
            if level_value <= remaining_amount:
                # Sell entire level
                self.placeLimitOrder(self.symbol, quantity, False, price)
                total_quantity += quantity
                remaining_amount -= level_value
            else:
                # Sell part of this level
                partial_quantity = int(remaining_amount / price)
                if partial_quantity > 0:
                    self.placeLimitOrder(self.symbol, partial_quantity, False, price)
                    total_quantity += partial_quantity
                    remaining_amount = 0
                break
                
            if remaining_amount <= 0:
                break
        
        if total_quantity > 0:
            log_print(f"MarketOnlyAgent {self.id}: CLOB SELL executed - {total_quantity} shares, {amount - remaining_amount:.2f} value")
            return True
        return False

    def calculateCFMMAmount(self, is_buy_order, remaining_amount):
        """Calculate CFMM tradable amount using Document 3 formulas"""
        if not self.cfmm_data:
            return 0
            
        x_reserve, y_reserve = self.cfmm_data['pool_reserves']
        k = x_reserve * y_reserve
        phi = 1 - self.cfmm_fee
        
        if is_buy_order:
            # Buying X with Y
            current_price = y_reserve / x_reserve if x_reserve > 0 else 0
            slippage_price = current_price * (1 + self.max_slippage)
            effective_price = min(current_price / phi, slippage_price)
            
            if effective_price <= 0:
                return 0
                
            # Solve for Δx: min(P_aL[n], slippage_price) = (1/φ) * (y / (x - Δx))
            # Δx = x - (1/φ) * (y / min(P_aL[n], slippage_price))
            delta_x = x_reserve - (1/phi) * (y_reserve / effective_price)
            delta_x = max(0, min(delta_x, x_reserve * 0.1))  # Limit to 10% of pool
            
            max_tradable_amount = delta_x * effective_price
            
        else:
            # Selling X for Y
            current_price = y_reserve / x_reserve if x_reserve > 0 else 0
            slippage_price = current_price * (1 - self.max_slippage)
            effective_price = max(current_price * phi, slippage_price)
            
            if effective_price <= 0:
                return 0
                
            # Solve for Δx: max(P_bL[n], slippage_price) = φ * (y / (x + φ * Δx))
            # Δx = (1/φ) * (φ * (y / max(P_bL[n], slippage_price)) - x)
            delta_x = (1/phi) * (phi * (y_reserve / effective_price) - x_reserve)
            delta_x = max(0, min(delta_x, x_reserve * 0.1))  # Limit to 10% of pool
            
            max_tradable_amount = delta_x * effective_price
        
        return min(remaining_amount, max_tradable_amount)

    def executeCFMMTradeFlowChart(self, currentTime, is_buy_order, amount):
        """Execute CFMM trade following flow chart logic"""
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
            return True
        return False

    
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
        
        # Simple fallback - try the other venue
        if venue == 'CLOB' and self.cfmm_exchange_id:
            log_print(f"MarketOnlyAgent {self.id}: Falling back to CFMM")
            self.executeCFMMTradeDocument3(currentTime, self.trade_direction, self.trade_amount)
        elif venue == 'CFMM' and self.clob_exchange_id:
            log_print(f"MarketOnlyAgent {self.id}: Falling back to CLOB")
            self.executeCLOBTradeDocument3(currentTime, self.trade_direction, self.trade_amount)
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