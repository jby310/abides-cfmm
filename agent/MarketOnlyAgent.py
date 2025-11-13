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
        """Analyze both markets and execute optimal trade as per Document 3"""
        if not self.clob_data and not self.cfmm_data:
            log_print(f"MarketOnlyAgent {self.id}: No market data available")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())
            return
            
        # Use fixed trade direction as per Document 3 (not random)
        is_buy_order = self.trade_direction
        action = "BUY" if is_buy_order else "SELL"
        
        log_print(f"MarketOnlyAgent {self.id}: Planning to {action} {self.trade_amount} worth of {self.symbol}")
        
        # Compare venues and execute trade following Document 3 logic
        venue_analysis = self.compareVenuesDocument3(is_buy_order)
        
        if venue_analysis['best_venue'] or venue_analysis['split_trade']:
            self.executeTradeDocument3(currentTime, is_buy_order, venue_analysis)
        else:
            log_print(f"MarketOnlyAgent {self.id}: No suitable venue found within slippage limits")
            self.state = 'AWAITING_WAKEUP'
            self.setWakeup(currentTime + self.getWakeFrequency())

    def compareVenuesDocument3(self, is_buy_order):
        """
        Compare CLOB and CFMM following Document 3 specifications
        Implements the price comparison logic from Document 3
        """
        analysis = {
            'best_venue': None,
            'split_trade': False,
            'clob_best_price': None,
            'cfmm_best_price': None,
            'slippage_price': None
        }
        
        # Get best prices from both venues
        if self.clob_data:
            if is_buy_order and self.clob_data['asks']:
                analysis['clob_best_price'] = self.clob_data['asks'][0][0]
            elif not is_buy_order and self.clob_data['bids']:
                analysis['clob_best_price'] = self.clob_data['bids'][0][0]
        
        if self.cfmm_data:
            x_reserve, y_reserve = self.cfmm_data['pool_reserves']
            if x_reserve > 0 and y_reserve > 0:
                pool_price = y_reserve / x_reserve
                phi = 1 - self.cfmm_fee  # phi = 1 - fee_rate
                
                if is_buy_order:
                    # CFMM ask price = pool_price / phi
                    analysis['cfmm_best_price'] = pool_price / phi
                else:
                    # CFMM bid price = pool_price * phi
                    analysis['cfmm_best_price'] = pool_price * phi
        
        # Calculate slippage price as per Document 3
        if is_buy_order:
            best_ask = float('inf')
            if analysis['clob_best_price'] is not None:
                best_ask = min(best_ask, analysis['clob_best_price'])
            if analysis['cfmm_best_price'] is not None:
                best_ask = min(best_ask, analysis['cfmm_best_price'])
            
            if best_ask != float('inf'):
                analysis['slippage_price'] = best_ask * (1 + self.max_slippage)
                analysis['best_venue'] = 'CLOB' if analysis['clob_best_price'] == best_ask else 'CFMM'
        else:
            best_bid = 0
            if analysis['clob_best_price'] is not None:
                best_bid = max(best_bid, analysis['clob_best_price'])
            if analysis['cfmm_best_price'] is not None:
                best_bid = max(best_bid, analysis['cfmm_best_price'])
            
            if best_bid > 0:
                analysis['slippage_price'] = best_bid * (1 - self.max_slippage)
                analysis['best_venue'] = 'CLOB' if analysis['clob_best_price'] == best_bid else 'CFMM'
        
        # Check if split trading is beneficial (both venues within slippage limits)
        if (analysis['clob_best_price'] is not None and 
            analysis['cfmm_best_price'] is not None):
            
            if is_buy_order:
                clob_acceptable = analysis['clob_best_price'] <= analysis['slippage_price']
                cfmm_acceptable = analysis['cfmm_best_price'] <= analysis['slippage_price']
            else:
                clob_acceptable = analysis['clob_best_price'] >= analysis['slippage_price']
                cfmm_acceptable = analysis['cfmm_best_price'] >= analysis['slippage_price']
            
            if clob_acceptable and cfmm_acceptable:
                analysis['split_trade'] = True
                # Simple 50-50 split, can be optimized based on depth
                analysis['split_ratios'] = {'CLOB': 0.5, 'CFMM': 0.5}
        
        return analysis

    def executeTradeDocument3(self, currentTime, is_buy_order, venue_analysis):
        """Execute trade following Document 3 specifications"""
        self.state = 'EXECUTING_TRADE'
        self.current_strategy = venue_analysis
        
        if venue_analysis['split_trade']:
            self.executeSplitTradeDocument3(currentTime, is_buy_order, venue_analysis)
        else:
            self.executeSingleVenueTradeDocument3(currentTime, is_buy_order, venue_analysis)

    def executeSingleVenueTradeDocument3(self, currentTime, is_buy_order, venue_analysis):
        """Execute trade on a single venue following Document 3"""
        venue = venue_analysis['best_venue']
        
        if venue == 'CLOB':
            self.executeCLOBTradeDocument3(currentTime, is_buy_order, self.trade_amount)
        elif venue == 'CFMM':
            self.executeCFMMTradeDocument3(currentTime, is_buy_order, self.trade_amount)

    def executeSplitTradeDocument3(self, currentTime, is_buy_order, venue_analysis):
        """Execute split trade across both venues following Document 3"""
        ratios = venue_analysis['split_ratios']
        clob_amount = self.trade_amount * ratios['CLOB']
        cfmm_amount = self.trade_amount * ratios['CFMM']
        
        log_print(f"MarketOnlyAgent {self.id}: Splitting trade - CLOB: {clob_amount:.0f}, CFMM: {cfmm_amount:.0f}")
        
        # Execute both trades
        if clob_amount >= self.min_trade_size:
            self.executeCLOBTradeDocument3(currentTime, is_buy_order, clob_amount)
            
        if cfmm_amount >= self.min_trade_size:
            self.executeCFMMTradeDocument3(currentTime, is_buy_order, cfmm_amount)

    def executeCLOBTradeDocument3(self, currentTime, is_buy_order, amount):
        """Execute trade on CLOB following Document 3"""
        if is_buy_order:
            if self.clob_data and self.clob_data['asks']:
                best_ask = self.clob_data['asks'][0][0]
                if best_ask > 0:
                    quantity = int(amount / best_ask)
                    if quantity > 0:
                        # Use limit order at best ask to ensure execution
                        self.placeLimitOrder(self.symbol, quantity, True, best_ask)
                        log_print(f"MarketOnlyAgent {self.id}: CLOB BUY order placed - {quantity} @ {best_ask:.4f}")
        else:
            if self.clob_data and self.clob_data['bids']:
                best_bid = self.clob_data['bids'][0][0]
                if best_bid > 0:
                    quantity = int(amount / best_bid)
                    if quantity > 0:
                        # Use limit order at best bid to ensure execution
                        self.placeLimitOrder(self.symbol, quantity, False, best_bid)
                        log_print(f"MarketOnlyAgent {self.id}: CLOB SELL order placed - {quantity} @ {best_bid:.4f}")

    def executeCFMMTradeDocument3(self, currentTime, is_buy_order, amount):
        """Execute trade on CFMM following Document 3"""
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