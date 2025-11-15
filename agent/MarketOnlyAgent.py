from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import log_print
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from agent.CFMMAgent import CFMMAgent
from util.order.LimitOrder import LimitOrder

class MarketOnlyAgent(TradingAgent):
    """
    Market Only Agent with modified logic:
    - is_buy_order=True: use Y to buy X (spend Y, get X)
    - is_buy_order=False: sell X to get Y (spend X, get Y)
    """
    
    def __init__(self, id, name, type, symbol, starting_cash=100000, 
                 max_slippage=0.05, wake_up_freq='60s', min_trade_size=10,
                 log_orders=False, random_state=None, hybrid=False):
        
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state)
        
        self.symbol = symbol
        self.max_slippage = max_slippage
        self.wake_up_freq = wake_up_freq
        self.min_trade_size = min_trade_size
        
        # Asset holdings (X and Y) - initial: all in Y (cash), no X
        self.x_holdings = 0  # Base asset (e.g., ETH)
        self.y_holdings = starting_cash  # Quote asset (e.g., USDT)
        self.holdings[self.symbol] = 0
        
        # Venue identifiers
        self.clob_exchange_id = None
        self.cfmm_exchange_id = None
        
        # State tracking
        self.state = 'AWAITING_WAKEUP'
        self.pending_queries = 0
        
        # Price data from both venues
        self.clob_data = None
        self.cfmm_data = None
        self.cfmm_fee = None
        
        # CLOB order book management
        self.remaining_clob_bids = []
        self.remaining_clob_asks = []

        # 判断CFMM资产池是否需要重置
        self.reset_buy = False
        self.reset_sell = False

        self.hybrid = hybrid

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
        
        # Reset price data
        self.clob_data = None
        self.cfmm_data = None
        self.cfmm_fee = None
        
        # Reset CLOB order book state
        self.remaining_clob_bids = []
        self.remaining_clob_asks = []
        
        self.setWakeup(currentTime + self.getWakeFrequency())
        
        # Query CLOB markets
        if self.clob_exchange_id is not None:
            self.getCurrentSpread(self.symbol, depth=10)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        if self.state == 'QUERYING_MARKETS':
            if msg.body['msg'] == 'QUERY_SPREAD':
                self.handleCLOBData(currentTime, msg)
                self.analyzeMarketsAndTrade(currentTime)
                
        elif self.state == 'EXECUTING_TRADE':
            if msg.body['msg'] in ['ORDER_EXECUTED', 'CFMM_TRADE_EXECUTED']:
                self.handleTradeExecution(currentTime, msg)
            elif msg.body['msg'] in ['ORDER_REJECTED', 'CFMM_TRADE_REJECTED']:
                self.handleTradeRejection(currentTime, msg)

    def analyzeMarketsAndTrade(self, currentTime):
        """Analyze both markets and execute optimal trade"""
        is_buy_order = self.random_state.choice([True, False])
        action = "BUY" if is_buy_order else "SELL"
        
        # Calculate actual trade amount based on current holdings and modified logic
        if is_buy_order:
            # For buy orders: use Y to buy X - amount is limited by Y holdings
            max_trade_amount = self.y_holdings
            log_print(f"MarketOnlyAgent {self.id}: Planning to {action} X using up to {max_trade_amount:.2f} Y (Y holdings: {self.y_holdings:.2f})")
        else:
            # For sell orders: sell X to get Y - amount is limited by X holdings
            max_trade_amount = self.x_holdings
            log_print(f"MarketOnlyAgent {self.id}: Planning to {action} X to get Y, quantity: {max_trade_amount:.2f} X (X holdings: {self.x_holdings:.2f})")
        
        # Check if we have enough holdings to trade
        if max_trade_amount < self.min_trade_size:
            log_print(f"MarketOnlyAgent {self.id}: Insufficient holdings for minimum trade size")
            return
        
        # Execute trade following the detailed flow chart logic
        # 在每个时刻交易之前，判断是否需要重置CFMM池子。
        # need_reset = CFMMAgent.check_cfmm_reset_needed(self.symbol)
        need_reset = self.reset_buy and self.reset_sell
        if need_reset:
            CFMMAgent.reset_cfmm_pool(self.symbol)
            log_print(f"MarketOnlyAgent {self.id}: CFMM pool reset for {self.symbol}")

        self.executeTradeFlowChart(currentTime, is_buy_order, max_trade_amount)

    def handleCLOBData(self, currentTime, msg):
        """Process CLOB market data and initialize remaining order book"""
        self.clob_data = {
            'bids': msg.body['bids'],
            'asks': msg.body['asks'],
            'timestamp': currentTime,
            'last_trade': msg.body['data']
        }
        
        # Initialize remaining order book with full depth
        self.remaining_clob_bids = deepcopy(self.clob_data['bids'])
        self.remaining_clob_asks = deepcopy(self.clob_data['asks'])
        
        log_print(f"MarketOnlyAgent {self.id}: CLOB data received - "
                 f"Best Bid: {self.clob_data['bids'][0] if self.clob_data['bids'] else 'N/A'}, "
                 f"Best Ask: {self.clob_data['asks'][0] if self.clob_data['asks'] else 'N/A'}")

    def executeTradeFlowChart(self, currentTime, is_buy_order, max_trade_amount):
        """Execute trade following the detailed flow chart from experimental design"""
        self.state = 'EXECUTING_TRADE'
        remaining_amount = max_trade_amount
        
        # Get initial best prices and calculate slippage limit
        clob_price, cfmm_price = self.getCurrentBestPrices(is_buy_order)
        if clob_price is None and cfmm_price is None:
            log_print(f"MarketOnlyAgent {self.id}: No best prices available for trade")
            return 
        best_price = min([p for p in [clob_price, cfmm_price] if p is not None]) if is_buy_order else max([p for p in [clob_price, cfmm_price] if p is not None])
        max_slippage_price = best_price * (1 + self.max_slippage) if is_buy_order else best_price * (1 - self.max_slippage)
        
        if self.hybrid:
            if is_buy_order and cfmm_price > max_slippage_price:
                self.reset_buy = True 
            elif not is_buy_order and cfmm_price < max_slippage_price:
                self.reset_sell = True

        log_print(f"MarketOnlyAgent {self.id}: Best price: {best_price:.4f}, Slippage limit: {max_slippage_price:.4f}")
        
        # Main trading loop
        while remaining_amount > self.min_trade_size:
            clob_price, cfmm_price = self.getCurrentBestPrices(is_buy_order)
            
            # Find available venues within slippage limits
            available_venues = self.getAvailableVenues(is_buy_order, clob_price, cfmm_price, max_slippage_price)
            
            if not available_venues:
                log_print(f"MarketOnlyAgent {self.id}: No venues available within slippage limits")
                break
            
            # Find the best venue
            best_venue, best_price = min(available_venues, key=lambda x: x[1]) if is_buy_order else max(available_venues, key=lambda x: x[1])
            
            # Calculate and execute trade
            trade_amount = self.calculateTradeAmount(best_venue, is_buy_order, best_price, remaining_amount, max_slippage_price)
            
            if trade_amount < self.min_trade_size or trade_amount > remaining_amount:
                log_print(f"MarketOnlyAgent {self.id}: Trade amount {trade_amount} below minimum, and over holdings, skipping")
                break
            
            # Execute trade on selected venue
            if best_venue == 'CLOB':
                executed_amount = self.executeCLOBTrade(currentTime, is_buy_order, trade_amount)
                if executed_amount:
                    remaining_amount -= executed_amount
                    log_print(f"MarketOnlyAgent {self.id}: Trade executed on {best_venue}, remaining: {remaining_amount:.2f}")
                else:
                    log_print(f"MarketOnlyAgent {self.id}: Trade failed on {best_venue}")
            else:  # CFMM
                executed = self.executeCFMMTrade(currentTime, is_buy_order, trade_amount)
                if executed:
                    remaining_amount -= trade_amount
                    log_print(f"MarketOnlyAgent {self.id}: Trade executed on {best_venue}, remaining: {remaining_amount:.2f}")
                else:
                    log_print(f"MarketOnlyAgent {self.id}: Trade failed on {best_venue}")
        
    def getCurrentBestPrices(self, is_buy_order):
        """Get current best prices from both venues"""
        clob_price = None
        cfmm_price = None
        
        # Get CLOB price from remaining order book
        if is_buy_order and self.remaining_clob_asks:
            clob_price = self.remaining_clob_asks[0][0]
        elif not is_buy_order and self.remaining_clob_bids:
            clob_price = self.remaining_clob_bids[0][0]
        
        # Get CFMM price using static interface
        if CFMMAgent.get_cfmm_instance(self.symbol) is not None:
            bid_price, ask_price = CFMMAgent.get_cfmm_bid_ask_prices(self.symbol)
            if bid_price is not None and ask_price is not None:
                cfmm_price = ask_price if is_buy_order else bid_price
        
        return clob_price, cfmm_price

    def getAvailableVenues(self, is_buy_order, clob_price, cfmm_price, max_slippage_price):
        """Get available venues within slippage limits"""
        available_venues = []
        
        if clob_price is not None:
            if (is_buy_order and clob_price <= max_slippage_price) or (not is_buy_order and clob_price >= max_slippage_price):

                available_venues.append(('CLOB', clob_price))
        
        if cfmm_price is not None:
            if (is_buy_order and cfmm_price <= max_slippage_price) or (not is_buy_order and cfmm_price >= max_slippage_price):
                available_venues.append(('CFMM', cfmm_price))
        
        return available_venues

    def calculateTradeAmount(self, venue, is_buy_order, price, remaining_amount, max_slippage_price):
        """Calculate how much to trade on a given venue"""
        if venue == 'CLOB':
            return self.calculateCLOBAmount(is_buy_order)
        else:  # CFMM
            return self.calculateCFMMAmount(is_buy_order, remaining_amount, max_slippage_price)

    def calculateCLOBAmount(self, is_buy_order):
        """Calculate tradable amount on CLOB"""
        order_book = self.remaining_clob_asks if is_buy_order else self.remaining_clob_bids
        return order_book[0][1] * order_book[0][0] if order_book else 0

    def calculateCFMMAmount(self, is_buy_order, remaining_amount, max_slippage_price):
        """Calculate CFMM tradable amount following modified logic"""
        if CFMMAgent.get_cfmm_instance(self.symbol) is None or not self.clob_data:
            return 0
            
        x_reserve, y_reserve = CFMMAgent.get_cfmm_reserves(self.symbol)
        self.cfmm_fee = CFMMAgent.get_cfmm_fee_rate(self.symbol)
        k = x_reserve * y_reserve
        phi = 1 - self.cfmm_fee
        
        if is_buy_order:
            # Buying X with Y: amount is in Y
            P_aL = self.remaining_clob_asks[0][0] if self.remaining_clob_asks else float('inf')
            boundary_price = min(P_aL, max_slippage_price) if P_aL != float('inf') else max_slippage_price
            delta_x = x_reserve - (1/phi) * (y_reserve / boundary_price)
            delta_y = (1/phi) * (k / (x_reserve - delta_x) - y_reserve)

            if delta_y >= remaining_amount:
                return remaining_amount
            else:
                return delta_y
        else:
            # Selling X for Y: amount is in X
            P_bL = self.remaining_clob_bids[0][0] if self.remaining_clob_bids else 0
            boundary_price = max(P_bL, max_slippage_price)
            
            delta_x = (y_reserve / boundary_price) - (x_reserve / phi)
            if delta_x >= remaining_amount:
                return remaining_amount
            else:
                return delta_x
                
    def placeLimitOrder (self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk = True, tag = None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)

        if quantity > 0:
            # Test if this order can be permitted given our at-risk limits.
            new_holdings = self.holdings.copy()

            q = order.quantity if order.is_buy_order else -order.quantity

            if order.symbol in new_holdings: new_holdings[order.symbol] += q
            else: new_holdings[order.symbol] = q

            # If at_risk is lower, always allow.  Otherwise, new_at_risk must be below starting cash.
            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.markToMarket(self.holdings) - self.holdings['CASH']
                new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']

                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    log_print ("TradingAgent ignored limit order due to at-risk constraints: {}\n{}", order, self.fmtHoldings(self.holdings))
                    return

            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.  Eventually
            # it might be nice to make the whole history of the order into transaction
            # objects inside the order (we're halfway there) so there CAN be just a single
            # object per order, that never alters its original state, and eliminate all these copies.
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID, Message({ "msg" : "LIMIT_ORDER", "sender": self.id,
                                                        "order" : order })) 

            # Log this activity.
            if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

        else:
            log_print ("TradingAgent ignored limit order of quantity zero: {}", order)

    def executeCLOBTrade(self, currentTime, is_buy_order, amount):
        """Execute CLOB trade with modified logic - only process first level at a time"""
        order_book = self.remaining_clob_asks if is_buy_order else self.remaining_clob_bids
        if not order_book:
            return 0
        
        # Only process the first level
        price, quantity = order_book[0]
        
        if quantity <= amount:
            # Can buy the entire first level
            self.placeLimitOrder(self.symbol, quantity, is_buy_order, price)
            # Remove the first level since it's fully executed
            del order_book[0]
            
            action = "BUY" if is_buy_order else "SELL"
            log_print(f"{currentTime}: MarketOnlyAgent {self.id}: CLOB {action} executed - {quantity} shares {'Y spent' if is_buy_order else 'X sold'}")
            return quantity
        else:
            self.placeLimitOrder(self.symbol, amount, is_buy_order, price)
            del order_book[0]

            action = "BUY" if is_buy_order else "SELL"
            log_print(f"{currentTime}: MarketOnlyAgent {self.id}: CLOB {action} executed - {quantity} shares {'Y spent' if is_buy_order else 'X sold'}")
            return amount

    def executeCFMMTrade(self, currentTime, is_buy_order, amount):
        """Execute CFMM trade using static method with modified logic"""
        # Use static method to execute CFMM trade directly
        executed_qty = CFMMAgent.execute_cfmm_trade_static(
            symbol=self.symbol,
            agent_id=self.id,
            quantity=amount,
            is_buy_order=is_buy_order,
            current_time=currentTime
        )
        
        if executed_qty > 0:
            log_print(f"MarketOnlyAgent {self.id}: CFMM trade executed via static method - "
                    f"{executed_qty:.4f} {'X received' if is_buy_order else 'Y received'} ")
            
            # Update holdings immediately since we're not waiting for message response
            if is_buy_order:
                self.x_holdings += executed_qty  # Receive X
                self.y_holdings -= amount  # Spend Y
                self.holdings[self.symbol] += executed_qty
                self.holdings['CASH'] -= amount
            else:
                self.x_holdings -= amount  # Spend X  
                self.y_holdings += executed_qty  # Receive Y
                self.holdings[self.symbol] -= amount
                self.holdings['CASH'] += executed_qty
            
            log_print(f"MarketOnlyAgent {self.id}: Updated holdings - X: {self.x_holdings:.2f}, Y: {self.y_holdings:.2f}")
            
            return {
                'executed': True,
                'quantity': executed_qty,
            }
        else:
            log_print(f"MarketOnlyAgent {self.id}: CFMM trade failed via static method for {'BUY' if is_buy_order else 'SELL'}")
            return {
                'executed': False,
                'quantity': 0,
            }

    def handleTradeExecution(self, currentTime, msg):
        """Handle successful trade execution with modified logic"""
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            quantity = order.quantity
            price = order.fill_price
            is_buy_order = order.is_buy_order
            
            # Update holdings for CLOB trades
            if is_buy_order:
                self.x_holdings += quantity  # Receive X
                self.y_holdings -= quantity * price  # Spend Y
            else:
                self.x_holdings -= quantity  # Spend X
                self.y_holdings += quantity * price  # Receive Y
                
        log_print(f"MarketOnlyAgent {self.id}: Trade executed - {quantity} @ {price:.4f}")
        log_print(f"MarketOnlyAgent {self.id}: Updated holdings - X: {self.x_holdings:.2f}, Y: {self.y_holdings:.2f}")

    def handleTradeRejection(self, currentTime, msg):
        """Handle trade rejection"""
        venue = 'CLOB' if msg.body['msg'] == 'ORDER_REJECTED' else 'CFMM'
        reason = msg.body.get('reason', 'unknown')
        log_print(f"MarketOnlyAgent {self.id}: Trade rejected by {venue} - Reason: {reason}")

    # def getWakeFrequency(self):
    #     return pd.Timedelta(self.wake_up_freq)

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=30, high=100), unit='s')