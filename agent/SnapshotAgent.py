import pandas as pd
from agent.TradingAgent import TradingAgent
from util.util import log_print
from agent.CFMMAgent import CFMMAgent

class SnapshotAgent(TradingAgent):
    def __init__(self, id, name, type, symbol, log_orders=False, random_state=None, hybrid=False):
        super().__init__(id, name, type, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.snapshot_interval = pd.Timedelta(1, unit='s')  # 每秒记录一次
        self.state = 'AWAITING_MARKET_OPEN'
        self.hybrid = hybrid

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        
        # 确保市场已开放且未关闭
        if not (self.mkt_open and self.mkt_close and not self.mkt_closed):
            return False
            
        # 初始状态：等待市场开放
        if self.state == 'AWAITING_MARKET_OPEN':
            self.state = 'ACTIVE'
            self.schedule_next_snapshot(currentTime)
            return True
            
        # 活跃状态：记录快照并安排下一次
        if self.state == 'ACTIVE':
            self.take_snapshot(currentTime)
            self.schedule_next_snapshot(currentTime)
            return True
            
        return False

    def schedule_next_snapshot(self, currentTime):
        # 计算下一次快照时间（当前时间加1秒）
        next_wake = currentTime + self.snapshot_interval
        # 确保不超过市场关闭时间
        if next_wake <= self.mkt_close:
            self.setWakeup(next_wake)
        else:
            self.state = 'INACTIVE'

    def take_snapshot(self, currentTime):
        # CLOB快照
        # 查询当前价差
        self.getCurrentSpread(self.symbol, depth=10)
        # 查询交易量
        self.get_transacted_volume(self.symbol, lookback_period='1s')

        
        # 记录当前时间用于后续处理响应
        self.last_snapshot_time = currentTime

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        # 处理价差查询响应
        if msg.body['msg'] == 'QUERY_SPREAD' and hasattr(self, 'last_snapshot_time'):
            symbol = msg.body['symbol']
            if symbol != self.symbol:
                return
                
            # 计算价差
            bids = msg.body['bids']
            asks = msg.body['asks']
            spread = None
            if bids and asks:
                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None
                # mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
                if best_bid and best_ask:
                    spread = (best_ask - best_bid) 
                    if self.hybrid:
                        # CFMM快照
                        # —— 新增 CFMM 指标记录 —— 
                        cfmm_market_data = CFMMAgent.get_cfmm_market_data(self.symbol)
                        bids_cfmm, asks_cfmm = cfmm_market_data['bids'], cfmm_market_data['asks']
                        if bids and asks:
                            bid_p_cfmm, bid_v = bids_cfmm[0]
                            ask_p_cfmm, ask_v = asks_cfmm[0]

                        self.logEvent('SPREAD', {
                            'timestamp': currentTime,
                            'spread': min(best_ask, ask_p_cfmm) - max(best_bid, bid_p_cfmm),
                        })
                    else:

                        self.logEvent('SPREAD', {
                            'timestamp': currentTime,
                            'spread': spread,
                        })
            
            # 计算±1%范围内的流动性
            if bids and asks and best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
                bid_threshold = mid_price * 0.99  # 1% below mid
                ask_threshold = mid_price * 1.01  # 1% above mid
                
                # 计算买单流动性（1%范围内）
                bid_liquidity = sum(vol for price, vol in bids if price >= bid_threshold)
                # 计算卖单流动性（1%范围内）
                ask_liquidity = sum(vol for price, vol in asks if price <= ask_threshold)
                
                depth = min(bid_liquidity, ask_liquidity)
                if self.hybrid:
                    self.logEvent('LIQUIDITY_1PCT', {
                        'timestamp': currentTime,
                        'mid_price': mid_price,
                        'bid_liquidity': bid_liquidity,
                        'ask_liquidity': ask_liquidity,
                        'bid_threshold': bid_threshold,
                        'ask_threshold': ask_threshold,
                        'depth': depth + cfmm_market_data['depth_1pct']
                    })
                
                else:
                    self.logEvent('LIQUIDITY_1PCT', {
                        'timestamp': currentTime,
                        'mid_price': mid_price,
                        'bid_liquidity': bid_liquidity,
                        'ask_liquidity': ask_liquidity,
                        'bid_threshold': bid_threshold,
                        'ask_threshold': ask_threshold,
                        'depth': depth
                    })
                
        # 处理交易量查询响应
        elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME' and hasattr(self, 'last_snapshot_time'):
            symbol = msg.body['symbol']
            if symbol != self.symbol:
                return
                
            volume = msg.body['transacted_volume']
            if self.hybrid:
                # 成交量（最近 1 秒）
                volume_cfmm = CFMMAgent.get_transacted_volume_static(self.symbol, '1s')
                self.logEvent('VOLUME', {
                    'timestamp': currentTime,
                    'volume': volume + volume_cfmm
                })
            else:
                self.logEvent('VOLUME', {
                    'timestamp': currentTime,
                    'volume': volume
                })
            
    def getWakeFrequency(self):
        return pd.Timedelta(0)  # 初始唤醒频率，实际会动态调整
    