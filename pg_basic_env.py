import gym
import gym.spaces as Spaces
from typing import Dict, List
import numpy as np
import random
import pandas as pd
import logging
from stockstats import StockDataFrame as Sdf


class PGBasicEnv(gym.Env):

    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, cfg: Dict,tic_list: List) -> None:
        super().__init__()
        self.cfg = cfg
        self._data_root_dir = cfg["data_root_dir"]
        self.tech_list=cfg["tech_list"]
        self._stock_pool = tic_list
        self.hold=cfg["hold"]
        spaces={
                "graph":Spaces.Box(-1, 1, shape=(1, self.cfg["asset_num"], self.cfg["asset_num"])),
                "px_cov_mat": Spaces.Box(-1, 1, shape=(1, self.cfg["asset_num"], self.cfg["asset_num"])),
                "pct_cov_mat": Spaces.Box(-1, 1, shape=(1, self.cfg["asset_num"], self.cfg["asset_num"])), 
                "current_px": Spaces.Box(-3, 3, shape=(self.cfg["asset_num"],)),
                "current_pct": Spaces.Box(-3, 3, shape=(self.cfg["asset_num"],))
            }
        for item in tic_list:
            spaces["{}_tech".format(item)] = Spaces.Box(float("-inf"), float("inf"), shape=(20, 20))  
        self.observation_space = Spaces.Dict(
           spaces=spaces
        )
        self.action_space = Spaces.Box(low=0, high=1, shape=(self.cfg["asset_num"], ))

    def look_back(self, key_name: str) -> np.ndarray:
        # 前m天每个量的相关性
        sample_ds = []
        for item in self.current_asset_pool:
            sample_data = item["stock_data"][key_name].iloc[self.n - self.cfg["look_back"]:self.n].values
            sample_ds.append(sample_data)
        sample_array = np.stack(sample_ds)
        return sample_array

    def get_tec(self): ##计算technical indicator
        res = {}
        for item in self.current_asset_pool:
            sample_data = item["stock_data"]
            sample_data = sample_data.reset_index()
            sample_data.columns = ["trade_date", "ts_code", "open",
                                   "high", "low", "close",
                                   "pre_close", "change", "pct_chg",
                                   "volume", "amount"]
            tech_frame = Sdf.retype(sample_data.copy())
            res[item["stock_name"]] = tech_frame
        return res
    
    def get_simi(self):
        mv=[]
        n=len(self._stock_pool)
        one=np.identity(n)
        Z=np.identity(n)
        Zv=[]
        M=np.zeros((n,n))
        for item in self.cfg["graph_ft"]:
            mv.append(self.look_back(item))
        k=len(mv)
        w=np.ones(k)/k
        for i in range(5):
            np.maximum(Z,0)
            Z=(Z+Z.T)/2
            Zold=Z
            D=np.diag(np.sum(Z,axis=0))
            L=D-Z
            for j in range(k):
                Z_t=(np.dot(mv[j],mv[j].T)+self.cfg["alpha"]*one+self.cfg["beta"]*w[j]*one)/(self.cfg["beta"]*w[j]*Z+np.dot(mv[j],mv[j].T))
                np.maximum(Z_t,0)
                Z_t=(Z_t+Z_t.T)/2
                Zv.append(Z_t)
                w[j]=0.5/np.linalg.norm(Z_t-Z,ord='fro')
                M=M+w[j]*Z_t
            
            for j  in range(n):
                Z[j,:]=M[j,:]/np.sum(w)
            
            if  np.linalg.norm(Z-Zold,ord='fro') < 0.001:
                break
        return Z
     
    
    def get_state(self):
        graph=self.get_simi()
        px_look_back = self.look_back(self.ob_price) # 用来计算cov信息的
        px_cov_matrix = np.corrcoef(px_look_back)
        px_mean = px_look_back.mean(axis=-1)
        px_std = px_look_back.std(axis=-1)
        px_last = px_look_back[:, -1]
        norm_px = (px_last - px_mean) / (px_std + 1e-5)
        pct_look_back = self.look_back("pct_chg")
        pct_cov_matrix = np.corrcoef(pct_look_back)
        pct_mean = pct_look_back.mean(axis=-1)
        pct_std = pct_look_back.std(axis=-1)
        pct_last = pct_look_back[:, -1]
        norm_pct = (pct_last - pct_mean) / (pct_std + 1e-5)
        norm_px = np.clip(norm_px, -3, 3)
        norm_pct = np.clip(norm_pct, -3, 3)
        graph=graph.astype(np.float32)
        px_cov_matrix = px_cov_matrix.astype(np.float32)
        pct_cov_matrix = pct_cov_matrix.astype(np.float32)
        graph=graph.reshape((1, 10, 10))
        px_cov_matrix = px_cov_matrix.reshape((1, 10, 10))
        pct_cov_matrix = pct_cov_matrix.reshape((1, 10, 10))
        # print("state shape is {}".format(px_cov_matrix.shape))
        norm_px = norm_px.astype(np.float32)
        norm_pct = norm_pct.astype(np.float32)     
        res = {
            "graph":graph,
            "px_cov_mat": px_cov_matrix,
            "pct_cov_mat": pct_cov_matrix,
            "current_px": norm_px,
            "current_pct": norm_pct
        }
        for item in self.tec_inc:
            stock_data = self.tec_inc[item]
            tech_data = stock_data[self.tech_list][self.n - 20:self.n].values
            res["{}_tech".format(item)] = tech_data
        return res

    def reset(self) -> Dict:
        return self._reset()

    def _reset(self) -> Dict:
        """
        随机选择N支股票，进行投资组合测试
        """
        #_select_asset = random.sample(self._stock_pool, self.cfg["asset_num"])
        # inner idx
        _first_data = pd.read_csv("{}/{}.csv".format(self._data_root_dir, self._stock_pool[0]), index_col="trade_date")
        _inner_idx = _first_data.index
        self.current_asset_pool = []
        for item in self._stock_pool:
            _tmp_data = pd.read_csv("{}/{}.csv".format(self._data_root_dir, item), index_col="trade_date")
            _inner_idx = _inner_idx.join(_tmp_data.index, how="inner")
            self.current_asset_pool.append({"stock_name": item, "stock_data": _tmp_data})
        for i in range(len(self.current_asset_pool)):
            self.current_asset_pool[i]["stock_data"] = self.current_asset_pool[i]["stock_data"].loc[_inner_idx]
        #logging.info("select {} stocks randomly".format(len(self.current_asset_pool))) 
        #logging.info("stocks are:")
        #for item in self.current_asset_pool:
        #    logging.info(item["stock_name"])
        # init your asset pos
        self._idx = _inner_idx
        self.n = self.cfg["look_back"] # daily exchange set 30, 
        self.init_n = self.n
        self.in_time = self.cfg["in_time"] # open, close, vwap， 我们用昨天之前的状态作为我们的观测，今天的开盘价的时刻进行交易
        self.ob_price = self.cfg["ob_price"]
        self.out_time = self.cfg["out_time"]
        self.cur_day = _inner_idx[self.n]
        self.total_money = 100000.0
        self.asset_allocation = None
        self.daily_return=[]
        self.info = {}
        self.info["date"] = self.cur_day
        self.info["money"] = self.total_money
        self.wlist=[]
        self.wlist.append([[0]*10])
        # self.info["daily_return"]=self.daily_return
        self.tec_inc = self.get_tec()
        state = self.get_state()
        return state

    def step(self, action: np.ndarray) -> tuple:
        return self._step(action)

    def _step(self, action: np.ndarray) -> tuple:
        # 进场
        # 对action进行一个softmax
        # print("action is {}".format(action))
        action = np.exp(action)
        action = action / action.sum()
        self.cur_day = self._idx[self.n]
        self.asset_allocation = action
        # 配额度，那股票
        stock_money = self.total_money * self.asset_allocation
        # 股票单价
        if self.n+self.hold == len(self._idx):
            hold=len(self._idx)-self.n
            done = True
        else:
            hold=self.hold
            done =False
        self.out_day = self._idx[self.n+hold-1]
        in_pxs = []
        out_pxs = []
        for item in self.current_asset_pool:
            in_px = item["stock_data"].loc[self.cur_day][self.in_time]
            out_px = item["stock_data"].loc[self.out_day][self.out_time]
            in_pxs.append(in_px)
            out_pxs.append(out_px)
        pxs = np.array(in_pxs, dtype=np.float32)
        stock_share = stock_money / pxs
        last_stock_money = self.total_money
        # holding to out
        out_pxs = np.array(out_pxs, dtype=np.float32)
        # sell to total money
        self.total_money = (stock_share * out_pxs).sum()-0.001 * (last_stock_money * np.maximum(self.wlist[-1]-action,0)).sum()
        # 0.14% 的交易成本
        #exchange_cost = self.total_money * 0.0014
        #self.total_money = self.total_money - exchange_cost
        #append exchange cost
        earn_money = self.total_money - last_stock_money
        earn_rate = earn_money / last_stock_money
        market_vol = out_pxs / in_pxs - 1
        #log for exchange information
        #logging.info("in day {}, you earn money {}, earn rate {}, "
        #             "current money is {}".format(self.cur_day, earn_money, earn_rate, self.total_money))
        self.info["date"] = self.cur_day
        self.info["money"] = self.total_money
        r = earn_rate - market_vol.mean()
        self.n += hold
        #if self.n == len(self._idx):
        #    done = True
        #else:
        #    done =False
        #self.cur_day = self._idx[self.n]
        self.wlist.append(action)
        state = self.get_state()
        return state, r, done, self.info
    
    def seed(self, seed) -> None:
        random.seed(seed)
    
    def render(self, mode="human"):
        pass

    def close():
        pass
        
        
        
    
        
        