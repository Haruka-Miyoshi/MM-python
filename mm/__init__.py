import numpy as np

"""Markov Model:マルコフモデル"""
class MarkovModel(object):
    """コンストラクタ"""
    def __init__(self, K:int, M:int) -> None:
        # サイコロの種類K
        self.K = K
        # サイコロの目M
        self.M = M
        # サイコロの種類 遷移行列A[wi, wj] -> P(s_t|s_t-1)
        self.A = None
        # サイコロの目 遷移行列B[wi, vk] -> P(x_t|s_t)
        self.B = None
        # 初期状態分布P(s1) = row
        self.row = np.zeros(self.K)
        # サイコロの種類 頻度
        self.category_histgram = np.zeros((self.K, self.K))
        # サイコロの目　頻度
        self.data_histgram = np.zeros((self.K, self.M))

    """出現回数計算"""
    def calc_histgram(self, categorys:list, data:list) -> None:
        # 出現回数を計算
        self.N = len(categorys)
        for n in range(1, self.N):
            state = int(categorys[n-1])
            next_state = int(categorys[n])
            self.category_histgram[state, next_state] += 1
        
        for t in range(self.N):
            category_ = int(categorys[t])
            data_ = int(data[t]) - 1
            self.data_histgram[category_, data_] += 1
    
    """パラメータ推定"""
    def parameter_inference(self, categorys:list, data:list):
        # 出現回数を計算
        self.calc_histgram(categorys, data)
        
        # 遷移行列Aの推定
        self.A = self.category_histgram
        self.A = self.A.T / np.sum(self.A, axis=1)
        self.A = self.A.T
        print("遷移行列A:" + "\n" + f"{self.A}" + "\n")

        # 遷移行列Bの推定
        self.B = self.data_histgram
        self.B = self.B.T / np.sum(self.B, axis=1)
        self.B = self.B.T
        print("遷移行列B:" + "\n" + f"{self.B}" + "\n")

        # 確率P(s1)の推定
        index = int(categorys[0])
        self.row[index] += 1
        print("確率分布P(s1):" + "\n" + f"{self.row}")

        return self.A, self.B, self.row