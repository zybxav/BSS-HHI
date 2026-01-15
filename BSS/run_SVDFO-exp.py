from datetime import datetime
import sys
import json

import numpy as np

sys.path.append("..")
from data_rm import Data

from metrics_item import MetricsItem
from svim import SVIM
from svd_fo import LDP_RM



eps_list = [1 ,1.5, 2, 2.5, 3, 3.5, 4]
k_list = [16, 32, 48, 64, 80, 96, 112, 128]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"../results/svdfo_kos_{timestamp}.json"

with open(filename, "w") as f:
    json.dump([], f)

for eps in eps_list:
    for topk in k_list:
        f1_list = []
        ncr_list = []
        mcr_list = []
        data = Data(dataname='kosarak', limit=100000, domain_size=41270, user_total=990002)
        data_pair = Data(dataname='kosarak_pair', limit=100000, domain_size=204, user_total=990002)
        # data = Data(dataname='movielens', limit=100000, domain_size=209163, user_total=162414)
        # data_pair = Data(dataname='movielens_pairs', limit=100000, domain_size=458, user_total=162414)
        # data = Data(dataname='UserBehavior1300', limit=100000, domain_size=528150, user_total=863327)
        # data_pair = Data(dataname='UserBehavior1300_pairs', limit=100000, domain_size=727, user_total=863327)
        metrics = MetricsItem(data, top_k=topk)
        for repeat in range(10):
            ldp_rm = LDP_RM(data_pair, epsilon=eps, top_k=64, top_ks=topk, top_kc=topk, submat=0)
            estimate_dict_svd = ldp_rm.find_item_svd(method='AMN', singnum=0, use_group=True, group_num=5, test='test_constant')
            f1 = metrics.item_f1(estimate_dict_svd)
            ncr = metrics.item_ncr(estimate_dict_svd)
            mcr = metrics.cal_mcr(estimate_dict_svd)
            print(f"[eps={eps}, k={topk}] → F1: {f1:.4f}, NCR: {ncr:.4f}, MCR: {mcr:.4f}")
            f1_list.append(f1)
            ncr_list.append(ncr)
            mcr_list.append(mcr)
            single_result = {
                'epsilon': eps,
                'top_k': topk,
                'repeat_id': repeat + 1,
                'f1': round(f1, 4),
                'ncr': round(ncr, 4),
                'mcr': round(mcr, 4)
            }
            with open(filename, "r+") as f:
                data = json.load(f)
                data.append(single_result)
                f.seek(0)
                json.dump(data, f, indent=2)
        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
        ncr_mean, ncr_std = np.mean(ncr_list), np.std(ncr_list)
        mcr_mean, mcr_std = np.mean(mcr_list), np.std(mcr_list)
        result_entry = {
            'epsilon': eps,
            'top_k': topk,
            'f1_mean': round(f1_mean, 4),
            'f1_std': round(f1_std, 4),
            'ncr_mean': round(ncr_mean, 4),
            'ncr_std': round(ncr_std, 4),
            'mcr_mean': round(mcr_mean,4),
            'mcr_std': round(mcr_std,4)
        }

        print(f"[eps={eps}, k={topk}] → F1: {f1_mean:.4f} ± {f1_std:.4f}, NCR: {ncr_mean:.4f}, MCR: {mcr_mean:.4f} ± {mcr_std:.4f}")
        with open(filename, "r+") as f:
            data = json.load(f)
            data.append(result_entry)
            f.seek(0)
            json.dump(data, f, indent=2)

print(f"\nAll results saved to: {filename}")
# print('f1',F1_svd,'\nNCR',NCR_svd,'\nVAR',VARt_svd)