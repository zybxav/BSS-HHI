from pem import PEM
import matplotlib.pyplot as plt
import numpy as np
from data import data


from collections import Counter

def load_true_topk_from_multivalue_txt(filename: str, k: int, delimiter=None):
    counter = Counter()
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(delimiter)
            for it in items:
                it = it.strip()
                if not it:
                    continue
                counter[int(it)] += 1
    topk = counter.most_common(k)
    return [x for x, _ in topk], [c for _, c in topk]

def evaluate(cand_dict, data, k):
    pred_topk = sorted(cand_dict.keys(), key=lambda x: cand_dict[x], reverse=True)[:k]
    true_topk = data.heavy_hitters[:k]

    c = []
    for i in range(len(true_topk)):
        if true_topk[i] in pred_topk:
            c.append(true_topk[i])
    lc = len(c)

    p = lc / len(true_topk) if len(true_topk) > 0 else 0
    r = lc / len(pred_topk) if len(pred_topk) > 0 else 0
    if p == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    hh_dict = dict(zip(data.heavy_hitters, data.heavy_hitter_freq))
    dist_ct = [hh_dict.get(item, 0) for item in true_topk]
    dist_cg = [cand_dict.get(item, 0) for item in pred_topk]

    dist_t1 = [0 for _ in range(k)]
    dist_t2 = [0 for _ in range(k)]
    for i in range(k):
        dist_t1[i] = dist_ct[i]
        dist_t2[i] = dist_cg[i]

    order_ct, order_cg = [], []
    for i in range(k):
        t_index1 = dist_t1.index(max(dist_t1))
        order_ct.append(true_topk[t_index1])
        dist_t1[t_index1] = 0

        t_index2 = dist_t2.index(max(dist_t2))
        order_cg.append(pred_topk[t_index2])
        dist_t2[t_index2] = 0

    ncr_score = 0
    for i in range(k):
        t = order_ct[i]
        if t in order_cg:
            ncr_score += (k - order_ct.index(t))

    ncr = ncr_score / (k * (k + 1) / 2)

    return f1, ncr



def cal_mcr(Real_k_data, Pred_k_data, Real_k_dist):
    if not isinstance(Real_k_dist, dict):
        Real_k_dist = {Real_k_data[i]: Real_k_dist[i] for i in range(len(Real_k_data))}

    true_set = set(Real_k_data)
    pred_set = set(Pred_k_data)

    inter = true_set & pred_set

    numerator = sum(Real_k_dist[v] for v in inter)
    denominator = sum(Real_k_dist[v] for v in true_set)

    return numerator / denominator if denominator else 0


top_k_list = [32]
length = 2
eps_values = [1,1.5,2,2.5,3,3.5]


n_bits = 16
input_file = "kosarak.txt"#change with data
true_source_file = "kosarak_true.txt"
output_file = "kos_f1_ncr_mcr_single_result.txt"
output_pred_distribution_file = "kos_pred_distribution_single_result.txt"
output_true_distribution_file = "kos_true_distribution_single_result.txt"
K_TRUE = max(top_k_list)
TRUE_ITEMS, TRUE_FREQS = load_true_topk_from_multivalue_txt(true_source_file, K_TRUE, delimiter=None)

with open(output_file, "a") as f_out, \
     open(output_pred_distribution_file, "a") as f_dist, \
     open(output_true_distribution_file, "a") as f_true_dist:

    f_out.write("eps\tk\trun\tf1\tncr\tmcr\n")
    f_dist.write("eps\tk\trun\titem:frequency (top-k distribution)\n")
    f_true_dist.write("eps\tk\trun\titem:frequency (top-k true distribution)\n")

    for k in top_k_list:
        for eps in eps_values:

            f1_runs, ncr_runs, mcr_runs = [], [], []

            for run in range(10):
                data = data(n_bits=n_bits, heavy_size=k, input_file=input_file)
                finder = PEM(data, top_k=k, length=length, eps=eps, n_bits=n_bits)
                cand_dict = finder.find()

                f1, ncr = evaluate(cand_dict, data, k)
                f1_runs.append(f1)
                ncr_runs.append(ncr)


                pred_topk = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)[:k]
                pred_topk_items = [item for item, _ in pred_topk]

                real_topk_items = TRUE_ITEMS[:k]
                real_topk_freqs = TRUE_FREQS[:k]

                mcr = cal_mcr(
                    Real_k_data=real_topk_items,
                    Pred_k_data=pred_topk_items,
                    Real_k_dist=real_topk_freqs
                )
                mcr_runs.append(mcr)


                pred_dist_str = ", ".join([f"{item}:{freq:.4f}" for item, freq in pred_topk])
                true_dist_str = ", ".join(
                    [f"{item}:{freq:.4f}" for item, freq in zip(real_topk_items, real_topk_freqs)]
                )

                f_dist.write(f"{eps}\t{k}\t{run}\t{pred_dist_str}\n")
                f_true_dist.write(f"{eps}\t{k}\t{run}\t{true_dist_str}\n")


                f_out.write(f"{eps}\t{k}\t{run}\t{f1:.4f}\t{ncr:.4f}\t{mcr:.4f}\n")

            mean_f1, std_f1 = np.mean(f1_runs), np.std(f1_runs)
            mean_ncr, std_ncr = np.mean(ncr_runs), np.std(ncr_runs)
            mean_mcr, std_mcr = np.mean(mcr_runs), np.std(mcr_runs)

            result_line = (
                f"{eps}\t{k}\tmean\t"
                f"{mean_f1:.4f}±{std_f1:.4f}\t"
                f"{mean_ncr:.4f}±{std_ncr:.4f}\t"
                f"{mean_mcr:.4f}±{std_mcr:.4f}\n"
            )

            print(result_line.strip())
            f_out.write(result_line)
