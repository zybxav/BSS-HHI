class MetricsItem:
    def __init__(self, data, top_k):
        self.data = data
        self.top_k = int(top_k)
        self.true_item_dict = self.data.true_item_cand(self.top_k) 
        self.true_set = set(self.true_item_dict.keys())
        self.true_mass = sum(self.true_item_dict.values())

    def item_f1(self, estimate_item_dict):
        true_items = set(self.true_item_dict.keys())
        pred_items = set(estimate_item_dict.keys())

        tp = len(true_items & pred_items)  # True Positives

        f1 = tp/len(pred_items)
        return round(f1, 4)

    def item_ncr(self, estimate_item_dict):

        true_item_list = list(self.true_item_dict.keys())  # ground-truth top-k, ranked
        k = len(true_item_list)
        denom = k * (k + 1) / 2  # normalization factor
        estimate_items = list(estimate_item_dict.keys())  # predicted ranking

        res = 0
        for item in estimate_items[:k]:
            if item in self.true_item_dict:
                r = true_item_list.index(item) + 1  # 1-based rank in true list
                score = k - (r - 1)  # q(v) = k - (r - 1)
                res += score

        return round(res / denom, 4)



    def cal_mcr(self, pred_topk_items):

        if self.true_mass == 0:
            return 0.0

        pred_set = set(pred_topk_items)
        inter = self.true_set & pred_set

        covered_mass = sum(self.true_item_dict[v] for v in inter)
        return covered_mass / self.true_mass