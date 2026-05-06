import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_full_catalog_structural(
        model,
        dataloader,
        num_items,
        device,
        top_k=10,
        is_sem_baseline=False,
        ild_use_raw_semantic=True,
):
    """
    计算 Catalog Coverage@K 和 ILD@K。

    Coverage@K:
        所有用户 Top-K 推荐列表中出现过的不同物品数 / 物品总数。

    ILD@K:
        每个用户 Top-K 推荐列表中 item embedding 两两距离的平均值。
        默认使用 raw semantic embedding 计算 item semantic diversity。
    """
    model.eval()

    recommended_items = set()
    ild_sum = 0.0
    total_users = 0

    with torch.no_grad():
        # 推荐打分使用 projection 后的 item 表征，保持与 sampled evaluation 一致
        all_item_embs = model.projection_layer(model.pretrained_item_embedding.weight)
        all_item_embs[0] = 0.0

        # ILD 默认用原始语义 embedding，更符合“列表语义多样性”的定义
        if ild_use_raw_semantic:
            ild_item_embs = model.pretrained_item_embedding.weight
        else:
            ild_item_embs = all_item_embs

        for batch in tqdm(dataloader, desc=f"Structural@{top_k}", leave=False):
            train_seq, val_item, test_item = batch
            train_seq = train_seq.to(device)
            val_item = val_item.to(device)

            eval_seq = torch.cat([train_seq[:, 1:], val_item.unsqueeze(1)], dim=1)
            user_rep = model.encode_sequence(eval_seq, is_sem_baseline=is_sem_baseline)

            scores = torch.matmul(user_rep, all_item_embs.T)
            scores[:, 0] = -1e9

            # 排除用户历史和验证物品，避免推荐已交互物品
            history = torch.cat([train_seq, val_item.unsqueeze(1)], dim=1)
            for i in range(history.size(0)):
                h = history[i]
                h = h[h > 0]
                scores[i, h] = -1e9

            top_items = torch.topk(scores, k=top_k, dim=1).indices

            for row in top_items:
                item_list = row.tolist()
                recommended_items.update(item_list)

                emb = ild_item_embs[row]
                emb = F.normalize(emb, p=2, dim=1)
                sim = torch.matmul(emb, emb.T)

                k = emb.size(0)
                if k > 1:
                    # 只取非对角线
                    diversity_matrix = 1.0 - sim
                    diversity = (diversity_matrix.sum() - torch.diag(diversity_matrix).sum()) / (k * (k - 1))
                    ild_sum += diversity.item()

                total_users += 1

    coverage = len(recommended_items) / float(max(num_items, 1))
    ild = ild_sum / float(max(total_users, 1))

    return coverage, ild
