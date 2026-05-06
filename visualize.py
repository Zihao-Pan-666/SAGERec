import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 导入项目中的模块
from baselines.bert4rec import BERT4RecWithDomainAlignment
from utils import resolve_embedding_path, load_pretrained_embeddings

# ==========================================
# 1. 配置参数与路径
# ==========================================
SOURCE_DOMAIN = "amazon_movies_and_tv"
TARGET_DOMAIN_1 = "amazon_cds_and_vinyl"
TARGET_DOMAIN_2 = "steam"  # 新增第二个目标域

CKPT_DIR = "./saved_ckpts/"
CKPT_SEM = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_sem_loss.pth")
CKPT_RECG = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_recg_loss.pth")
CKPT_SAGE = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_sage_loss.pth")

HIDDEN_UNITS = 256
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_LEN = 50

# 采样数量调整：每个域采 300 个点，总共 900 个点，保证图面丰满不拥挤
TSNE_SAMPLES = 300
HEATMAP_SAMPLES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 辅助函数
# ==========================================
def load_domain_raw_embeddings(domain_name):
    emb_path = resolve_embedding_path(domain_name)
    return load_pretrained_embeddings(emb_path)


def init_model(ckpt_path, source_raw_embs):
    model = BERT4RecWithDomainAlignment(
        hidden_units=HIDDEN_UNITS, max_seq_length=MAX_LEN,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout_rate=0.0,
        pretrained_item_embeddings=source_raw_embs
    ).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"成功加载模型: {ckpt_path}")
    else:
        print(f"警告: 找不到模型文件 {ckpt_path}，将使用随机初始化权重！")
    return model


def compute_cosine_similarity(x, y, eps=1e-8):
    x_n = F.normalize(x, p=2, dim=1, eps=eps)
    y_n = F.normalize(y, p=2, dim=1, eps=eps)
    return torch.matmul(x_n, y_n.T)

def row_minmax_normalize(mat, eps=1e-8):
    """
    对每一行做 min-max 归一化到 [0, 1]，
    用于突出每个 source item 对 target items 的相对偏好模式。
    """
    row_min = mat.min(axis=1, keepdims=True)
    row_max = mat.max(axis=1, keepdims=True)
    return (mat - row_min) / (row_max - row_min + eps)


def get_consistent_order(mat):
    """
    根据 SAGE 矩阵的峰值位置对行列排序，
    让局部结构更清晰，并保证两张图使用相同顺序。
    """
    row_peak = np.argmax(mat, axis=1)
    row_strength = np.max(mat, axis=1)

    col_peak = np.argmax(mat, axis=0)
    col_strength = np.max(mat, axis=0)

    row_order = np.lexsort((-row_strength, row_peak))
    col_order = np.lexsort((-col_strength, col_peak))
    return row_order, col_order

# ==========================================
# 3. 提取特征的主流程 (1源 + 2目标)
# ==========================================
print("正在加载原始语义 Embeddings...")
raw_source_full = load_domain_raw_embeddings(SOURCE_DOMAIN).to(device)
raw_target1_full = load_domain_raw_embeddings(TARGET_DOMAIN_1).to(device)
raw_target2_full = load_domain_raw_embeddings(TARGET_DOMAIN_2).to(device)

models = {
    "-Sem": init_model(CKPT_SEM, raw_source_full),
    "-RecG": init_model(CKPT_RECG, raw_source_full),
    "SAGERec": init_model(CKPT_SAGE, raw_source_full)
}

# 去除索引 0 的 padding
raw_source_items = raw_source_full[1:]
raw_target1_items = raw_target1_full[1:]
raw_target2_items = raw_target2_full[1:]

torch.manual_seed(2026)
idx_source = torch.randperm(raw_source_items.size(0))[:TSNE_SAMPLES]
idx_target1 = torch.randperm(raw_target1_items.size(0))[:TSNE_SAMPLES]
idx_target2 = torch.randperm(raw_target2_items.size(0))[:TSNE_SAMPLES]

sample_raw_s = raw_source_items[idx_source]
sample_raw_t1 = raw_target1_items[idx_target1]
sample_raw_t2 = raw_target2_items[idx_target2]

projected_features = {}

with torch.no_grad():
    for name, model in models.items():
        if name == "-Sem":
            proj_s = model.projection_layer(sample_raw_s)
            proj_t1 = model.projection_layer(sample_raw_t1)
            proj_t2 = model.projection_layer(sample_raw_t2)
        else:
            proj_s = model.domain_alignment_projection_layer(sample_raw_s)
            proj_t1 = model.domain_alignment_projection_layer(sample_raw_t1)
            proj_t2 = model.domain_alignment_projection_layer(sample_raw_t2)

        projected_features[name] = (proj_s.cpu().numpy(), proj_t1.cpu().numpy(), proj_t2.cpu().numpy())

# # ==========================================
# # 4. 绘制 t-SNE 对比图 (三色版 + 标注增大)
# # ==========================================
# print("正在运行 t-SNE 降维...")
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# titles = [
#     '(a) -Sem: Separated Spaces',
#     '(b) -RecG: Uniformly Over-aligned',
#     '(c) SAGERec: Heterogeneous & Selective'
# ]
#
# # 颜色配置：源域(蓝), 目标1(橙), 目标2(绿)
# colors = ['#4C72B0', '#DD8452', '#55A868']
# labels = ['Source (AMT)', 'Target 1 (ACV)', 'Target 2 (Steam)']
#
# for ax, (name, title) in zip(axes, zip(["-Sem", "-RecG", "SAGERec"], titles)):
#     proj_s, proj_t1, proj_t2 = projected_features[name]
#     combined = np.vstack((proj_s, proj_t1, proj_t2))
#
#     # 增加 perplexity 以适应更多的数据点
#     tsne = TSNE(n_components=2, perplexity=35, random_state=42, init='pca', learning_rate='auto')
#     reduced = tsne.fit_transform(combined)
#
#     s_data = reduced[:TSNE_SAMPLES]
#     t1_data = reduced[TSNE_SAMPLES:2 * TSNE_SAMPLES]
#     t2_data = reduced[2 * TSNE_SAMPLES:]
#
#     # 增大点的大小 s=25，增强视觉冲击力
#     ax.scatter(s_data[:, 0], s_data[:, 1], c=colors[0], label=labels[0], alpha=0.6, s=25, edgecolors='none')
#     ax.scatter(t1_data[:, 0], t1_data[:, 1], c=colors[1], label=labels[1], alpha=0.6, s=25, edgecolors='none')
#     ax.scatter(t2_data[:, 0], t2_data[:, 1], c=colors[2], label=labels[2], alpha=0.6, s=25, edgecolors='none')
#
#     # 标题字体增大且加粗
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # 图注放在左下角，并增大字号
#     ax.legend(loc='lower left', fontsize=12, frameon=True, edgecolor='black')
#
# plt.tight_layout()
# plt.savefig('real_tsne_3domains_updated.pdf', dpi=300, bbox_inches='tight')

# ==========================================
# 5. 绘制 Heatmap（统一尺度 + 统一排序 + 更可比）
# ==========================================
print("正在绘制可比较的 Heatmap...")

sample_raw_s_heat = raw_source_items[idx_source[:HEATMAP_SAMPLES]]
sample_raw_t1_heat = raw_target1_items[idx_target1[:HEATMAP_SAMPLES]]

# 原始跨域相似度
sim_matrix = compute_cosine_similarity(sample_raw_s_heat, sample_raw_t1_heat)

# 两种方法对应的原始权重矩阵
sage_w_ij = F.relu(sim_matrix).cpu().numpy()
recg_dense_sim = (sim_matrix / 0.1)
recg_w_ij = F.softmax(recg_dense_sim, dim=1).cpu().numpy()

# === 关键修改 1：统一成“相对模式可比”的可视化矩阵 ===
# 对每一行做 min-max 归一化到 [0,1]
sage_vis = row_minmax_normalize(sage_w_ij)
recg_vis = row_minmax_normalize(recg_w_ij)

# === 关键修改 2：使用同一排序，让局部结构更清晰 ===
# 用 SAGE 的峰值分布来决定行列顺序，再同步作用到两张图
row_order, col_order = get_consistent_order(sage_vis)

sage_vis = sage_vis[row_order][:, col_order]
recg_vis = recg_vis[row_order][:, col_order]

# === 关键修改 3：统一色标范围和显示风格 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

titles = [
    'Implicit Pull in -RecG',
    'Selective Weights in -SAGE'
]

heatmaps = []
for ax, data, title in zip(axes, [recg_vis, sage_vis], titles):
    hm = sns.heatmap(
        data,
        cmap="YlGnBu",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cbar=False,           # 先不单独画 colorbar，后面统一加一个
        square=True,
        xticklabels=5,
        yticklabels=5,
        linewidths=0.0
    )
    heatmaps.append(hm)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Target Items (ACV)', fontsize=14)
    ax.set_ylabel('Source Items (AMT)', fontsize=14)
    ax.tick_params(labelsize=10)

# 共用一个 colorbar，明确两图是同一范围 [0,1]
cbar = fig.colorbar(
    heatmaps[-1].collections[0],
    ax=axes,
    fraction=0.03,
    pad=0.02
)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Relative response (row-normalized)', fontsize=12)

plt.savefig('real_heatmap_weights_updated.pdf', dpi=300, bbox_inches='tight')
print("更新后的 Heatmap 已保存：real_heatmap_weights_updated.pdf")
