import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 导入你项目中的模块
from baselines.bert4rec import BERT4RecWithDomainAlignment
from utils import resolve_embedding_path, load_pretrained_embeddings

# ==========================================
# 1. 配置参数与路径
# ==========================================
SOURCE_DOMAIN = "amazon_movies_and_tv"
TARGET_DOMAIN_1 = "amazon_cds_and_vinyl"
TARGET_DOMAIN_2 = "steam"

CKPT_DIR = "./saved_ckpts/"
CKPT_SEM = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_sem_loss.pth")
CKPT_RECG = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_recg_loss.pth")
CKPT_SAGE = os.path.join(CKPT_DIR, f"bert4rec_{SOURCE_DOMAIN}_sage_loss.pth")

HIDDEN_UNITS = 256
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_LEN = 50

# 采样数量调整：因为有3个域，每个域采300个，总计900个点，保证图面充实但不拥挤
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

# 去除 padding
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

# ==========================================
# 4. 绘制 t-SNE 对比图 (3个域)
# ==========================================
print("正在运行 t-SNE 降维 (这可能需要一两分钟)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = [
    '(a) -Sem: Separated Spaces',
    '(b) -RecG: Uniformly Over-aligned',
    '(c) SAGERec: Heterogeneous & Selective Alignment'
]

# 定义三个域的颜色: 源域(蓝), 目标1(橙), 目标2(绿)
colors = ['#4C72B0', '#DD8452', '#55A868']
labels = ['Source (AMT)', 'Target 1 (ACV)', 'Target 2 (Steam)']

for ax, (name, title) in zip(axes, zip(["-Sem", "-RecG", "SAGERec"], titles)):
    proj_s, proj_t1, proj_t2 = projected_features[name]
    combined = np.vstack((proj_s, proj_t1, proj_t2))

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)

    s_data = reduced[:TSNE_SAMPLES]
    t1_data = reduced[TSNE_SAMPLES:2 * TSNE_SAMPLES]
    t2_data = reduced[2 * TSNE_SAMPLES:]

    ax.scatter(s_data[:, 0], s_data[:, 1], c=colors[0], label=labels[0], alpha=0.6, s=15, edgecolors='none')
    ax.scatter(t1_data[:, 0], t1_data[:, 1], c=colors[1], label=labels[1], alpha=0.6, s=15, edgecolors='none')
    ax.scatter(t2_data[:, 0], t2_data[:, 1], c=colors[2], label=labels[2], alpha=0.6, s=15, edgecolors='none')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('real_tsne_3domains.pdf', dpi=300, bbox_inches='tight')
print("3个域的 t-SNE 可视化已保存为 real_tsne_3domains.pdf")

# ==========================================
# 5. 绘制 Heatmap (保持 1对1 比较，专注于证明选择性)
# ==========================================
sample_raw_s_heat = raw_source_items[idx_source[:HEATMAP_SAMPLES]]
sample_raw_t1_heat = raw_target1_items[idx_target1[:HEATMAP_SAMPLES]]

sim_matrix = compute_cosine_similarity(sample_raw_s_heat, sample_raw_t1_heat)
sage_w_ij = F.relu(sim_matrix).cpu().numpy()

recg_dense_sim = (sim_matrix / 0.1)
recg_w_ij = F.softmax(recg_dense_sim, dim=1).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(recg_w_ij, cmap="YlGnBu", ax=axes[0], cbar=True)
axes[0].set_title('Implicit Cross-Domain Pull in -RecG\n(Dense & Non-selective)', fontsize=14)
axes[0].set_xlabel('Target Items (ACV)')
axes[0].set_ylabel('Source Items (AMT)')

sns.heatmap(sage_w_ij, cmap="YlGnBu", ax=axes[1], cbar=True)
axes[1].set_title('Similarity-Aware Weights in SAGERec ($w_{ij}$)\n(Sparse & Selective)', fontsize=14)
axes[1].set_xlabel('Target Items (ACV)')
axes[1].set_ylabel('Source Items (AMT)')

plt.tight_layout()
plt.savefig('real_heatmap_weights.pdf', dpi=300, bbox_inches='tight')
print("Heatmap 可视化已保存为 real_heatmap_weights.pdf")