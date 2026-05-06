import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 颜色与标签配置
# ==========================================
COLORS = {'Sem': '#4C72B0', 'RecG': '#DD8452', 'SAGE': '#55A868'}
BACKBONES = ['BERT4Rec', 'SASRec', 'GRU4Rec']
VARIANTS = ['Sem', 'RecG', 'SAGE']

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 16})


# ==========================================
# 2. 日志解析核心逻辑
# ==========================================
def extract_times(backbone, variant):
    pattern = f"*{backbone.lower()}*{variant.lower()}*"
    files = glob.glob(pattern + ".log") + glob.glob(pattern + ".txt")
    if not files: return []

    times = []
    with open(files[0], 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'Time:\s+([\d\.]+)\s*s', line, re.IGNORECASE)
            if match:
                times.append(float(match.group(1)))
    return times


# ==========================================
# 3. 平滑函数 (EMA - 指数移动平均)
# ==========================================
def smooth_curve(points, factor=0.75):
    """
    factor 越大，曲线越平滑 (0 <= factor < 1)
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# ==========================================
# 4. 绘制单图 (正文排版用)
# ==========================================
def plot_single_backbone(target_backbone='BERT4Rec'):
    plt.figure(figsize=(7, 5))

    for variant in VARIANTS:
        times = extract_times(target_backbone, variant)
        if not times: continue

        epochs = list(range(1, len(times) + 1))
        label_name = 'SAGERec' if variant == 'SAGE' else f'-{variant}'
        smoothed_times = smooth_curve(times, factor=0.8)  # 平滑因子设为 0.8

        # 先画真实的带毛刺曲线，设为半透明作为底纹
        plt.plot(epochs, times, color=COLORS[variant], linewidth=1.0, alpha=0.25)

        # 再画平滑后的主曲线
        plt.plot(epochs, smoothed_times, label=f"{target_backbone}{label_name}",
                 color=COLORS[variant], linewidth=2.5, alpha=1.0)

    plt.title(f'Per-Epoch Training Time on {target_backbone}', pad=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Epoch Time (s)')
    plt.xlim(left=0)
    plt.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(f'efficiency_curve_{target_backbone.lower()}_smooth.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 5. 绘制 1x3 全量大图
# ==========================================
def plot_all_backbones():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ['(a) BERT4Rec', '(b) SASRec', '(c) GRU4Rec']

    for i, backbone in enumerate(BACKBONES):
        ax = axes[i]
        for variant in VARIANTS:
            times = extract_times(backbone, variant)
            if not times: continue

            epochs = list(range(1, len(times) + 1))
            label_name = 'SAGERec' if variant == 'SAGE' else f'-{variant}'
            smoothed_times = smooth_curve(times, factor=0.8)

            # 半透明原始数据背景
            ax.plot(epochs, times, color=COLORS[variant], linewidth=1.0, alpha=0.25)
            # 实线平滑主数据
            ax.plot(epochs, smoothed_times, label=f"{backbone}{label_name}",
                    color=COLORS[variant], linewidth=2.5, alpha=1.0)

        ax.set_title(titles[i], pad=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Epoch Time (s)')
        ax.set_xlim(left=0)
        ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig('efficiency_curve_all_smooth.pdf', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_single_backbone('BERT4Rec')
    plot_all_backbones()
    print("带有 TensorBoard 风格平滑阴影的图表已生成完毕！")