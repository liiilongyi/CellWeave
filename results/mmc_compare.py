import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd

# =========================
# 1. 读取两个注释文件
# =========================
our_file = "<MAPMYCELLS_ANNOTATION_CSV>"
bin8_file = "<MAPMYCELLS_ANNOTATION_CSV>"

our_mmc = pd.read_csv(our_file, comment="#")
bin8_mmc = pd.read_csv(bin8_file, comment="#")

# =========================
# 2. 定义要统计的列
# =========================
prob_cols = [
    "subclass_bootstrapping_probability",
    "supertype_bootstrapping_probability",
    "cluster_bootstrapping_probability"
]

# 转成数值型，防止被读成字符串
for col in prob_cols:
    our_mmc[col] = pd.to_numeric(our_mmc[col], errors="coerce")
    bin8_mmc[col] = pd.to_numeric(bin8_mmc[col], errors="coerce")

# =========================
# 3. 计算整体平均值
# =========================
print("===== our_seg 全部对象的平均注释置信度 =====")
for col in prob_cols:
    print(f"{col}: mean = {our_mmc[col].mean():.4f}, median = {our_mmc[col].median():.4f}, n = {our_mmc[col].notna().sum()}")

print("\n===== 8um bin 全部对象的平均注释置信度 =====")
for col in prob_cols:
    print(f"{col}: mean = {bin8_mmc[col].mean():.4f}, median = {bin8_mmc[col].median():.4f}, n = {bin8_mmc[col].notna().sum()}")

exit()

# =========================
# 1. 读取两个注释文件
# =========================
our_file = "<MAPMYCELLS_ANNOTATION_CSV>"
bin8_file = "<MAPMYCELLS_ANNOTATION_CSV>"

our_mmc = pd.read_csv(our_file, comment="#")
bin8_mmc = pd.read_csv(bin8_file, comment="#")

our_mmc["cell_id"] = our_mmc["cell_id"].astype(str)
bin8_mmc["cell_id"] = bin8_mmc["cell_id"].astype(str)

# =========================
# 2. 定义层级和对应 probability 列
# =========================
level2prob = {
    "subclass_name": "subclass_bootstrapping_probability",
    "supertype_name": "supertype_bootstrapping_probability",
    "cluster_name": "cluster_bootstrapping_probability"
}

# =========================
# 3. 工具函数：找 unique labels
# =========================
def get_unique_labels(df1, df2, label_col):
    s1 = set(df1[label_col].dropna().astype(str).unique())
    s2 = set(df2[label_col].dropna().astype(str).unique())
    only_in_df1 = sorted(s1 - s2)
    only_in_df2 = sorted(s2 - s1)
    return only_in_df1, only_in_df2

# =========================
# 4. 工具函数：统计 unique labels 的 probability
# =========================
def summarize_unique_label_confidence(df, label_col, prob_col, unique_labels, group_name):
    """
    对一组 unique labels，对应的所有对象的 probability 做统计
    """
    sub = df[df[label_col].astype(str).isin(unique_labels)].copy()

    # probability 转数值，防止读入成字符串
    sub[prob_col] = pd.to_numeric(sub[prob_col], errors="coerce")

    print(f"\n----- {group_name} -----")
    print(f"unique {label_col} 数量: {len(unique_labels)}")
    print(f"这些 unique labels 对应的对象总数: {len(sub)}")

    if len(sub) == 0:
        print("无可统计对象")
        return None

    probs = sub[prob_col].dropna()

    if len(probs) == 0:
        print("probability 全为缺失")
        return None

    print(f"{prob_col} overall mean   : {probs.mean():.4f}")
    print(f"{prob_col} overall median : {probs.median():.4f}")
    print(f"{prob_col} overall std    : {probs.std():.4f}")
    print(f"{prob_col} overall min    : {probs.min():.4f}")
    print(f"{prob_col} overall max    : {probs.max():.4f}")

    # 每个 unique label 的统计
    per_label = (
        sub.groupby(label_col)[prob_col]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values(["mean", "count"], ascending=[False, False])
    )

    print(f"\n每个 unique {label_col} 的置信度统计：")
    print(per_label.to_string())

    return per_label

# =========================
# 5. 主循环：逐层级比较
# =========================
for label_col, prob_col in level2prob.items():
    print("\n" + "=" * 80)
    print(f"层级: {label_col}  |  置信度列: {prob_col}")
    print("=" * 80)

    only_our, only_bin8 = get_unique_labels(our_mmc, bin8_mmc, label_col)

    print(f"\n仅在 our_seg 中出现的 {label_col} 数: {len(only_our)}")
    print(f"仅在 bin8 中出现的 {label_col} 数: {len(only_bin8)}")

    summarize_unique_label_confidence(
        our_mmc, label_col, prob_col, only_our, f"only in our_seg ({label_col})"
    )

    summarize_unique_label_confidence(
        bin8_mmc, label_col, prob_col, only_bin8, f"only in bin8 ({label_col})"
    )

exit()


# =========================
# 1. 读取两个注释文件
# =========================
our_file = "<MAPMYCELLS_ANNOTATION_CSV>"
bin8_file = "<MAPMYCELLS_ANNOTATION_CSV>"

our_mmc = pd.read_csv(our_file, comment="#")
bin8_mmc = pd.read_csv(bin8_file, comment="#")

our_mmc["cell_id"] = our_mmc["cell_id"].astype(str)
bin8_mmc["cell_id"] = bin8_mmc["cell_id"].astype(str)

# =========================
# 2. 定义比较函数
# =========================
def compare_unique_types(df1, df2, col, name1="our", name2="bin8"):
    s1 = set(df1[col].dropna().astype(str).unique())
    s2 = set(df2[col].dropna().astype(str).unique())

    only_in_df1 = sorted(s1 - s2)
    only_in_df2 = sorted(s2 - s1)
    common = sorted(s1 & s2)

    print(f"\n================ {col} ================")
    print(f"{name1} 中类型数: {len(s1)}")
    print(f"{name2} 中类型数: {len(s2)}")
    print(f"共同类型数: {len(common)}")
    print(f"仅在 {name1} 中出现、但 {name2} 中没有的类型数: {len(only_in_df1)}")
    print(f"仅在 {name2} 中出现、但 {name1} 中没有的类型数: {len(only_in_df2)}")

    if len(only_in_df1) > 0:
        print(f"\n仅在 {name1} 中出现的 {col}:")
        for x in only_in_df1:
            print(f"  {x}")
    else:
        print(f"\n没有发现仅在 {name1} 中出现的 {col}")

    return only_in_df1, only_in_df2, common

# =========================
# 3. 分别比较四个层级
# =========================
levels = ["class_name", "subclass_name", "supertype_name", "cluster_name"]
results = {}

for level in levels:
    only_our, only_bin8, common = compare_unique_types(
        our_mmc, bin8_mmc, level, name1="our_seg", name2="bin8"
    )
    results[level] = {
        "only_in_our": only_our,
        "only_in_bin8": only_bin8,
        "common": common
    }

# =========================
# 4. 额外输出：only_in_our 的数量统计
# =========================
for level in levels:
    print(f"\n######## {level}: 仅在 our_seg 中出现的类型及其细胞数 ########")
    only_types = results[level]["only_in_our"]

    if len(only_types) == 0:
        print("无")
        continue

    counts = our_mmc[level].value_counts()
    for t in only_types:
        print(f"{t}: {counts.get(t, 0)}")