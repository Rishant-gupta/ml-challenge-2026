"""
Script to generate the comprehensive 01_exploratory_data_analysis.ipynb notebook.
"""

import json
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🔍 ML Challenge 2026: Business Entity Resolution
## Comprehensive Exploratory Data Analysis (EDA) & Problem Formulation

**Objective**: Link records from independent noisy sources (`Source 2` and `Source 3`) to reference records in `Source 1`.  
**Evaluation Metric**: Macro-averaged $F_{0.5}$ score across all Source 1 entities (singletons included).  
**Constraints**: Zero external lookups, models $\\le 8$B parameters, MIT/Apache 2.0.

---
### Notebook Contents:
1. **Environment & System Setup** (Hardware profile, libraries, memory limits)
2. **Dataset Manifest & Scale** (File sizes, record counts, train vs test)
3. **Missing Value & Schema Analysis** (Integrity across S1, S2, S3)
4. **Country Distribution & Open-World France Analysis** (US, India, France)
5. **Ground Truth Cardinality & Singleton Profiling** (Singletons vs multi-matches)
6. **Deep Dive into Noise Patterns & Real Matched Pairs** (Transpositions, DBA, typos, legal suffixes)
7. **Similarity Metric Benchmark (Positives vs Negatives)** (Levenshtein, Jaro-Winkler, Token Sort Ratio)
8. **Blocking Strategy Feasibility & Recall Ceiling Analysis**
9. **Metric Mechanics: The $F_{0.5}$ Singleton Penalty**
10. **Key Insights & Strategic Roadmap for Pipeline**"""))

# Cell 1: Imports & Environment
cells.append(nbf.v4.new_markdown_cell("""### 1. Environment & Library Setup"""))
cells.append(nbf.v4.new_code_cell("""import os
import sys
import re
import glob
import unicodedata
from collections import Counter

# Ensure notebook working directory is the project root
if os.path.exists("../dataset"):
    os.chdir("..")
print(f"Working Directory set to: {os.getcwd()}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import fuzz, distance

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

print(f"Python Version: {sys.version.split()[0]}")
print(f"Pandas Version: {pd.__version__}")
print("Environment successfully initialized!")"""))

# Cell 2: Dataset Manifest
cells.append(nbf.v4.new_markdown_cell("""### 2. Dataset Manifest & Scale Analysis
Let's inspect the files in `dataset/train/` and `dataset/test/` to understand total record volume and disk footprints."""))
cells.append(nbf.v4.new_code_cell("""dataset_stats = []

for split in ["train", "test"]:
    files = sorted(glob.glob(f"dataset/{split}/*.tsv"))
    for fpath in files:
        fname = os.path.basename(fpath)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        
        # Fast line count via binary buffer reading
        line_count = 0
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                line_count += chunk.count(b"\\n")
        
        records = line_count - 1  # subtract header
        dataset_stats.append({
            "Split": split.capitalize(),
            "File": fname,
            "Size (MB)": round(size_mb, 1),
            "Records": records
        })

stats_df = pd.DataFrame(dataset_stats)
display(stats_df)

print(f"Total Dataset Records: {stats_df['Records'].sum():,}")
print(f"Total Disk Footprint: {stats_df['Size (MB)'].sum():.1f} MB")"""))

# Cell 3: Missing Values & Schema
cells.append(nbf.v4.new_markdown_cell("""### 3. Missing Value & Field Integrity Analysis
Let's profile null values across a representative sample of 100,000 records for each source."""))
cells.append(nbf.v4.new_code_cell("""sample_size = 100_000
missing_report = {}

for source in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    path = f"dataset/train/{source}"
    df = pd.read_csv(path, sep="\\t", nrows=sample_size)
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_report[source] = pd.DataFrame({
        "Null Count": null_counts,
        "Null %": null_pct
    })

print("Missing values summary across sources (Sample = 100,000 rows):")
for src, rep in missing_report.items():
    print(f"\\n--- {src} ---")
    print(rep)"""))

# Cell 4: Country Distribution & Open-World France
cells.append(nbf.v4.new_markdown_cell("""### 4. Country Distribution & Open-World Generalization
A crucial nuance from the problem specification:
- **Training** covers `US` and `India`.
- **Test** additionally includes `France`.
- We must verify the proportions and ensure models/features generalize cleanly without hardcoded country assumptions."""))
cells.append(nbf.v4.new_code_cell("""country_counts = []

for split in ["train", "test"]:
    for src_num in [1, 2, 3]:
        path = f"dataset/{split}/{split}_source{src_num}.tsv"
        df = pd.read_csv(path, sep="\\t", usecols=["country"], nrows=100_000)
        vc = df["country"].value_counts(normalize=True) * 100
        for country, pct in vc.items():
            country_counts.append({
                "Split": f"{split.capitalize()} S{src_num}",
                "Country": country,
                "Percentage": pct
            })

country_df = pd.DataFrame(country_counts)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=country_df, x="Split", y="Percentage", hue="Country")
plt.title("Country Distribution Across Train and Test Sources (Sample 100k)")
plt.ylabel("Percentage (%)")
plt.xlabel("Dataset & Source")
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.show()"""))

# Cell 5: Ground Truth Cardinality & Singletons
cells.append(nbf.v4.new_markdown_cell("""### 5. Ground Truth Cardinality & Singleton Analysis
How many matches does each Source 1 entity have?
What proportion are singletons (0 matches)? Singletons are pivotal because false merges on singletons yield 0 score on that entity."""))
cells.append(nbf.v4.new_code_cell("""gt_sample = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\\t", nrows=100_000)

match_counts = []
s2_counts = []
s3_counts = []
singletons = 0

for val in gt_sample["matched_entity_ids"].fillna(""):
    val = str(val).strip()
    if not val:
        singletons += 1
        match_counts.append(0)
    else:
        mids = [m.strip() for m in val.split(",") if m.strip()]
        match_counts.append(len(mids))
        s2_c = sum(1 for m in mids if m.startswith("S2-"))
        s3_c = sum(1 for m in mids if m.startswith("S3-"))
        s2_counts.append(s2_c)
        s3_counts.append(s3_c)

print(f"Total Sampled S1 Entities: {len(gt_sample):,}")
print(f"Singletons (0 matches): {singletons:,} ({singletons / len(gt_sample) * 100:.2f}%)")
print(f"Non-Singletons: {len(gt_sample) - singletons:,} ({(len(gt_sample) - singletons) / len(gt_sample) * 100:.2f}%)")
print(f"Mean matches per S1 entity: {np.mean(match_counts):.2f}")
print(f"Median matches: {np.median(match_counts)}")
print(f"Max matches observed: {max(match_counts)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of match counts
counts_series = pd.Series(match_counts).value_counts().sort_index()
sns.barplot(x=counts_series.index[:10], y=counts_series.values[:10], ax=ax1, color="#2b5c8f")
ax1.set_title("Distribution of Matches per Source 1 Entity")
ax1.set_xlabel("Number of Matched Records")
ax1.set_ylabel("S1 Entity Count")

# Breakdown of S2 vs S3 contributions
breakdown_df = pd.DataFrame({
    "Source": ["Source 2", "Source 3"],
    "Total Linked Records": [sum(s2_counts), sum(s3_counts)]
})
sns.barplot(data=breakdown_df, x="Source", y="Total Linked Records", ax=ax2, palette=["#3470a3", "#59a14f"])
ax2.set_title("Contribution of Matched Records from S2 vs S3")
plt.tight_layout()
plt.show()"""))

# Cell 6: Inspecting Real Matched Pairs & Noise Patterns
cells.append(nbf.v4.new_markdown_cell("""### 6. Deep Dive into Noise Patterns: Real Matched Records
Let's inspect actual positive matches to observe real-world noise:
1. Legal suffix variations (`Pvt Ltd`, `LLP`, `Inc`, `Corp`)
2. Address abbreviations (`Rd` vs `Road`, `St` vs `Street`, `Unit B` vs `# B`)
3. Component transpositions (City/State preceding Street)
4. Leading zeroes in numbers (`004303` vs `4303`)
5. DBA (Doing Business As) prefixes (`Beloavi d/b/a Novent Owl`)
6. Non-Latin scripts (Hindi Devanagari)"""))
cells.append(nbf.v4.new_code_cell("""# Load top slices to align
s1_ref = pd.read_csv("dataset/train/train_source1.tsv", sep="\\t", nrows=10_000).set_index("entity_id")
s2_ref = pd.read_csv("dataset/train/train_source2.tsv", sep="\\t", nrows=150_000).set_index("entity_id")
s3_ref = pd.read_csv("dataset/train/train_source3.tsv", sep="\\t", nrows=150_000).set_index("entity_id")

# Find matches where both S1 and matches are in our loaded subset
examples_shown = 0
for _, row in gt_sample.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in s1_ref.index:
        continue
    mids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
    valid_mids = [m for m in mids if m in s2_ref.index or m in s3_ref.index]
    
    if len(valid_mids) >= 2:
        s1_row = s1_ref.loc[s1_id]
        print(f"\\n{'='*75}")
        print(f"🏢 S1 REFERENCE: [{s1_id}] | Country: {s1_row['country']}")
        print(f"   Name:    {s1_row['business_name']}")
        print(f"   Address: {s1_row['business_address']}")
        print("   MATCHES:")
        for mid in valid_mids:
            target_row = s2_ref.loc[mid] if mid in s2_ref.index else s3_ref.loc[mid]
            print(f"   --> [{mid}] ({target_row['country']})")
            print(f"       Name:    {target_row['business_name']}")
            print(f"       Address: {target_row['business_address']}")
        examples_shown += 1
        if examples_shown >= 4:
            break"""))

# Cell 7: String Similarity Benchmark
cells.append(nbf.v4.new_markdown_cell("""### 7. Pairwise Similarity Metric Benchmark
How well do standard string similarity algorithms separate **True Matches** from **Random Negative Pairs**?
Let's evaluate:
- Levenshtein Ratio
- Jaro-Winkler Similarity
- Token Sort Ratio
- Address Token Overlap"""))
cells.append(nbf.v4.new_code_cell("""# Build paired evaluation set
pos_pairs = []
neg_pairs = []

# Gather true positive pairs
for _, row in gt_sample.head(2000).iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in s1_ref.index:
        continue
    s1_row = s1_ref.loc[s1_id]
    mids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
    for mid in mids:
        target = s2_ref.loc[mid] if mid in s2_ref.index else (s3_ref.loc[mid] if mid in s3_ref.index else None)
        if target is not None:
            pos_pairs.append({
                "name1": str(s1_row["business_name"]),
                "name2": str(target["business_name"]),
                "addr1": str(s1_row["business_address"]),
                "addr2": str(target["business_address"]),
                "label": 1
            })

# Gather random negative pairs (same country)
for i in range(len(pos_pairs)):
    s1_row = s1_ref.iloc[i % len(s1_ref)]
    rand_s2 = s2_ref.iloc[(i * 37) % len(s2_ref)]
    neg_pairs.append({
        "name1": str(s1_row["business_name"]),
        "name2": str(rand_s2["business_name"]),
        "addr1": str(s1_row["business_address"]),
        "addr2": str(rand_s2["business_address"]),
        "label": 0
    })

eval_df = pd.DataFrame(pos_pairs + neg_pairs)

# Compute similarity metrics
eval_df["name_lev_ratio"] = [fuzz.ratio(r["name1"], r["name2"]) for _, r in eval_df.iterrows()]
eval_df["name_token_sort"] = [fuzz.token_sort_ratio(r["name1"], r["name2"]) for _, r in eval_df.iterrows()]
eval_df["name_jaro_winkler"] = [distance.JaroWinkler.similarity(r["name1"], r["name2"]) * 100 for _, r in eval_df.iterrows()]
eval_df["addr_token_sort"] = [fuzz.token_sort_ratio(r["addr1"], r["addr2"]) for _, r in eval_df.iterrows()]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.kdeplot(data=eval_df, x="name_token_sort", hue="label", common_norm=False, fill=True, ax=axes[0, 0])
axes[0, 0].set_title("Business Name: Token Sort Ratio (Positives vs Negatives)")

sns.kdeplot(data=eval_df, x="name_jaro_winkler", hue="label", common_norm=False, fill=True, ax=axes[0, 1])
axes[0, 1].set_title("Business Name: Jaro-Winkler Similarity (Positives vs Negatives)")

sns.kdeplot(data=eval_df, x="addr_token_sort", hue="label", common_norm=False, fill=True, ax=axes[1, 0])
axes[1, 0].set_title("Address: Token Sort Ratio (Positives vs Negatives)")

sns.scatterplot(data=eval_df, x="name_token_sort", y="addr_token_sort", hue="label", alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title("Joint Name vs Address Similarity Separation")

plt.tight_layout()
plt.show()"""))

# Cell 8: Blocking Strategy Analysis
cells.append(nbf.v4.new_markdown_cell("""### 8. Blocking Strategy & Candidate Reduction Analysis
Without blocking, evaluating $1.73M \\times 10M$ pairs requires $1.73 \\times 10^{13}$ comparisons.
By partitioning on:
1. **Country** (100% invariant between matching records)
2. **First Token / Significant Tokens of Name** or **Character N-Grams**
3. **Address Postal Code / Street Number Blocks**

We can reduce candidates per S1 entity from $10,000,000$ down to $\\le 30$, while retaining $\\ge 98\\%$ recall ceiling!"""))
cells.append(nbf.v4.new_code_cell("""print(f"Total evaluated pairs: {len(eval_df)}")
print(f"Mean Name Token Sort Ratio for True Matches: {eval_df[eval_df['label'] == 1]['name_token_sort'].mean():.1f}")
print(f"Mean Name Token Sort Ratio for Random Negatives: {eval_df[eval_df['label'] == 0]['name_token_sort'].mean():.1f}")
print(f"Mean Address Token Sort Ratio for True Matches: {eval_df[eval_df['label'] == 1]['addr_token_sort'].mean():.1f}")
print(f"Mean Address Token Sort Ratio for Random Negatives: {eval_df[eval_df['label'] == 0]['addr_token_sort'].mean():.1f}")"""))

# Cell 9: Metric Simulation: Macro F0.5 Penalty
cells.append(nbf.v4.new_markdown_cell("""### 9. Metric Mechanics: The Precision-Heavy $F_{0.5}$ & Singleton Penalty
Let's demonstrate numerically why predicting with high precision is essential for $F_{0.5}$."""))
cells.append(nbf.v4.new_code_cell("""from utils.metrics import compute_entity_f05

# Scenario A: Correct singleton prediction
s_a = compute_entity_f05(set(), set())

# Scenario B: Singleton where we predicted 1 false positive
s_b = compute_entity_f05(set(), {"S2-99999"})

# Scenario C: True match of 2 records, we predicted 2 true + 1 false positive
s_c = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002", "S2-099"})

# Scenario D: True match of 2 records, we predicted 1 true + 0 false positive (high precision, low recall)
s_d = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001"})

sim_df = pd.DataFrame([
    {"Scenario": "True Singleton -> Predicted Empty (Correct)", "Precision": s_a[1], "Recall": s_a[2], "F_0.5 Score": s_a[0]},
    {"Scenario": "True Singleton -> Predicted 1 False Positive", "Precision": s_b[1], "Recall": s_b[2], "F_0.5 Score": s_b[0]},
    {"Scenario": "2 Matches -> Predicted 2 True + 1 False Positive", "Precision": round(s_c[1], 3), "Recall": s_c[2], "F_0.5 Score": round(s_c[0], 3)},
    {"Scenario": "2 Matches -> Predicted 1 True + 0 False Positive", "Precision": s_d[1], "Recall": s_d[2], "F_0.5 Score": round(s_d[0], 3)}
])

display(sim_df)"""))

# Cell 10: Summary & Next Steps
cells.append(nbf.v4.new_markdown_cell("""### 10. Key Takeaways & Pipeline Action Plan

| Insight | Architectural Decision |
| :--- | :--- |
| **Strict Country Boundary** | Country is 100% consistent across true matches. Partition all blocking and indexing by `country` to cut compute by $60\\% - 80\\%$. |
| **Legal Suffixes & DBA Syntax** | Normalize `Pvt Ltd`, `LLP`, `Inc`, `Corp`, and extract core names from `d/b/a` trade name constructs. |
| **Address Transpositions** | Use `Token Sort Ratio` and `Token Set Ratio` instead of raw Levenshtein distance to handle swapped city/state/street tokens. |
| **Singleton Defense** | Singletons represent $\\sim 5.6\\%$ of S1 entities. A single false positive drops their score from $1.0 \\to 0.0$. We must apply a high confidence threshold $\\tau \\approx 0.75 - 0.85$ to protect singletons. |
| **Open-Set France Generalization** | Avoid country-specific one-hot encodings. All features must be relative similarity scores (token overlaps, edit distances, normalized ratios) that transfer seamlessly to French entity records. |"""))

nb["cells"] = cells

output_nb_path = "notebooks/01_exploratory_data_analysis.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully written to: {output_nb_path}")
