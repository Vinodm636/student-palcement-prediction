"""
Student Placement Prediction System — Engineering College
CO2 | L5
Full pipeline: preprocessing → feature engineering → 7 classifiers → evaluation → recommendations
"""

from flask import Flask, render_template, request, redirect, session, flash, jsonify
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing     import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                       f1_score, confusion_matrix, roc_curve, auc,
                                       classification_report)
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestClassifier, GradientBoostingClassifier,
                                       AdaBoostClassifier)
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.naive_bayes       import GaussianNB

from users import users

app = Flask(__name__)
app.secret_key = "engg_placement_2024"

CSV       = "placement_data.csv"
CHART_DIR = "static/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Raw feature columns (before engineering) ──────────────────────────────────
RAW_NUMERIC = [
    "tenth_percentage","twelfth_percentage","cgpa","backlogs",
    "aptitude_score","logical_score","communication_score","english_score",
    "programming_skill","dsa_skill","database_skill","web_skill","ml_skill","cloud_skill",
    "num_projects","num_certifications","hackathons","paper_publications",
    "internship_done","internship_months","internship_stipend",
    "leadership_score","teamwork_score","gd_score",
]

# ── Engineered feature columns (added after feature engineering) ──────────────
ENGINEERED = [
    "academic_score",       # weighted combo of cgpa, 10th, 12th
    "aptitude_composite",   # aptitude + logical + english
    "tech_score",           # weighted tech skills
    "activity_score",       # projects + certs + hackathons + papers
    "soft_skill_score",     # leadership + teamwork + gd
    "internship_score",     # done × months × stipend-normalised
    "overall_readiness",    # final composite
]

# ── Final training features ────────────────────────────────────────────────────
FEATURES = RAW_NUMERIC + ENGINEERED

FEATURE_LABELS = {
    "tenth_percentage":    "10th %",
    "twelfth_percentage":  "12th %",
    "cgpa":                "CGPA",
    "backlogs":            "Backlogs",
    "aptitude_score":      "Aptitude",
    "logical_score":       "Logical Reasoning",
    "communication_score": "Communication",
    "english_score":       "English Score",
    "programming_skill":   "Programming",
    "dsa_skill":           "DSA",
    "database_skill":      "Database",
    "web_skill":           "Web Dev",
    "ml_skill":            "ML/AI",
    "cloud_skill":         "Cloud",
    "num_projects":        "Projects",
    "num_certifications":  "Certifications",
    "hackathons":          "Hackathons",
    "paper_publications":  "Publications",
    "internship_done":     "Internship Done",
    "internship_months":   "Internship Duration",
    "internship_stipend":  "Internship Stipend",
    "leadership_score":    "Leadership",
    "teamwork_score":      "Teamwork",
    "gd_score":            "GD Score",
    "academic_score":      "Academic Composite",
    "aptitude_composite":  "Aptitude Composite",
    "tech_score":          "Tech Composite",
    "activity_score":      "Activity Score",
    "soft_skill_score":    "Soft Skills Score",
    "internship_score":    "Internship Score",
    "overall_readiness":   "Overall Readiness",
}

SKILL_TIPS = {
    "cgpa":                "Improve CGPA above 7.5 — most companies have a CGPA cutoff.",
    "backlogs":            "Clear all backlogs immediately; they disqualify you in most shortlists.",
    "aptitude_score":      "Practice aptitude daily: R.S. Agarwal, IndiaBix, PrepInsta.",
    "logical_score":       "Solve logical reasoning puzzles and verbal ability daily.",
    "communication_score": "Join Toastmasters or college debate club; practice daily speaking.",
    "english_score":       "Read English newspapers; practice grammar and comprehension.",
    "programming_skill":   "Build projects in Python/Java/C++; solve 150+ LeetCode problems.",
    "dsa_skill":           "Master arrays, trees, graphs, DP — DSA is the #1 interview filter.",
    "database_skill":      "Learn SQL deeply (joins, indexes, procedures); practice on HackerRank.",
    "web_skill":           "Build 2–3 full-stack projects (React + Flask/Node) and host on GitHub.",
    "ml_skill":            "Complete Andrew Ng's ML course; build Kaggle competition projects.",
    "cloud_skill":         "Get AWS Cloud Practitioner or Azure Fundamentals certified (free tier).",
    "num_certifications":  "Earn 3+ industry certifications from Coursera, NPTEL, or Udemy.",
    "num_projects":        "Complete 4+ end-to-end projects — quality > quantity.",
    "hackathons":          "Participate in Smart India Hackathon, HackWithInfy, and campus hackathons.",
    "paper_publications":  "Publish a paper in an international conference or IEEE journal.",
    "internship_done":     "Apply for internships via LinkedIn, Internshala, and company portals.",
    "internship_months":   "Aim for 2–3 month paid internship for real-world experience.",
    "gd_score":            "Practice group discussion topics: business, tech, social — be structured.",
    "leadership_score":    "Lead a club, organize an event — demonstrate initiative on your resume.",
    "teamwork_score":      "Contribute to open-source projects or college team events.",
    "tech_score":          "Strengthen your overall technical foundation across all CS domains.",
    "academic_score":      "Focus on consistent performance across all semesters.",
    "aptitude_composite":  "Daily practice across aptitude, logical, and verbal sections.",
    "activity_score":      "Increase your extra-curricular portfolio: projects, certs, hackathons.",
    "soft_skill_score":    "Work on communication, leadership, and teamwork simultaneously.",
    "overall_readiness":   "Build a balanced profile — academics + skills + experience + activities.",
}

# ── Global model cache ─────────────────────────────────────────────────────────
_cache = {}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING + PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def load_raw():
    return pd.read_csv(CSV)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline:
    1. Duplicate removal
    2. Missing value imputation (median for numeric, most-frequent for categorical)
    3. Outlier detection (IQR)
    4. Type casting
    Returns cleaned df and a preprocessing report dict.
    """
    report = {}
    df = df.copy()

    # 1. Duplicates
    dups = df.duplicated().sum()
    df   = df.drop_duplicates()
    report["duplicates_removed"] = int(dups)

    # 2. Missing values — record before imputation
    missing_before = df[RAW_NUMERIC].isnull().sum().to_dict()
    report["missing_before"] = {k: int(v) for k, v in missing_before.items() if v > 0}

    # Impute numeric with median
    for col in RAW_NUMERIC:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    report["missing_after"] = int(df[[c for c in RAW_NUMERIC if c in df.columns]].isnull().sum().sum())

    # 3. Outlier capping (IQR, only for continuous scores)
    clip_cols = ["aptitude_score","logical_score","communication_score",
                 "english_score","gd_score"]
    outlier_count = 0
    for col in clip_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        out = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_count += int(out)
        df[col] = df[col].clip(lo, hi)
    report["outliers_capped"] = outlier_count

    # 4. Ensure correct types
    for col in ["backlogs","internship_done","internship_months",
                "num_projects","num_certifications","hackathons","paper_publications"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    report["final_rows"] = len(df)
    report["final_cols"] = len(df.columns)
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Creates 7 composite features from raw columns.
    Returns enriched df and an engineering report dict.
    """
    df = df.copy()

    # 1. Academic composite (0–100)
    df["academic_score"] = (
        df["cgpa"]               * 5.0 +   # /10 * 50 points
        df["tenth_percentage"]   * 0.25 +  # /100 * 25 points
        df["twelfth_percentage"] * 0.25    # /100 * 25 points
    ).clip(0, 100).round(2)

    # 2. Aptitude composite (0–100)
    df["aptitude_composite"] = (
        df["aptitude_score"]      * 0.40 +
        df["logical_score"]       * 0.35 +
        df["english_score"]       * 0.25
    ).clip(0, 100).round(2)

    # 3. Technical skill composite (0–10)
    df["tech_score"] = (
        df["programming_skill"]  * 0.30 +
        df["dsa_skill"]          * 0.25 +
        df["database_skill"]     * 0.15 +
        df["web_skill"]          * 0.12 +
        df["ml_skill"]           * 0.10 +
        df["cloud_skill"]        * 0.08
    ).clip(0, 10).round(2)

    # 4. Activity score (0–10)
    df["activity_score"] = (
        df["num_projects"]        * 0.8 +
        df["num_certifications"]  * 0.6 +
        df["hackathons"]          * 0.5 +
        df["paper_publications"]  * 1.2
    ).clip(0, 10).round(2)

    # 5. Soft skill composite (0–10)
    df["soft_skill_score"] = (
        df["communication_score"] * 0.04 +   # /100 * 4 points
        df["leadership_score"]    * 0.30 +
        df["teamwork_score"]      * 0.30 +
        df["gd_score"]            * 0.36
    ).clip(0, 10).round(2)

    # 6. Internship score (0–10)
    stip_norm = df["internship_stipend"] / (df["internship_stipend"].max() + 1)
    df["internship_score"] = (
        df["internship_done"]   * 4.0 +
        df["internship_months"] * 0.8 +
        stip_norm               * 2.0
    ).clip(0, 10).round(2)

    # 7. Overall readiness (0–100) — master composite
    df["overall_readiness"] = (
        df["academic_score"]    * 0.28 +
        df["aptitude_composite"]* 0.20 +
        df["tech_score"]        * 10 * 0.25 +   # scale to 100
        df["activity_score"]    * 10 * 0.10 +
        df["soft_skill_score"]  * 10 * 0.08 +
        df["internship_score"]  * 10 * 0.09
    ).clip(0, 100).round(2)

    eng_report = {
        "features_created": ENGINEERED,
        "descriptions": {
            "academic_score":     "Weighted combination of CGPA×5, 10th%×0.25, 12th%×0.25",
            "aptitude_composite": "Aptitude×0.40 + Logical×0.35 + English×0.25",
            "tech_score":         "Weighted avg of 6 technical skills (prog, DSA, DB, web, ML, cloud)",
            "activity_score":     "Projects×0.8 + Certs×0.6 + Hackathons×0.5 + Papers×1.2",
            "soft_skill_score":   "Communication + Leadership + Teamwork + GD Score composite",
            "internship_score":   "Internship done×4 + Months×0.8 + Stipend (normalised)×2",
            "overall_readiness":  "Master composite: academic 28% + aptitude 20% + tech 25% + activity 10% + soft 8% + internship 9%",
        }
    }
    return df, eng_report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN ALL CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
def get_classifiers():
    return {
        "Logistic Regression":  LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        "SVM":                  SVC(probability=True, kernel="rbf", C=1.5, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=9, metric="minkowski"),
        "Naive Bayes":          GaussianNB(),
    }


def train_all(force=False):
    global _cache
    if _cache and not force:
        return _cache

    # ── Load + preprocess + engineer ──────────────────────────────────────────
    raw_df                   = load_raw()
    clean_df, pre_report     = preprocess(raw_df)
    eng_df,   eng_report     = feature_engineering(clean_df)

    X = eng_df[FEATURES].copy()
    # Final safety net: fill any remaining NaN with column median
    X = X.fillna(X.median())
    y = eng_df["placed"]

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, clf in get_classifiers().items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        cv_acc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        cv_f1  = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1")
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        results[name] = {
            "model":     clf,
            "accuracy":  round(accuracy_score(y_te, y_pred)    * 100, 2),
            "precision": round(precision_score(y_te, y_pred)   * 100, 2),
            "recall":    round(recall_score(y_te, y_pred)      * 100, 2),
            "f1":        round(f1_score(y_te, y_pred)          * 100, 2),
            "roc_auc":   round(auc(fpr, tpr)                   * 100, 2),
            "cv_acc":    round(cv_acc.mean()                   * 100, 2),
            "cv_acc_std":round(cv_acc.std()                    * 100, 2),
            "cv_f1":     round(cv_f1.mean()                    * 100, 2),
            "cv_f1_std": round(cv_f1.std()                     * 100, 2),
            "cm":        confusion_matrix(y_te, y_pred).tolist(),
            "fpr":       fpr.tolist(),
            "tpr":       tpr.tolist(),
            "report":    classification_report(y_te, y_pred, output_dict=True),
        }

    # Feature importance from Random Forest
    rf        = results["Random Forest"]["model"]
    raw_imp   = dict(zip(FEATURES, rf.feature_importances_))
    importances = dict(sorted(raw_imp.items(), key=lambda x: x[1], reverse=True))

    # Best model by F1
    best_name = max(results, key=lambda k: results[k]["f1"])

    # EDA stats
    eda = compute_eda(eng_df)

    _cache = {
        "results":     results,
        "scaler":      scaler,
        "importances": importances,
        "best_model":  best_name,
        "pre_report":  pre_report,
        "eng_report":  eng_report,
        "eng_df":      eng_df,
        "raw_df":      raw_df,
        "X_te":        X_te,
        "y_te":        y_te,
        "eda":         eda,
    }
    return _cache


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
def compute_eda(df: pd.DataFrame) -> dict:
    placed   = df[df["placed"] == 1]
    nplaced  = df[df["placed"] == 0]

    # Key numeric stats for placed vs not placed
    key_cols = ["cgpa","aptitude_score","programming_skill","dsa_skill",
                "communication_score","num_certifications","internship_done",
                "overall_readiness"]
    compare = {}
    for col in key_cols:
        compare[col] = {
            "placed_mean":   round(float(placed[col].mean()),  2),
            "nplaced_mean":  round(float(nplaced[col].mean()), 2),
        }

    branch_stats = (df.groupby("branch")["placed"]
                      .agg(["sum","count"])
                      .rename(columns={"sum":"placed","count":"total"}))
    branch_stats["rate"] = (branch_stats["placed"] / branch_stats["total"] * 100).round(1)
    branch_stats = branch_stats.sort_values("rate", ascending=False)

    return {
        "total":        len(df),
        "placed":       int(df["placed"].sum()),
        "not_placed":   int((df["placed"] == 0).sum()),
        "placement_rate": round(df["placed"].mean() * 100, 1),
        "avg_cgpa":     round(float(df["cgpa"].mean()), 2),
        "avg_package":  round(float(placed["package_lpa"].mean()), 2),
        "max_package":  round(float(placed["package_lpa"].max()), 2),
        "avg_cgpa_placed":  round(float(placed["cgpa"].mean()),  2),
        "avg_cgpa_nplaced": round(float(nplaced["cgpa"].mean()), 2),
        "branches":     branch_stats.to_dict("index"),
        "compare":      compare,
        "batch_dist":   df["batch"].value_counts().to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SKILL RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
BENCHMARKS = {
    "cgpa":                7.5,  "tenth_percentage":  75,  "twelfth_percentage": 75,
    "backlogs":            0,    "aptitude_score":    72,  "logical_score":      68,
    "communication_score": 70,   "english_score":     70,  "gd_score":           7,
    "programming_skill":   7.5,  "dsa_skill":         7.0, "database_skill":     6.5,
    "web_skill":           6.0,  "ml_skill":          5.5, "cloud_skill":        5.0,
    "num_projects":        4,    "num_certifications": 3,  "hackathons":         2,
    "paper_publications":  1,    "internship_done":   1,   "internship_months":  3,
    "leadership_score":    6.5,  "teamwork_score":    7.0,
    "tech_score":          7.0,  "aptitude_composite": 70, "academic_score":     75,
    "activity_score":      5.0,  "soft_skill_score":  6.5, "overall_readiness":  65,
}

def recommend(inp: dict, importances: dict) -> list:
    recs = []
    for feat, imp in importances.items():
        if feat not in BENCHMARKS:
            continue
        cur   = float(inp.get(feat, 0))
        bench = BENCHMARKS[feat]
        gap   = (cur - bench) if feat == "backlogs" else (bench - cur)
        if gap <= 0:
            continue
        priority = "High" if imp > 0.06 else "Medium" if imp > 0.025 else "Low"
        recs.append({
            "feature":   feat,
            "label":     FEATURE_LABELS.get(feat, feat),
            "current":   round(cur,   2),
            "benchmark": bench,
            "gap":       round(gap,   2),
            "importance":round(imp * 100, 1),
            "tip":       SKILL_TIPS.get(feat, "Work on improving this skill."),
            "priority":  priority,
            "score":     imp * gap,
        })
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:9]


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DARK    = "#0f172a"
CARD    = "#1e293b"
BORDER  = "#334155"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
PALETTE = ["#6366f1","#34d399","#f472b6","#fb923c","#60a5fa","#a78bfa","#fbbf24","#22d3ee"]


def _ax(ax, title=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linewidth=0.4, linestyle="--", alpha=0.6)
    return ax


def save_charts(cache):
    results    = cache["results"]
    importances= cache["importances"]
    eng_df     = cache["eng_df"]

    # 1. Model comparison grouped bar
    names   = list(results.keys())
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    mlabels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    x = np.arange(len(names))
    w = 0.14

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(DARK)
    _ax(ax, "Model Comparison — All Metrics (%)")
    for i, (m, lbl) in enumerate(zip(metrics, mlabels)):
        vals = [results[n][m] for n in names]
        ax.bar(x + i*w, vals, w, label=lbl, color=PALETTE[i], alpha=0.92, edgecolor=DARK, linewidth=0.5)
        for xi, v in zip(x + i*w, vals):
            ax.text(xi, v+0.3, f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=6.5)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels([n.replace(" ","\n") for n in names], color=MUTED, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)", color=MUTED)
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK)
    _ax(ax, "ROC Curves — All Models")
    for i, (name, r) in enumerate(results.items()):
        ax.plot(r["fpr"], r["tpr"], color=PALETTE[i], lw=2,
                label=f"{name} (AUC={r['roc_auc']}%)")
    ax.plot([0,1],[0,1], "w--", lw=1, alpha=0.35, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3. Feature importance (top 15)
    top = dict(list(importances.items())[:15])
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK)
    _ax(ax, "Top Feature Importances (Random Forest)")
    labels = [FEATURE_LABELS.get(k,k) for k in top][::-1]
    vals   = [v*100 for v in list(top.values())][::-1]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    ax.barh(labels, vals, color=colors[::-1], edgecolor=DARK, linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(v+0.1, i, f"{v:.1f}%", va="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Importance (%)")
    ax.set_yticklabels(labels, color=MUTED, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4. Placement distribution charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(DARK)

    # 4a. Donut
    ax0 = axes[0]
    ax0.set_facecolor(DARK)
    placed_n = int(eng_df["placed"].sum())
    not_n    = len(eng_df) - placed_n
    wp = dict(width=0.5, edgecolor=DARK, linewidth=2)
    ax0.pie([not_n, placed_n], labels=["Not Placed","Placed"],
            colors=["#f87171","#34d399"], autopct="%1.1f%%",
            pctdistance=0.75, wedgeprops=wp,
            textprops={"color":TEXT,"fontsize":10})
    ax0.set_title("Placement Rate", color=TEXT, fontsize=11, fontweight="bold")

    # 4b. CGPA histogram
    ax1 = axes[1]; _ax(ax1, "CGPA Distribution")
    ax1.hist(eng_df[eng_df["placed"]==1]["cgpa"],  bins=25, alpha=0.7, color="#34d399",
             label="Placed",     edgecolor=DARK)
    ax1.hist(eng_df[eng_df["placed"]==0]["cgpa"],  bins=25, alpha=0.7, color="#f87171",
             label="Not Placed", edgecolor=DARK)
    ax1.set_xlabel("CGPA"); ax1.set_ylabel("Count")
    ax1.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)

    # 4c. Branch placement rate
    ax2 = axes[2]; _ax(ax2, "Branch-wise Placement Rate (%)")
    bp = eng_df.groupby("branch")["placed"].mean().sort_values(ascending=False)*100
    ax2.bar(bp.index, bp.values,
            color=[PALETTE[i%len(PALETTE)] for i in range(len(bp))],
            edgecolor=DARK, linewidth=0.5)
    ax2.set_ylabel("Placement %")
    ax2.set_xticklabels(bp.index, rotation=30, color=MUTED, fontsize=8)
    for i, v in enumerate(bp.values):
        ax2.text(i, v+0.5, f"{v:.0f}%", ha="center", color=TEXT, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/placement_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Confusion matrix per model
    for name, r in results.items():
        cm   = np.array(r["cm"])
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(DARK)
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
                    xticklabels=["Not Placed","Placed"],
                    yticklabels=["Not Placed","Placed"],
                    ax=ax, linewidths=1, linecolor=DARK,
                    annot_kws={"color":"white","fontsize":13,"fontweight":"bold"})
        ax.set_title(f"Confusion Matrix — {name}", color=TEXT, fontsize=11, pad=10)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual",    color=MUTED)
        ax.tick_params(colors=MUTED)
        plt.tight_layout()
        fname = name.replace(" ","_").lower()
        plt.savefig(f"{CHART_DIR}/cm_{fname}.png", dpi=110, bbox_inches="tight")
        plt.close()

    # 6. Preprocessing before/after missing values bar
    pre = cache["pre_report"]
    if pre.get("missing_before"):
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor(DARK)
        _ax(ax, "Missing Values Before Imputation")
        cols = list(pre["missing_before"].keys())
        vals = list(pre["missing_before"].values())
        ax.barh(cols, vals, color="#f87171", edgecolor=DARK)
        ax.set_xlabel("Missing Count")
        plt.tight_layout()
        plt.savefig(f"{CHART_DIR}/missing_values.png", dpi=110, bbox_inches="tight")
        plt.close()

    # 7. Placed vs Not — feature means comparison
    key = ["cgpa","aptitude_score","programming_skill","dsa_skill",
           "communication_score","overall_readiness"]
    placed_m  = [eng_df[eng_df["placed"]==1][c].mean() for c in key]
    nplaced_m = [eng_df[eng_df["placed"]==0][c].mean() for c in key]
    xlabels   = [FEATURE_LABELS.get(c,c) for c in key]
    x = np.arange(len(xlabels))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK)
    _ax(ax, "Avg Feature Values — Placed vs Not Placed")
    ax.bar(x-0.2, placed_m,  0.38, label="Placed",     color="#34d399", edgecolor=DARK)
    ax.bar(x+0.2, nplaced_m, 0.38, label="Not Placed", color="#f87171", edgecolor=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color=MUTED, fontsize=9, rotation=15)
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/feature_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        r = request.form["role"]
        if u in users and users[u]["password"] == p and users[u]["role"] == r:
            session["user"] = u
            session["role"] = r
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials. Please try again.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    df     = pd.read_csv(CSV)
    search = request.args.get("search","").strip().lower()
    branch = request.args.get("branch","").strip()

    disp = df.copy()
    if search:
        disp = disp[disp["name"].str.lower().str.contains(search) |
                    disp["usn"].str.lower().str.contains(search)]
    if branch:
        disp = disp[disp["branch"] == branch]

    total    = len(df)
    placed   = int(df["placed"].sum())
    rate     = round(placed/total*100, 1)
    avg_cgpa = round(df["cgpa"].mean(), 2)
    avg_pkg  = round(df[df["placed"]==1]["package_lpa"].mean(), 2)
    branches = sorted(df["branch"].unique().tolist())

    bp = df.groupby("branch")["placed"].agg(["sum","count"])
    bp["rate"] = (bp["sum"]/bp["count"]*100).round(1)

    return render_template("dashboard.html",
        df=disp, role=session["role"],
        total=total, placed=placed, not_placed=total-placed,
        rate=rate, avg_cgpa=avg_cgpa, avg_pkg=avg_pkg,
        branches=branches, search=search, branch_filter=branch,
        branch_labels=json.dumps(bp.index.tolist()),
        branch_rates=json.dumps(bp["rate"].tolist()),
        branch_placed=json.dumps(bp["sum"].tolist()),
        branch_total=json.dumps(bp["count"].tolist()),
    )


# ── Preprocessing page ────────────────────────────────────────────────────────
@app.route("/preprocessing")
def preprocessing():
    if "user" not in session:
        return redirect("/")

    cache = train_all()
    pre   = cache["pre_report"]
    eng   = cache["eng_report"]
    df    = cache["raw_df"]
    clean = cache["eng_df"]

    # Descriptive stats on raw numeric cols (before engineering)
    stats_raw = df[["cgpa","aptitude_score","programming_skill","communication_score",
                    "internship_done","backlogs"]].describe().round(2).to_dict()

    # Correlation matrix data (top 10 features with placed)
    corr = clean[FEATURES + ["placed"]].corr()["placed"].drop("placed")
    corr = corr.abs().sort_values(ascending=False).head(12)
    corr_labels = json.dumps([FEATURE_LABELS.get(k,k) for k in corr.index])
    corr_values = json.dumps(corr.values.round(3).tolist())

    return render_template("preprocessing.html",
        pre=pre, eng=eng,
        raw_rows=len(df), clean_rows=pre["final_rows"],
        raw_cols=len(df.columns), clean_cols=pre["final_cols"],
        stats=stats_raw,
        corr_labels=corr_labels, corr_values=corr_values,
        missing_before=pre.get("missing_before",{}),
        eng_features=eng["features_created"],
        eng_descriptions=eng["descriptions"],
    )


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET","POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    results_out = None
    recs        = None
    form        = {}
    ensemble    = None
    avg_prob    = None

    if request.method == "POST":
        cache = train_all()

        try:
            raw_inp = {f: float(request.form.get(f, 0)) for f in RAW_NUMERIC}
        except ValueError:
            flash("Please enter valid numbers for all fields.", "danger")
            return render_template("predict.html",
                results=None, recs=None, form={},
                ensemble=None, avg_prob=None, best_model="")

        # Apply same feature engineering to input
        inp_df = pd.DataFrame([raw_inp])
        inp_df, _ = feature_engineering(inp_df)

        # Ensure all feature columns exist (fill 0 for any missing)
        for col in FEATURES:
            if col not in inp_df.columns:
                inp_df[col] = 0.0

        inp_arr = inp_df[FEATURES].values
        X_s     = cache["scaler"].transform(inp_arr)

        results_out = {}
        for name, r in cache["results"].items():
            pred = int(r["model"].predict(X_s)[0])
            prob = round(float(r["model"].predict_proba(X_s)[0][1]) * 100, 1)
            results_out[name] = {
                "pred": pred, "prob": prob,
                "label": "Placed" if pred == 1 else "Not Placed",
                "f1":  r["f1"], "acc": r["accuracy"],
            }

        votes    = sum(1 for r in results_out.values() if r["pred"] == 1)
        ensemble = 1 if votes > len(results_out)/2 else 0
        avg_prob = round(np.mean([r["prob"] for r in results_out.values()]), 1)

        # Composite inp for recommendations
        full_inp = {**raw_inp, **inp_df[ENGINEERED].iloc[0].to_dict()}
        recs     = recommend(full_inp, cache["importances"])
        form     = raw_inp

        return render_template("predict.html",
            results=results_out, recs=recs, form=form,
            ensemble=ensemble, avg_prob=avg_prob,
            votes=votes, total_models=len(results_out),
            best_model=cache["best_model"])

    return render_template("predict.html",
        results=None, recs=None, form=form,
        ensemble=None, avg_prob=None,
        best_model=train_all().get("best_model",""))


# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/analytics")
def analytics():
    if "user" not in session:
        return redirect("/")

    cache = train_all()
    save_charts(cache)

    metrics_rows = sorted([
        {"name": n, **{k: r[k] for k in
                       ["accuracy","precision","recall","f1","roc_auc","cv_acc","cv_acc_std","cv_f1"]}}
        for n, r in cache["results"].items()
    ], key=lambda x: x["f1"], reverse=True)

    top_imp   = dict(list(cache["importances"].items())[:12])
    imp_labels= json.dumps([FEATURE_LABELS.get(k,k) for k in top_imp])
    imp_values= json.dumps([round(v*100,2) for v in top_imp.values()])

    roc_ds = []
    for i,(name,r) in enumerate(cache["results"].items()):
        step = max(1, len(r["fpr"])//60)
        roc_ds.append({
            "label": f"{name} ({r['roc_auc']}%)",
            "data": [{"x":round(x,3),"y":round(y,3)}
                     for x,y in zip(r["fpr"][::step], r["tpr"][::step])],
            "borderColor": PALETTE[i%len(PALETTE)],
            "fill": False, "tension": 0.3, "pointRadius": 0,
        })

    best = cache["best_model"]
    cm   = cache["results"][best]["cm"]

    return render_template("analytics.html",
        metrics=metrics_rows, best_model=best,
        imp_labels=imp_labels, imp_values=imp_values,
        roc_datasets=json.dumps(roc_ds),
        best_cm=cm, best_cm_model=best,
        total_features=len(FEATURES), dataset_size=len(cache["eng_df"]),
        eda=cache["eda"])


# ── Preprocessing visual page ─────────────────────────────────────────────────
# (already defined above as /preprocessing)


# ── Model Justify ─────────────────────────────────────────────────────────────
@app.route("/justify")
def justify():
    if "user" not in session:
        return redirect("/")

    cache   = train_all()
    results = cache["results"]
    best    = cache["best_model"]
    br      = results[best]

    # Rank models by each metric
    def rank(metric):
        return sorted(results.items(), key=lambda x: x[1][metric], reverse=True)

    ranks = {
        "f1":       rank("f1"),
        "accuracy": rank("accuracy"),
        "roc_auc":  rank("roc_auc"),
        "recall":   rank("recall"),
    }

    # Justification text per model type
    model_notes = {
        "Logistic Regression":  "Linear model — fast, interpretable, works well when classes are linearly separable. Good baseline.",
        "Decision Tree":        "Non-linear, highly interpretable tree structure. Prone to overfitting without pruning.",
        "Random Forest":        "Ensemble of 200 trees — reduces variance, handles non-linearity, robust to noise.",
        "Gradient Boosting":    "Sequential boosting — high accuracy by correcting prior errors. Excellent for tabular data.",
        "AdaBoost":             "Focuses on hard samples iteratively. Works well with weak learners.",
        "SVM":                  "Effective in high dimensions with an RBF kernel. Good generalisation but slower on large data.",
        "K-Nearest Neighbors":  "Instance-based learning — no training phase but slow at inference and sensitive to scale.",
        "Naive Bayes":          "Assumes feature independence — very fast but often sacrifices accuracy on correlated features.",
    }

    # Build comparison JSON for radar chart
    best_metrics = {
        "Accuracy":  br["accuracy"],
        "Precision": br["precision"],
        "Recall":    br["recall"],
        "F1":        br["f1"],
        "ROC-AUC":   br["roc_auc"],
        "CV-F1":     br["cv_f1"],
    }

    return render_template("justify.html",
        results=results, best=best, br=br,
        ranks=ranks, model_notes=model_notes,
        best_metrics=json.dumps(best_metrics),
        all_names=json.dumps(list(results.keys())),
        all_f1=json.dumps([results[n]["f1"] for n in results]),
        all_auc=json.dumps([results[n]["roc_auc"] for n in results]),
        all_acc=json.dumps([results[n]["accuracy"] for n in results]),
    )


# ── Student Profile ───────────────────────────────────────────────────────────
@app.route("/student/<usn>")
def student_profile(usn):
    if "user" not in session:
        return redirect("/")

    df  = pd.read_csv(CSV)
    row = df[df["usn"] == usn]
    if row.empty:
        flash("Student not found.", "danger")
        return redirect("/dashboard")

    student = row.iloc[0].to_dict()
    cache   = train_all()

    raw_inp = {f: float(student.get(f, 0)) for f in RAW_NUMERIC}
    inp_df  = pd.DataFrame([raw_inp])
    inp_df, _ = feature_engineering(inp_df)
    for col in FEATURES:
        if col not in inp_df.columns:
            inp_df[col] = 0.0

    X_s  = cache["scaler"].transform(inp_df[FEATURES].values)
    best_m = cache["results"][cache["best_model"]]["model"]
    pred   = int(best_m.predict(X_s)[0])
    prob   = round(float(best_m.predict_proba(X_s)[0][1]) * 100, 1)

    full_inp = {**raw_inp, **inp_df[ENGINEERED].iloc[0].to_dict()}
    recs = recommend(full_inp, cache["importances"])

    skill_labels = json.dumps(["Programming","DSA","Database","Web Dev","ML/AI","Cloud"])
    skill_vals   = json.dumps([
        round(float(student.get("programming_skill", 0))*10, 1),
        round(float(student.get("dsa_skill", 0))*10, 1),
        round(float(student.get("database_skill", 0))*10, 1),
        round(float(student.get("web_skill", 0))*10, 1),
        round(float(student.get("ml_skill", 0))*10, 1),
        round(float(student.get("cloud_skill", 0))*10, 1),
    ])

    return render_template("profile.html",
        student=student, pred=pred, prob=prob,
        recs=recs, best_model=cache["best_model"],
        skill_labels=skill_labels, skill_vals=skill_vals)


# ── Leaderboard ───────────────────────────────────────────────────────────────
@app.route("/leaderboard")
def leaderboard():
    if "user" not in session:
        return redirect("/")

    df     = pd.read_csv(CSV)
    branch = request.args.get("branch","")
    status = request.args.get("status","")

    disp = df.copy()
    if branch: disp = disp[disp["branch"] == branch]
    if status == "placed":     disp = disp[disp["placed"] == 1]
    elif status == "not_placed": disp = disp[disp["placed"] == 0]

    disp = disp.sort_values("cgpa", ascending=False).reset_index(drop=True)
    disp.index += 1

    return render_template("leaderboard.html",
        df=disp, branches=sorted(df["branch"].unique().tolist()),
        branch_filter=branch, status_filter=status)


# ── Model detail ──────────────────────────────────────────────────────────────
@app.route("/model/<model_name>")
def model_detail(model_name):
    if "user" not in session:
        return redirect("/")

    cache = train_all()
    name  = model_name.replace("_", " ")
    if name not in cache["results"]:
        flash("Model not found.", "danger")
        return redirect("/analytics")

    r = cache["results"][name]
    step = max(1, len(r["fpr"]) // 60)
    roc_data = json.dumps([
        {"x": round(x,3), "y": round(y,3)}
        for x,y in zip(r["fpr"][::step], r["tpr"][::step])
    ])
    cr = r["report"]
    cr_rows = [
        {"label":"Not Placed","precision":round(cr["0"]["precision"]*100,1),
         "recall":round(cr["0"]["recall"]*100,1),"f1":round(cr["0"]["f1-score"]*100,1),
         "support":cr["0"]["support"]},
        {"label":"Placed","precision":round(cr["1"]["precision"]*100,1),
         "recall":round(cr["1"]["recall"]*100,1),"f1":round(cr["1"]["f1-score"]*100,1),
         "support":cr["1"]["support"]},
    ]
    fname = name.replace(" ","_").lower()
    return render_template("model_detail.html",
        model_name=name, r=r, cr_rows=cr_rows, roc_data=roc_data,
        cm_img=f"cm_{fname}.png",
        is_best=(name == cache["best_model"]))


# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/students")
def api_students():
    df = pd.read_csv(CSV)
    q  = request.args.get("q","").lower()
    out = df[["name","usn","branch","cgpa","placed"]]
    if q:
        out = out[out["name"].str.lower().str.contains(q) |
                  out["usn"].str.lower().str.contains(q)]
    return jsonify(out.head(8).to_dict("records"))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Pre-training all models...")
    train_all(force=True)
    print("Ready at http://127.0.0.1:5000")
    app.run(debug=True)
