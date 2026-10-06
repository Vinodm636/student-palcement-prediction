"""
Realistic Engineering College Placement Dataset
700 students across 8 BE/BTech branches, Karnataka-style USNs
"""
import pandas as pd
import numpy as np

np.random.seed(2024)
N = 700

# ── Names ─────────────────────────────────────────────────────────────────────
first = [
    "Aarav","Aditya","Akash","Amit","Ananya","Anjali","Arjun","Aryan","Ashok","Bhavana",
    "Charan","Chinmay","Deepa","Deepak","Divya","Ganesh","Geeta","Gopal","Harish","Harsha",
    "Hemanth","Indu","Jagadish","Jaya","Karthik","Kavitha","Kiran","Kriti","Kumar","Lakshmi",
    "Lavanya","Lokesh","Madhu","Mahesh","Mala","Meena","Mohan","Monisha","Mukesh","Naveen",
    "Neha","Nikhil","Nisha","Pallavi","Pooja","Prabhu","Prasad","Preethi","Priya","Rahul",
    "Rajesh","Rakesh","Ramesh","Ravi","Rekha","Rohan","Sahana","Sandeep","Sangeetha","Sanjay",
    "Santosh","Sarika","Seema","Shilpa","Shiva","Shreya","Siddharth","Sneha","Suhas","Sujan",
    "Suma","Sunitha","Suresh","Swathi","Tarun","Tejas","Uday","Uma","Varsha","Vijay",
    "Vikas","Vinod","Vishnu","Yamini","Yashoda","Zoya","Riya","Manish","Darshan","Keerthi",
]
last = [
    "Sharma","Verma","Patel","Nair","Reddy","Kumar","Singh","Rao","Joshi","Gupta",
    "Iyer","Pillai","Menon","Naidu","Hegde","Gowda","Patil","Shinde","More","Desai",
    "Bhat","Kamath","Shetty","Murthy","Kulkarni","Deshpande","Jain","Shah","Mehta","Tiwari",
]

# ── Branches ──────────────────────────────────────────────────────────────────
BRANCHES = {
    "CSE":   {"weight":0.22, "tech_boost":1.15, "placement_boost":1.10},
    "ISE":   {"weight":0.15, "tech_boost":1.10, "placement_boost":1.08},
    "ECE":   {"weight":0.18, "tech_boost":1.05, "placement_boost":1.05},
    "EEE":   {"weight":0.10, "tech_boost":0.95, "placement_boost":0.95},
    "MECH":  {"weight":0.12, "tech_boost":0.85, "placement_boost":0.88},
    "CIVIL": {"weight":0.08, "tech_boost":0.75, "placement_boost":0.80},
    "AIML":  {"weight":0.09, "tech_boost":1.20, "placement_boost":1.12},
    "DS":    {"weight":0.06, "tech_boost":1.18, "placement_boost":1.10},
}
branch_names = list(BRANCHES.keys())
branch_w     = [BRANCHES[b]["weight"] for b in branch_names]

# ── Generate ──────────────────────────────────────────────────────────────────
names   = [f"{first[i % len(first)]} {last[i % len(last)]}" for i in range(N)]
batches = np.random.choice([2020, 2021, 2022], N, p=[0.30, 0.35, 0.35])
branches= np.random.choice(branch_names, N, p=branch_w)

# USN: 1VB<YY><BR><NNN>
branch_code = {"CSE":"CS","ISE":"IS","ECE":"EC","EEE":"EE","MECH":"ME","CIVIL":"CV","AIML":"AI","DS":"DS"}
usns = [f"1VB{str(batches[i])[2:]}{branch_code[branches[i]]}{(i%100):03d}" for i in range(N)]

# ── Academic scores ───────────────────────────────────────────────────────────
# Underlying academic ability drives most scores
ability = np.clip(np.random.normal(65, 15, N), 30, 100)

cgpa            = np.clip(ability/10 + np.random.normal(0, 0.5, N), 4.0, 10.0).round(2)
tenth_pct       = np.clip(ability + np.random.normal(0, 8, N), 35, 100).round(1)
twelfth_pct     = np.clip(ability + np.random.normal(0, 10, N), 35, 100).round(1)
backlogs        = np.where(cgpa < 6.0,
                           np.random.choice([0,1,2,3,4], N, p=[0.2,0.3,0.25,0.15,0.10]),
                           np.where(cgpa < 7.5,
                                    np.random.choice([0,0,1,2], N, p=[0.55,0.25,0.15,0.05]),
                                    np.zeros(N, int)))

# ── Aptitude & verbal ─────────────────────────────────────────────────────────
aptitude_score      = np.clip(ability * 0.8 + np.random.normal(5, 12, N), 10, 100).round(1)
logical_score       = np.clip(ability * 0.75+ np.random.normal(5, 14, N), 10, 100).round(1)
communication_score = np.clip(np.random.normal(62, 16, N), 15, 100).round(1)
english_score       = np.clip(communication_score * 0.9 + np.random.normal(0, 8, N), 10, 100).round(1)

# ── Technical skills (0–10) ───────────────────────────────────────────────────
tech_boost = np.array([BRANCHES[b]["tech_boost"] for b in branches])
prog_base  = np.clip(ability/12 + np.random.normal(0, 1.8, N), 0, 10) * tech_boost
programming_skill = np.clip(prog_base, 0, 10).round(1)
database_skill    = np.clip(prog_base * 0.85 + np.random.normal(0, 1, N), 0, 10).round(1)
web_skill         = np.clip(prog_base * 0.80 + np.random.normal(0, 1.5, N), 0, 10).round(1)
ml_skill          = np.clip(prog_base * 0.70 + np.random.normal(0, 1.5, N), 0, 10).round(1)
cloud_skill       = np.clip(prog_base * 0.65 + np.random.normal(0, 1.5, N), 0, 10).round(1)
dsa_skill         = np.clip(prog_base * 0.90 + np.random.normal(0, 1.2, N), 0, 10).round(1)

# ── Extra-curricular ──────────────────────────────────────────────────────────
num_projects       = np.clip(np.random.poisson(2.5, N), 0, 8)
num_certifications = np.clip(np.random.poisson(2.0, N), 0, 7)
hackathons         = np.clip(np.random.poisson(1.2, N), 0, 5)
paper_publications = np.clip(np.random.poisson(0.4, N), 0, 3)

# ── Internship ────────────────────────────────────────────────────────────────
internship_prob = np.where(
    (cgpa > 7.5) & (programming_skill > 6),
    np.random.uniform(0.65, 0.90, N),
    np.where(cgpa > 6.5, np.random.uniform(0.30, 0.55, N),
             np.random.uniform(0.10, 0.30, N))
)
internship_done   = (np.random.rand(N) < internship_prob).astype(int)
internship_months = np.where(internship_done, np.random.randint(1, 7, N), 0)
internship_stipend= np.where(internship_done,
                              np.clip(programming_skill * 2000 + np.random.normal(5000, 3000, N), 3000, 35000).round(-2),
                              0).astype(int)

# ── Soft skills ───────────────────────────────────────────────────────────────
leadership_score = np.clip(np.random.normal(5.8, 1.8, N), 0, 10).round(1)
teamwork_score   = np.clip(np.random.normal(6.8, 1.5, N), 0, 10).round(1)
gd_score         = np.clip(communication_score * 0.08 + np.random.normal(5, 1.5, N), 0, 10).round(1)

# ── Placement label ───────────────────────────────────────────────────────────
place_boost = np.array([BRANCHES[b]["placement_boost"] for b in branches])
placement_score = (
    cgpa                   * 3.0 +
    aptitude_score         * 0.08 +
    logical_score          * 0.06 +
    communication_score    * 0.05 +
    programming_skill      * 1.8 +
    dsa_skill              * 1.5 +
    num_certifications     * 0.8 +
    internship_done        * 4.0 +
    internship_months      * 0.5 +
    num_projects           * 0.6 +
    hackathons             * 0.5 +
    paper_publications     * 0.4 -
    backlogs               * 1.5
) * place_boost

ps_min, ps_max = placement_score.min(), placement_score.max()
prob = (placement_score - ps_min) / (ps_max - ps_min)
prob = np.clip(prob + np.random.normal(0, 0.04, N), 0.03, 0.97)
placed = (prob > 0.48).astype(int)

# ── Package (CTC in LPA for placed students) ─────────────────────────────────
base_pkg = cgpa * 0.9 + programming_skill * 0.6 + dsa_skill * 0.5
package_lpa = np.where(
    placed == 1,
    np.clip(base_pkg * place_boost + np.random.normal(2, 1.5, N), 2.5, 22.0).round(2),
    0.0
)

df = pd.DataFrame({
    "name": names, "usn": usns, "batch": batches, "branch": branches,
    "tenth_percentage":     tenth_pct,
    "twelfth_percentage":   twelfth_pct,
    "cgpa":                 cgpa,
    "backlogs":             backlogs,
    "aptitude_score":       aptitude_score,
    "logical_score":        logical_score,
    "communication_score":  communication_score,
    "english_score":        english_score,
    "programming_skill":    programming_skill,
    "dsa_skill":            dsa_skill,
    "database_skill":       database_skill,
    "web_skill":            web_skill,
    "ml_skill":             ml_skill,
    "cloud_skill":          cloud_skill,
    "num_projects":         num_projects,
    "num_certifications":   num_certifications,
    "hackathons":           hackathons,
    "paper_publications":   paper_publications,
    "internship_done":      internship_done,
    "internship_months":    internship_months,
    "internship_stipend":   internship_stipend,
    "leadership_score":     leadership_score,
    "teamwork_score":       teamwork_score,
    "gd_score":             gd_score,
    "placed":               placed,
    "package_lpa":          package_lpa,
})

# Introduce realistic missing values (as would appear in real data)
for col in ["aptitude_score","logical_score","english_score","gd_score"]:
    mask = np.random.rand(N) < 0.04   # 4% missing
    df.loc[mask, col] = np.nan

df.to_csv("placement_data.csv", index=False)
print(f"Dataset saved: {len(df)} students")
print(f"Placed: {placed.sum()} ({placed.mean()*100:.1f}%)  |  Not Placed: {(placed==0).sum()}")
print(f"Branches: {dict(zip(*np.unique(branches, return_counts=True)))}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
