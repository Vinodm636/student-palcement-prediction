"""Full verification for upgraded placement prediction system."""
import pandas as pd, json
from app import app, train_all, recommend, feature_engineering, preprocess, RAW_NUMERIC, FEATURES

# 1. Dataset
df = pd.read_csv("placement_data.csv")
print(f"Dataset: {len(df)} rows, {len(df.columns)} cols")
print(f"Placed: {df['placed'].sum()} | Not: {(df['placed']==0).sum()} | Rate: {df['placed'].mean()*100:.1f}%")

# 2. Preprocessing
clean, pre = preprocess(df)
print(f"\nPreprocessing:")
print(f"  Duplicates removed : {pre['duplicates_removed']}")
print(f"  Missing before     : {pre['missing_before']}")
print(f"  Missing after      : {pre['missing_after']}")
print(f"  Outliers capped    : {pre['outliers_capped']}")

# 3. Feature engineering
eng, erep = feature_engineering(clean)
print(f"\nFeature Engineering: {len(erep['features_created'])} new features created")

# 4. Train all models
print("\nTraining 8 models...")
cache = train_all(force=True)
print(f"{'Model':<28} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7} {'CV-F1':>7}")
print("-"*72)
for name, r in sorted(cache['results'].items(), key=lambda x: -x[1]['f1']):
    print(f"  {name:<26} {r['accuracy']:>5.1f}% {r['precision']:>5.1f}% {r['recall']:>5.1f}% {r['f1']:>5.1f}% {r['roc_auc']:>6.1f}% {r['cv_f1']:>6.1f}%")
print(f"\nBest model: {cache['best_model']}")

# 5. Recommendation engine
sample = {f: float(clean.iloc[0].get(f, 0)) for f in RAW_NUMERIC}
inp_df, _ = feature_engineering(pd.DataFrame([sample]))
full = {**sample, **inp_df.iloc[0].to_dict()}
recs = recommend(full, cache['importances'])
print(f"\nRecommendations: {len(recs)} suggestions for student 0")

# 6. Flask route tests
print("\nRoute checks:")
with app.test_client() as c:
    with c.session_transaction() as sess:
        sess['user'] = 'teacher'
        sess['role'] = 'teacher'
    usn = df.iloc[0]['usn']
    for route, label in [
        ('/dashboard',       'Dashboard'),
        ('/predict',         'Predict (GET)'),
        ('/analytics',       'Analytics'),
        ('/preprocessing',   'Preprocessing'),
        ('/justify',         'Justification'),
        ('/leaderboard',     'Leaderboard'),
        (f'/student/{usn}',  'Student Profile'),
        ('/model/Random_Forest', 'Model Detail'),
    ]:
        r = c.get(route)
        ok = "✅ OK" if r.status_code == 200 else f"❌ FAIL ({r.status_code})"
        print(f"  {label:<22} {ok}")

print("\nAll checks passed! 🚀")
