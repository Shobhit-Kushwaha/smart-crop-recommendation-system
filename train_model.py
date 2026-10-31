from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; MODELS=ROOT/"models"; REPORTS=ROOT/"reports"
for p in (DATA, MODELS, REPORTS): p.mkdir(exist_ok=True)

csv_path=DATA/"Crop_recommendation.csv"
if not csv_path.exists():
    raise FileNotFoundError("Dataset missing. Put Crop_recommendation.csv in data/ (see README for download source).")

df=pd.read_csv(csv_path)
required={"N","P","K","temperature","humidity","ph","rainfall","label"}
missing=required-set(df.columns)
if missing: raise ValueError(f"Dataset missing columns: {sorted(missing)}")
df=df[list(required)].dropna()
X=df[["N","P","K","temperature","humidity","ph","rainfall"]]
y=df["label"].astype(str)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)

models={
 "Random Forest":RandomForestClassifier(n_estimators=350,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=42,n_jobs=-1),
 "Extra Trees":ExtraTreesClassifier(n_estimators=350,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1),
 "KNN":Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5,weights="distance"))]),
 "Gradient Boosting":HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,l2_regularization=.1,random_state=42)
}
rows=[]; fitted={}
for name,m in models.items():
    m.fit(X_train,y_train)
    pred=m.predict(X_test)
    rows.append({"model":name,"accuracy":accuracy_score(y_test,pred),"macro_f1":f1_score(y_test,pred,average="macro")})
    fitted[name]=m
results=pd.DataFrame(rows).sort_values(["accuracy","macro_f1"],ascending=False)
best_name=results.iloc[0]["model"]; best=fitted[best_name]
joblib.dump(best,MODELS/"crop_model.joblib")
results.to_csv(REPORTS/"model_comparison.csv",index=False)
pred=best.predict(X_test)
labels=sorted(y.unique())
pd.DataFrame(confusion_matrix(y_test,pred,labels=labels),index=labels,columns=labels).to_csv(REPORTS/"confusion_matrix.csv")
(REPORTS/"classification_report.txt").write_text(classification_report(y_test,pred),encoding="utf-8")
meta={"best_model":best_name,"features":list(X.columns),"classes":labels,"rows":len(df),"test_accuracy":float(accuracy_score(y_test,pred)),"test_size":.2,"random_state":42}
(MODELS/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("MODEL COMPARISON")
print(results.to_string(index=False,formatters={"accuracy":"{:.2%}".format,"macro_f1":"{:.3f}".format}))
print(f"\nSelected: {best_name} | Test accuracy: {meta['test_accuracy']:.2%}")
print("Use these metrics as measured results for this split only; validate on local soil and crop data before real-world use.")
