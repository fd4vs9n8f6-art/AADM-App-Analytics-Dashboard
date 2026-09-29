
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    silhouette_score
)

st.set_page_config(page_title="App Analytics Decision Dashboard", page_icon="📊", layout="wide")

DATA_PATH = "data/googleplaystore_DS11.csv"
REFERENCE_DATE = pd.Timestamp("2026-09-25")

REG_FEATURES = [
    "Price_INR","Estimated_Installs","App_Size_MB","Minimum_Android_Version",
    "Supported_Languages","Monthly_Active_Users","Average_Session_Minutes",
    "Crash_Rate_Percentage","App_Load_Time_Seconds","Average_Review_Sentiment",
    "Positive_Review_Percentage","Update_Frequency_Days",
    "Store_Listing_Conversion_Percentage","Days_Since_Update","App_Age_Days"
]
CLASS_FEATURES = [
    "Rating","Rating_Count","Estimated_Installs","App_Size_MB","Price_INR",
    "Minimum_Android_Version","Supported_Languages","Monthly_Active_Users",
    "Average_Session_Minutes","Crash_Rate_Percentage","App_Load_Time_Seconds",
    "Average_Review_Sentiment","Positive_Review_Percentage","Update_Frequency_Days",
    "Store_Listing_Conversion_Percentage","Days_Since_Update","App_Age_Days"
]
CLUSTER_FEATURES = [
    "Rating","Estimated_Installs","Monthly_Active_Users",
    "User_Retention_30D_Percentage","Average_Session_Minutes",
    "Crash_Rate_Percentage","App_Load_Time_Seconds",
    "Positive_Review_Percentage","Store_Listing_Conversion_Percentage",
    "Estimated_Monthly_Revenue_INR"
]

@st.cache_data
def load_data():
    d = pd.read_csv(DATA_PATH)
    d["Release_Date"] = pd.to_datetime(d["Release_Date"])
    d["Last_Updated"] = pd.to_datetime(d["Last_Updated"])
    d["Days_Since_Update"] = (REFERENCE_DATE - d["Last_Updated"]).dt.days
    d["App_Age_Days"] = (REFERENCE_DATE - d["Release_Date"]).dt.days
    d["Log_Monthly_Revenue"] = np.log1p(d["Estimated_Monthly_Revenue_INR"])
    median = d["User_Retention_30D_Percentage"].median()
    d["High_Retention"] = (d["User_Retention_30D_Percentage"] >= median).astype(int)
    return d, median

@st.cache_resource
def train_models(d):
    # Revenue regression
    X = d[REG_FEATURES]
    y = d["Estimated_Monthly_Revenue_INR"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42)
    raw = LinearRegression().fit(Xtr, ytr)
    pred = raw.predict(Xte)
    r2 = r2_score(yte, pred)
    raw_metrics = {
        "R²": r2,
        "Adjusted R²": 1-((1-r2)*(len(Xte)-1)/(len(Xte)-Xte.shape[1]-1)),
        "RMSE": np.sqrt(mean_squared_error(yte,pred)),
        "MAE": mean_absolute_error(yte,pred)
    }

    ylog = d["Log_Monthly_Revenue"]
    Xtr, Xte, ytr, yte = train_test_split(X, ylog, test_size=.20, random_state=42)
    log_model = LinearRegression().fit(Xtr, ytr)
    lp = log_model.predict(Xte)
    lr2 = r2_score(yte, lp)
    log_metrics = {
        "R²": lr2,
        "Adjusted R²": 1-((1-lr2)*(len(Xte)-1)/(len(Xte)-Xte.shape[1]-1)),
        "RMSE": np.sqrt(mean_squared_error(yte,lp)),
        "MAE": mean_absolute_error(yte,lp)
    }

    # Retention classification; target itself is excluded from predictors.
    Xc = d[CLASS_FEATURES]
    yc = d["High_Retention"]
    Xtr,Xte,ytr,yte = train_test_split(
        Xc,yc,test_size=.20,random_state=42,stratify=yc
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,min_samples_split=20,min_samples_leaf=10,random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,max_depth=8,min_samples_split=10,
            min_samples_leaf=5,random_state=42,n_jobs=-1
        )
    }
    fitted, metrics = {}, {}
    for name,m in models.items():
        m.fit(Xtr,ytr)
        fitted[name]=m
        p=m.predict(Xte)
        pr=m.predict_proba(Xte)[:,1]
        metrics[name]={
            "Accuracy":accuracy_score(yte,p),
            "Precision":precision_score(yte,p,zero_division=0),
            "Recall":recall_score(yte,p,zero_division=0),
            "ROC-AUC":roc_auc_score(yte,pr)
        }

    # K-Means, k=2 as used for final segmentation.
    scaler=StandardScaler()
    Z=scaler.fit_transform(d[CLUSTER_FEATURES])
    km=KMeans(n_clusters=2,random_state=42,n_init=10)
    labels=km.fit_predict(Z)
    cd=d.copy()
    cd["Cluster"]=labels
    sil=silhouette_score(Z,labels)

    return raw,log_model,raw_metrics,log_metrics,fitted,metrics,cd,sil

df, retention_median = load_data()
raw_model,log_model,raw_metrics,log_metrics,classifiers,class_metrics,cluster_df,sil = train_models(df)

st.sidebar.title("📊 App Analytics")
page=st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard","Revenue Analytics","Retention Prediction",
     "App Segmentation","Model Performance","Data Explorer"]
)

if page=="Executive Dashboard":
    st.title("📊 App Analytics Decision Dashboard")
    st.caption("Advanced Analytics for Decision-Making — interactive project deployment")

    a,b,c,d=st.columns(4)
    a.metric("Apps",f"{len(df):,}")
    b.metric("Total Estimated Monthly Revenue",f"₹{df.Estimated_Monthly_Revenue_INR.sum()/1e9:.2f}B")
    c.metric("Total MAUs",f"{df.Monthly_Active_Users.sum()/1e9:.2f}B")
    d.metric("Median 30-Day Retention",f"{retention_median:.2f}%")

    l,r=st.columns(2)
    with l:
        x=df.Category.value_counts().reset_index()
        x.columns=["Category","Apps"]
        fig=px.bar(x,x="Apps",y="Category",orientation="h",title="Apps by Category")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)
    with r:
        x=df.groupby("Category",as_index=False).Estimated_Monthly_Revenue_INR.mean()
        fig=px.bar(x,x="Estimated_Monthly_Revenue_INR",y="Category",
                   orientation="h",title="Average Monthly Revenue by Category")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Key Analytical Signals")
    a,b,c=st.columns(3)
    a.metric("Price ↔ Revenue",f"{df.Price_INR.corr(df.Estimated_Monthly_Revenue_INR):.3f}")
    b.metric("Installs ↔ Revenue",f"{df.Estimated_Installs.corr(df.Estimated_Monthly_Revenue_INR):.3f}")
    c.metric("MAUs ↔ Revenue",f"{df.Monthly_Active_Users.corr(df.Estimated_Monthly_Revenue_INR):.3f}")

elif page=="Revenue Analytics":
    st.title("💰 Revenue Analytics")
    driver=st.selectbox("Choose revenue driver",["Price_INR","Estimated_Installs","Monthly_Active_Users"])
    fig=px.scatter(
        df,x=driver,y="Estimated_Monthly_Revenue_INR",
        hover_data=["App_Name","Category"],log_y=True,
        title=f"{driver} vs Estimated Monthly Revenue"
    )
    st.plotly_chart(fig,use_container_width=True)

    a,b,c,e=st.columns(4)
    a.metric("Raw R²",f"{raw_metrics['R²']:.4f}")
    b.metric("Adjusted R²",f"{raw_metrics['Adjusted R²']:.4f}")
    c.metric("Raw RMSE",f"₹{raw_metrics['RMSE']/1e6:.1f}M")
    e.metric("Raw MAE",f"₹{raw_metrics['MAE']/1e6:.1f}M")

    st.info("The project also models log revenue because revenue is highly right-skewed. Log-scale error metrics should not be compared directly with raw-rupee error metrics.")

    st.subheader("Revenue Scenario Predictor")
    defaults={
        "Price_INR":67.36,"Estimated_Installs":1_000_000,"App_Size_MB":234.9,
        "Minimum_Android_Version":11,"Supported_Languages":18,
        "Monthly_Active_Users":1_000_000,"Average_Session_Minutes":43,
        "Crash_Rate_Percentage":2.25,"App_Load_Time_Seconds":4.3,
        "Average_Review_Sentiment":0.0,"Positive_Review_Percentage":65,
        "Update_Frequency_Days":90,"Store_Listing_Conversion_Percentage":9.2,
        "Days_Since_Update":180,"App_Age_Days":1500
    }
    vals={}
    cols=st.columns(3)
    for i,f in enumerate(REG_FEATURES):
        with cols[i%3]:
            vals[f]=st.number_input(f,value=float(defaults[f]))
    row=pd.DataFrame([vals],columns=REG_FEATURES)
    p1=max(0,raw_model.predict(row)[0])
    p2=max(0,np.expm1(log_model.predict(row)[0]))
    a,b=st.columns(2)
    a.metric("Raw Linear Prediction",f"₹{p1:,.0f}")
    b.metric("Log-Model Back-Transformed Prediction",f"₹{p2:,.0f}")

elif page=="Retention Prediction":
    st.title("🎯 Retention Prediction")
    st.caption(f"High Retention = 30-day retention ≥ {retention_median:.2f}% (project median).")

    defaults={
        "Rating":4.15,"Rating_Count":125000,"Estimated_Installs":1_000_000,
        "App_Size_MB":235,"Price_INR":67,"Minimum_Android_Version":11,
        "Supported_Languages":18,"Monthly_Active_Users":1_000_000,
        "Average_Session_Minutes":43,"Crash_Rate_Percentage":2.25,
        "App_Load_Time_Seconds":4.3,"Average_Review_Sentiment":0,
        "Positive_Review_Percentage":65,"Update_Frequency_Days":90,
        "Store_Listing_Conversion_Percentage":9.2,"Days_Since_Update":180,
        "App_Age_Days":1500
    }
    vals={}
    cols=st.columns(3)
    for i,f in enumerate(CLASS_FEATURES):
        with cols[i%3]:
            vals[f]=st.number_input(f,value=float(defaults[f]))
    row=pd.DataFrame([vals],columns=CLASS_FEATURES)
    rf=classifiers["Random Forest"]
    pred=int(rf.predict(row)[0])
    prob=float(rf.predict_proba(row)[0,1])
    if pred==1:
        st.success(f"Predicted segment: HIGH RETENTION — probability {prob:.1%}")
    else:
        st.warning(f"Predicted segment: LOWER RETENTION — high-retention probability {prob:.1%}")
    st.info("The project's retention classifiers showed limited discrimination. This should be treated as a scenario/decision-support tool, not a definitive forecast.")

elif page=="App Segmentation":
    st.title("🧩 App Segmentation")
    a,b,c=st.columns(3)
    a.metric("Clusters","2")
    b.metric("Silhouette Score",f"{sil:.3f}")
    c.metric("Cluster 0 Apps",f"{(cluster_df.Cluster==0).sum():,}")

    profile=cluster_df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
    st.subheader("Cluster Profiles")
    st.dataframe(profile,use_container_width=True)

    metric=st.selectbox("Compare clusters on",[
        "Estimated_Installs","Monthly_Active_Users",
        "Estimated_Monthly_Revenue_INR","User_Retention_30D_Percentage",
        "Average_Session_Minutes"
    ])
    x=cluster_df.groupby("Cluster",as_index=False)[metric].mean()
    fig=px.bar(x,x="Cluster",y=metric,title=f"Average {metric} by Cluster")
    st.plotly_chart(fig,use_container_width=True)

    st.write("The final two-cluster solution separates a smaller high-scale group from a much larger lower-scale group. The main differences are scale and monetization measures such as installs, MAUs and revenue.")

elif page=="Model Performance":
    st.title("🤖 Model Performance")
    st.subheader("Revenue Regression")
    reg=pd.DataFrame([
        {"Model":"Linear Regression — Raw Revenue",**raw_metrics},
        {"Model":"Linear Regression — Log Revenue",**log_metrics}
    ])
    st.dataframe(reg,use_container_width=True)

    st.subheader("Retention Classification")
    cm=pd.DataFrame(class_metrics).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(cm,use_container_width=True)
    fig=px.bar(cm,x="Model",y=["Accuracy","Precision","Recall","ROC-AUC"],
               barmode="group",title="Classification Model Metrics")
    st.plotly_chart(fig,use_container_width=True)

elif page=="Data Explorer":
    st.title("🔎 Data Explorer")
    cats=st.multiselect("Category filter",sorted(df.Category.unique()))
    view=df if not cats else df[df.Category.isin(cats)]
    a,b=st.columns(2)
    a.metric("Rows shown",f"{len(view):,}")
    b.metric("Missing values",f"{int(view.isna().sum().sum()):,}")
    st.dataframe(
        view[["App_Name","Category","Rating","Estimated_Installs",
              "Monthly_Active_Users","User_Retention_30D_Percentage",
              "Estimated_Monthly_Revenue_INR"]]
        .sort_values("Estimated_Monthly_Revenue_INR",ascending=False),
        use_container_width=True,height=520
    )

st.sidebar.markdown("---")
st.sidebar.caption("AADM End-Term Project")
