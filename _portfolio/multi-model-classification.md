---
title: "多分类模型性能比较与评估"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/multi-model-classification
date: 2026-01-18
excerpt: "通过Logistic Regression、Random Forest和SVM模型对多特征数据进行分类，并从准确率、F1分数和ROC曲线等方面进行综合评估，为分类任务选择最优模型提供依据。"
header:
  teaser: /images/portfolio/multi-model-classification/model_comparison.png
tags:
  - 机器学习
  - 分类算法
  - 模型评估
  - Python
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景  
本项目旨在通过机器学习模型解决多分类问题。数据集包含5个特征（x1-x5）和1个类别标签（group），目标是比较三种经典分类算法的性能：Logistic Regression、Random Forest和支持向量机（SVM），并通过混淆矩阵、ROC曲线等指标选择最优模型。


## 核心实现  
### 1. 数据预处理与划分  
```python
# 数据读取与变量定义
data = pd.read_excel("data.xlsx")
label_col = "group"
feature_cols = ["x1", "x2", "x3", "x4", "x5"]
X = data[feature_cols]
y = data[label_col]

# 分层抽样划分训练集/测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 2. 模型定义  
```python
# 构建模型 pipeline（含标准化预处理）
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            multi_class="multinomial", 
            class_weight="balanced", 
            max_iter=200, 
            random_state=42
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, 
        max_depth=6, 
        class_weight="balanced_subsample", 
        random_state=42
    ),
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear", 
            class_weight="balanced", 
            probability=True, 
            random_state=42
        ))
    ])
}
```

### 3. 模型评估函数  
```python
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # 训练与预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # 输出关键指标
    metrics = {
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro-F1": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, y_prob, multi_class="ovr")
    }
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    
    # ROC曲线（OvR策略）
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    plt.figure()
    for i, cls in enumerate(np.unique(y_test)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{name} - ROC Curve")
    
    return {"Model": name, **metrics}
```


## 分析结果  
### 1. 特征分布分析  
各特征在不同类别下的箱线图显示，x3和x5的分布差异较为显著，可能对分类贡献更大。  
![x3箱线图](/images/portfolio/multi-model-classification/x3_boxplot.png)  
![x5箱线图](/images/portfolio/multi-model-classification/x5_boxplot.png)  

### 2. 模型性能对比  
**混淆矩阵示例（Logistic Regression）**：  
![Random Forest混淆矩阵](/images/portfolio/multi-model-classification/logistic_cm.png)  
*该模型对类别2的识别准确率最高，类别1存在一定混淆。*

**ROC曲线示例（Random Forest）**：  
![SVM ROC曲线](/images/portfolio/multi-model-classification/rf_roc.png)  
*三类别的AUC均在0.7以上，模型整体区分能力一般。*

**关键指标汇总**：  
| 模型                | Balanced Accuracy | Macro-F1 | ROC-AUC |  
|---------------------|-------------------|----------|---------|  
| Random Forest       | 0.520             | 0.496    | 0.667   |  
| Logistic Regression | 0.408             | 0.410    | 0.669   |  
| Linear SVM          | 0.375             | 0.301    | 0.585   |  

**模型对比**：  
![模型性能对比](/images/portfolio/multi-model-classification/model_comparison.png)  
*Random Forest在平衡准确率和Macro-F1分数上均优于其他模型，是本次任务的最优选择。*


## 结论  
本项目通过系统化实验对比了三种分类模型的性能。结果表明，Random Forest在多特征分类任务中表现最佳，其平衡准确率达0.520，Macro-F1分数0.496，适合作为该数据集的最终模型。未来可进一步通过特征工程或超参数调优提升性能。
```
