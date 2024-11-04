import numpy as np
import pandas as pd

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,classification_report, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, StratifiedKFold,cross_validate, KFold, GridSearchCV
import shap
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

# 数据读取
df = pd.read_excel("D:\桌面\8-13 插补后数据\ditan+youan11287随机森林插补 清洗.xlsx") 
label = df.iloc[:, 0]
features = df.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42, stratify=label)
# 过采样
sampling_strategy = {1: 3500}
smote = SMOTE(random_state=42, sampling_strategy = sampling_strategy)
x_resample, y_resample = smote.fit_resample(X_train, y_train)
# 模型
rf = RandomForestClassifier(random_state=44)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_smote = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=44)
rf_smote.fit(x_resample, y_resample)
rf_smote_preds = rf_smote.predict(X_test)

xgb = XGBClassifier(random_state=44)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_smote = XGBClassifier(n_estimators=10, max_depth=4, reg_alpha=1, random_state=44)
xgb_smote.fit(x_resample, y_resample)
xgb_smote_preds = xgb_smote.predict(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, ), random_state=42)  # 设置随机种子
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_smote = MLPClassifier(hidden_layer_sizes=(100, ), random_state=42)  # 设置随机种子
mlp_smote.fit(x_resample, y_resample)
mlp_smote_pred = mlp_smote.predict(X_test)

lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
lgb_smote = LGBMClassifier(random_state=42)
lgb_smote.fit(x_resample, y_resample)
lgb_smote_pred = lgb_smote.predict(X_test)

gbm = GradientBoostingClassifier(random_state=44)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_smote = GradientBoostingClassifier(random_state=44)
gbm_smote.fit(x_resample, y_resample)
gbm_smote_pred = gbm_smote.predict(X_test)

adaboost = AdaBoostClassifier(random_state=44)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict(X_test)
adaboost_smote = AdaBoostClassifier(random_state=44)
adaboost_smote.fit(x_resample, y_resample)
adaboost_smote_pred = adaboost_smote.predict(X_test)

tabnet = TabNetClassifier()
tabnet.fit(X_train.values, y_train.values)
tabnet_pred = tabnet.predict(X_test.values)
tabnet_smote = TabNetClassifier()
tabnet_smote.fit(x_resample.values, y_resample.values)
tabnet_smote_pred = tabnet_smote.predict(X_test.values)

# 计算各模型的ROC曲线和AUC值
roc_values = {}

# 计算RF的ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_smote.predict_proba(X_test.values)[:, 1])
rf_auc = auc(fpr_rf, tpr_rf)
roc_values['RF'] = (fpr_rf, tpr_rf, rf_auc)

# 计算RF的ROC曲线
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_smote.predict_proba(X_test.values)[:, 1])
xgb_auc = auc(fpr_xgb, tpr_xgb)
roc_values['XGBoost'] = (fpr_xgb, tpr_xgb, xgb_auc)

# 计算MLP的ROC曲线
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_smote.predict_proba(X_test)[:, 1])
mlp_auc = auc(fpr_mlp, tpr_mlp)
roc_values['MLP'] = (fpr_mlp, tpr_mlp, mlp_auc)

# 计算MLP的ROC曲线
fpr_tabnet, tpr_tabnet, _ = roc_curve(y_test, tabnet_smote.predict_proba(X_test.values)[:, 1])
tabnet_auc = auc(fpr_tabnet, tpr_tabnet)
roc_values['TabNet'] = (fpr_tabnet, tpr_tabnet, tabnet_auc)

# 计算MLP的ROC曲线
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, gbm_smote.predict_proba(X_test.values)[:, 1])
gbm_auc = auc(fpr_gbm, tpr_gbm)
roc_values['GBM'] = (fpr_gbm, tpr_gbm, gbm_auc)

# 计算MLP的ROC曲线
fpr_adaboost, tpr_adaboost, _ = roc_curve(y_test, adaboost_smote.predict_proba(X_test.values)[:, 1])
adaboost_auc = auc(fpr_adaboost, tpr_adaboost)
roc_values['AdaBoost'] = (fpr_adaboost, tpr_adaboost, adaboost_auc)

# P的ROC曲线
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_smote.predict_proba(X_test.values)[:, 1])
lgb_auc = auc(fpr_lgb, tpr_lgb)
roc_values['LigntGBM'] = (fpr_lgb, tpr_lgb, lgb_auc)

# 按照AUC值对模型进行排序
sorted_roc_values = sorted(roc_values.items(), key=lambda x: x[1][2], reverse=True)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

for model_name, (fpr, tpr, roc_auc) in sorted_roc_values:
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# 添加图例和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 显示图形
plt.show()

class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:        
            print("There is NO significant difference")
            
feature_3 = ['CD4', 'CD4/CD8 ratio', 'CD8']
feature_8 = ['CD4', 'CD4/CD8 ratio', 'CD8', 'HGB', 'WBC', 'TBIL', 'Age at ART initial', 'PLT']
feature_9 = ['CD4', 'CD4/CD8 ratio', 'CD8', 'HGB', 'WBC', 'TBIL', 'Age at ART initial', 'PLT', "VL"]
feature_15 = ['CD4', 'CD4/CD8 ratio', 'CD8', 'HGB', 'WBC', 'TBIL', 'Age at ART initial', 'PLT', 'VL', 
            'WHO stage', 'ALT', 'Scr', 'AST', 'Duration of stating ART', 'Marital status']
recall_scores = []
f1_scores = []
auroc_scores = []
precision_scores = []
acc_scores = []
balance_acc = []
prob_scores = []
roc_values_redu = {}
reduction_name = ["redu_3",  "redu_9", "redu_15", "redu_20"]
for i, feature_name in enumerate([feature_3, feature_9, feature_15, features.columns.values]):
    feature_reduction = features.loc[:, feature_name]
    X_redu_train, X_redu_test, y_redu_train, y_redu_test = train_test_split(feature_reduction, 
                                                                      label, test_size=0.3, random_state=42, stratify=label)
    x_redu_resample, y_redu_resample = smote.fit_resample(X_redu_train, y_redu_train)
    rf_redu = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=44)  
    rf_redu.fit(x_redu_resample, y_redu_resample)  
      
    y_pred = rf_redu.predict(X_redu_test)  
    y_pred_proba = rf_redu.predict_proba(X_redu_test)[:, 1]  
      
    recall_scores.append(recall_score(y_redu_test, y_pred, average='weighted'))  
    f1_scores.append(f1_score(y_redu_test, y_pred, average='weighted'))  
    auroc_scores.append(roc_auc_score(y_redu_test, y_pred_proba))
    prob_scores.append(y_pred_proba)
    precision_scores.append(precision_score(y_redu_test, y_pred, average='weighted'))
    acc_scores.append(accuracy_score(y_redu_test, y_pred))
    balance_acc.append(balanced_accuracy_score(y_redu_test, y_pred))

    fpr_rf_redu, tpr_rf_redu, _ = roc_curve(y_test, rf_redu.predict_proba(X_redu_test.values)[:, 1])
    roc_values_redu[reduction_name[i]] = (fpr_rf_redu, tpr_rf_redu, roc_auc_score(y_redu_test, y_pred_proba))

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model
def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all
def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, color, model_name, model_only=False):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = color, label = model_name)
    if not model_only:
        ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
        ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax
    
thresh_group = np.arange(0, 1, 0.05)
net_benefit_20 = calculate_net_benefit_model(thresh_group, prob_scores[3], y_redu_test, )
net_benefit_8 = calculate_net_benefit_model(thresh_group, prob_scores[1], y_redu_test)
net_benefit_3 = calculate_net_benefit_model(thresh_group, prob_scores[0], y_redu_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_redu_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_20, net_benefit_all, color='r', model_name="RF model with 20 features", model_only=True)
ax = plot_DCA(ax, thresh_group, net_benefit_8, net_benefit_all, color='b', model_name="RF model with 8 features", model_only=True)
ax = plot_DCA(ax, thresh_group, net_benefit_3, net_benefit_all, color='g', model_name="RF model with 3 features")
plt.show()

x_new_resample, y_new_resample = smote.fit_resample(X_new_train, y_new_train)
rf1_smote = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=44)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
y_scores = cross_val_predict(rf1_smote, x_new_resample, y_new_resample, cv=cv, method='predict_proba') 
y_scores = y_scores[:, 1]  
  
# 准备画图的存储  
fpr_list = []  
tpr_list = []  
roc_auc_list = []  
  
# 遍历每一折  
for i, (_, test_index) in enumerate(cv.split(x_new_resample, y_new_resample)):  
    # 提取当前折的测试集真实标签和预测概率  
    y_test = y_new_resample[test_index]  
    y_scores_fold = y_scores[test_index]  
      
    # 计算当前折的ROC曲线数据  
    fpr, tpr, thresholds = roc_curve(y_test, y_scores_fold)  
    roc_auc = auc(fpr, tpr)  
      
    # 存储当前折的FPR, TPR和AUC  
    fpr_list.append(fpr)  
    tpr_list.append(tpr)  
    roc_auc_list.append(roc_auc)  
      
    # 可选：打印当前折的AUC  
    print(f"Fold {i+1} AUC: {roc_auc:.4f}")  

# 计算平均FPR和TPR  
all_fpr = np.unique(np.concatenate([a for a in fpr_list]))  
mean_tpr = np.zeros_like(all_fpr, dtype=np.float64)  
  
for i, fpr in enumerate(fpr_list):  
    tpr = tpr_list[i]  
    mean_tpr += np.interp(all_fpr, fpr, tpr, left=0, right=1)  
  
mean_tpr /= len(fpr_list)  
  
# 计算平均ROC曲线的AUC  
mean_auc = auc(all_fpr, mean_tpr) 
std_auc = np.std(roc_auc_list) 
  
# 绘制每一折的ROC曲线  
plt.figure(figsize=(8, 6))  
colors = ['blue', 'red', 'green', 'orange', 'purple']  
for i, color in enumerate(colors):  
    plt.plot(fpr_list[i], tpr_list[i], color=color, alpha=0.3, lw=2,  
             label=f'ROC fold {i+1} (AUC = {roc_auc_list[i]:.3f})')  

plt.plot(all_fpr, mean_tpr, color='black',  
         label=f'Mean ROC (AUC = {mean_auc:0.3f} ± {std_auc:0.3f})',  
         lw=2, alpha=0.8)
  
plt.plot([0, 1], [0, 1], 'k--', lw=2)  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Five fold')  
plt.legend(loc="lower right")  
plt.show()


explanier = shap.TreeExplainer(rf1_smote)
shap_values = explanier.shap_values(X_new_test)
shap.summary_plot(shap_values, X_new_test, plot_type='bar', class_names=["INR", "Non-INR"])
shap.summary_plot(shap_values[1], X_new_test)
shap.dependence_plot("CD4", shap_values[1], X_new_test, interaction_index=None, show=False)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
shap.plots.waterfall(exp[897][:, 0], max_display=10)
shap.initjs()
shap.force_plot(explanier.expected_value[1], shap_values[1], X_new_test)