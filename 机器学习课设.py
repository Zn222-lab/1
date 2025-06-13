import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import os
import gc
from datetime import datetime
import psutil
import warnings

warnings.filterwarnings('ignore')

# 打印内存使用情况
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)
    print(f"当前内存使用: {mem:.2f} GB")

# 1. 数据加载优化
print("开始加载数据...")
data_dir = os.path.join(os.getcwd(), "data_format1")

file_paths = {
    "user_info": os.path.join(data_dir, "user_info_format1.csv"),
    "user_log": os.path.join(data_dir, "user_log_format1.csv"),
    "train": os.path.join(data_dir, "train_format1.csv"),
    "test": os.path.join(data_dir, "test_format1.csv")
}

# 加载数据函数
def load_data():
    # 用户画像
    user_info = pd.read_csv(file_paths["user_info"], dtype={
        'user_id': 'int32',
        'age_range': 'float32',  # 先读为浮点数
        'gender': 'float32'  # 先读为浮点数
    }).fillna({'age_range': -1, 'gender': 2}).astype({
        'age_range': 'int8',
        'gender': 'int8'
    })

    # 训练测试数据
    train = pd.read_csv(file_paths["train"], dtype={
        'user_id': 'int32',
        'merchant_id': 'int32',
        'label': 'int8'
    })
    test = pd.read_csv(file_paths["test"], dtype={
        'user_id': 'int32',
        'merchant_id': 'int32'
    })

    # 用户日志
    chunks = []
    for chunk in pd.read_csv(file_paths["user_log"], chunksize=1000000, dtype={
        'user_id': 'int32',
        'item_id': 'int32',
        'cat_id': 'int32',
        'brand_id': 'float32',  # 先读为浮点数，处理可能的NA值
        'seller_id': 'int32',
        'time_stamp': 'str',
        'action_type': 'int8'
    }):
        chunk.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
        chunk['time_stamp'] = chunk['time_stamp'].str.zfill(4)
        chunk['month'] = chunk['time_stamp'].str[:2].astype('int8')
        chunk['day'] = chunk['time_stamp'].str[2:].astype('int8')

        # 处理可能的NA值并转换类型
        chunk['brand_id'] = chunk['brand_id'].fillna(-1).astype('int32')

        chunks.append(chunk)

    user_log = pd.concat(chunks, ignore_index=True)
    return user_info, train, test, user_log


user_info, train_data, test_data, user_log = load_data()
print("数据加载完成!")
print_memory_usage()


# 2. 高级特征工程
def create_advanced_features(df, log_df):
    print("创建高级特征...")

    # 基础特征合并
    df = pd.merge(df, user_info, on='user_id', how='left')
    df['age_range'] = df['age_range'].fillna(-1).astype('int8')
    df['gender'] = df['gender'].fillna(2).astype('int8')

    # 1. 用户-商家行为特征
    print("计算用户-商家行为特征...")
    user_merchant = log_df.groupby(['user_id', 'merchant_id']).agg({
        'action_type': [
            ('total_actions', 'count'),
            ('clicks', lambda x: (x == 0).sum()),
            ('add_to_cart', lambda x: (x == 1).sum()),
            ('purchases', lambda x: (x == 2).sum()),
            ('favorites', lambda x: (x == 3).sum()),
            ('action_mean', 'mean'),
            ('action_std', 'std')
        ],
        'item_id': [('unique_items', 'nunique')],
        'cat_id': [('unique_cats', 'nunique')],
        'brand_id': [('unique_brands', 'nunique')]
    }).reset_index()
    user_merchant.columns = ['_'.join(col).strip() for col in user_merchant.columns.values]
    user_merchant.rename(columns={'user_id_': 'user_id', 'merchant_id_': 'merchant_id'}, inplace=True)

    # 2. 时间相关特征
    print("计算时间特征...")
    time_features = log_df.groupby(['user_id', 'merchant_id']).agg({
        'month': [('month_min', 'min'), ('month_max', 'max')],
        'day': [('day_min', 'min'), ('day_max', 'max')]
    }).reset_index()
    time_features.columns = ['_'.join(col).strip() for col in time_features.columns.values]
    time_features.rename(columns={'user_id_': 'user_id', 'merchant_id_': 'merchant_id'}, inplace=True)

    # 3. 用户全局特征
    print("计算用户全局特征...")
    user_features = log_df.groupby('user_id').agg({
        'action_type': [
            ('user_total_actions', 'count'),
            ('user_clicks', lambda x: (x == 0).sum()),
            ('user_purchases', lambda x: (x == 2).sum()),
            ('user_action_mean', 'mean')
        ],
        'merchant_id': [('user_unique_merchants', 'nunique')]
    }).reset_index()
    user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
    user_features.rename(columns={'user_id_': 'user_id'}, inplace=True)

    # 4. 商家全局特征
    print("计算商家全局特征...")
    merchant_features = log_df.groupby('merchant_id').agg({
        'action_type': [
            ('merchant_total_actions', 'count'),
            ('merchant_clicks', lambda x: (x == 0).sum()),
            ('merchant_purchases', lambda x: (x == 2).sum()),
            ('merchant_action_mean', 'mean')
        ],
        'user_id': [('merchant_unique_users', 'nunique')]
    }).reset_index()
    merchant_features.columns = ['_'.join(col).strip() for col in merchant_features.columns.values]
    merchant_features.rename(columns={'merchant_id_': 'merchant_id'}, inplace=True)

    # 合并所有特征
    print("合并特征...")
    df = pd.merge(df, user_merchant, on=['user_id', 'merchant_id'], how='left')
    df = pd.merge(df, time_features, on=['user_id', 'merchant_id'], how='left')
    df = pd.merge(df, user_features, on='user_id', how='left')
    df = pd.merge(df, merchant_features, on='merchant_id', how='left')

    # 5. 衍生特征
    print("创建衍生特征...")
    # 行为比例特征
    df['click_ratio'] = (df['clicks_action_type'] / df['total_actions_action_type']).fillna(0)
    df['purchase_ratio'] = (df['purchases_action_type'] / df['total_actions_action_type']).fillna(0)
    df['cart_ratio'] = (df['add_to_cart_action_type'] / df['total_actions_action_type']).fillna(0)

    # 用户-商家互动特征
    df['user_merchant_interaction'] = (df['total_actions_action_type'] / df['user_total_actions_action_type']).fillna(0)
    df['user_merchant_purchase_ratio'] = (df['purchases_action_type'] / df['user_purchases_action_type']).fillna(0)

    # 时间跨度特征
    df['month_span'] = df['month_max_month'] - df['month_min_month']
    df['day_span'] = df['day_max_day'] - df['day_min_day']

    # 商家受欢迎程度
    df['merchant_popularity'] = (df['merchant_total_actions_action_type'] / df['merchant_unique_users_user_id']).fillna(
        0)

    # 用户活跃度
    df['user_activity'] = (df['user_total_actions_action_type'] / df['user_unique_merchants_merchant_id']).fillna(0)

    # 填充缺失值
    print("处理缺失值...")
    action_cols = [col for col in df.columns if 'actions' in col or 'clicks' in col or 'purchases' in col]
    for col in action_cols:
        df[col] = df[col].fillna(0)

    ratio_cols = [col for col in df.columns if 'ratio' in col]
    for col in ratio_cols:
        df[col] = df[col].fillna(0).astype('float32')

    # 优化内存
    print("优化内存...")
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    return df


# 创建特征
print("为训练数据创建特征...")
train_data = create_advanced_features(train_data, user_log)
print("为测试数据创建特征...")
test_data = create_advanced_features(test_data, user_log)

# 释放内存
del user_log, user_info
gc.collect()

# 3. 特征选择
print("特征选择...")
features = [
    # 基础特征
    'age_range', 'gender',

    # 用户-商家行为
    'total_actions_action_type', 'clicks_action_type',
    'add_to_cart_action_type', 'purchases_action_type',
    'favorites_action_type', 'action_mean_action_type',
    'unique_items_item_id', 'unique_cats_cat_id',
    'unique_brands_brand_id',

    # 时间特征
    'month_min_month', 'month_max_month',
    'day_min_day', 'day_max_day',

    # 用户特征
    'user_total_actions_action_type', 'user_clicks_action_type',
    'user_purchases_action_type', 'user_unique_merchants_merchant_id',

    # 商家特征
    'merchant_total_actions_action_type', 'merchant_clicks_action_type',
    'merchant_purchases_action_type', 'merchant_unique_users_user_id',

    # 衍生特征
    'click_ratio', 'purchase_ratio', 'cart_ratio',
    'user_merchant_interaction', 'user_merchant_purchase_ratio',
    'month_span', 'day_span', 'merchant_popularity', 'user_activity'
]

# 4. 模型训练优化
print("准备训练数据...")
X = train_data[features]
y = train_data['label'].astype('int8')
X_test = test_data[features]

# 定义优化的LightGBM参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'max_depth': 7,
    'min_child_samples': 100,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'max_bin': 255,
    'min_data_in_leaf': 50,
    'cat_smooth': 10,
    'early_stopping_round': 50,
    'boost_from_average': True
}

# 使用5折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(len(X_test))
feature_importance = pd.DataFrame()

print("开始交叉验证训练...")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Fold {fold_ + 1}")
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

    model = lgb.train(
        params,
        trn_data,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        num_boost_round=2000,
        early_stopping_rounds=100
    )

    # 保存特征重要性
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold_ + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    # 预测测试集
    predictions += model.predict(X_test) / folds.n_splits

# 5. 生成提交文件
submission = test_data[['user_id', 'merchant_id']].copy()
submission['prob'] = predictions
submission.columns = ['user_id', 'merchant_id', 'prob']
submission['prob'] = submission['prob'].clip(0, 1)

# 保存结果
submission_path = os.path.join(os.getcwd(), "improved_prediction.csv")
submission.to_csv(submission_path, index=False)
print(f'提交文件已保存为: {submission_path}')

# 分析特征重要性
feature_importance = feature_importance.groupby("feature")["importance"].mean().sort_values(ascending=False)
print("\nTop 20重要特征:")
print(feature_importance.head(20))

print("所有处理完成!")
