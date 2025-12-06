import warnings
import pandas as pd  # 用于数据处理的pandas库
import numpy as np  # 用于数学运算的numpy库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # 导入训练集拆分和网格搜索工具
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 导入随机森林和梯度提升回归模型
from sklearn.neural_network import MLPRegressor  # 导入多层感知机回归模型
from sklearn.linear_model import BayesianRidge  # 导入贝叶斯岭回归模型
from sklearn.svm import SVR  # 导入支持向量回归模型
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入模型评估指标
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # 用于图形输出的库
import xgboost as xgb  # 导入XGBoost库
from scipy.stats import t  # 导入t分布相关功能
from sklearn.inspection import PartialDependenceDisplay  # 用于绘制部分依赖图
import shap  # 导入SHAP库用于模型解释
import seaborn as sns  # 导入seaborn库
import scipy.stats as stats  # 导入scipy统计模块
from scipy.optimize import fsolve  # 导入scipy的方程求解器
try:
    import h5py  # 用于保存.h5格式
except ImportError:
    print("错误: 未找到 h5py 模块。请运行以下命令安装: pip install h5py")
    print("或者，如果您使用的是 conda 环境，请运行: conda install h5py")
    raise
import json  # 用于处理JSON数据
import pickle  # 用于序列化模型
import os  # 用于文件操作

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 自定义麻雀搜索算法实现
class SparrowSearchAlgorithm:
    def __init__(self, fit_func, lb, ub, population_size=30, epoch=50, elite_ratio=0.2, follower_ratio=0.8):
        self.fit_func = fit_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.population_size = population_size
        self.epoch = epoch
        self.elite_size = int(population_size * elite_ratio)
        self.follower_size = population_size - self.elite_size
        self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.array([self.fit_func(ind) for ind in self.positions])
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.positions[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

    def update_positions(self):
        # 选择精英和追随者
        elite_indices = self.fitness.argsort()[:self.elite_size]
        followers_indices = self.fitness.argsort()[self.elite_size:]
        elites = self.positions[elite_indices]
        followers = self.positions[followers_indices]

        # 更新精英位置
        for i in range(self.elite_size):
            rand = np.random.uniform(0, 1, self.dim)
            self.positions[elite_indices[i]] = elites[i] + rand * (self.best_position - elites[i])

        # 更新追随者位置
        for i in range(self.follower_size):
            rand = np.random.uniform(-1, 1, self.dim)
            self.positions[followers_indices[i]] = followers[i] + rand * (self.best_position - followers[i])

        # 确保位置在边界内
        self.positions = np.clip(self.positions, self.lb, self.ub)

    def optimize(self):
        for epoch in range(self.epoch):
            self.update_positions()
            current_fitness = np.array([self.fit_func(ind) for ind in self.positions])
            for i in range(self.population_size):
                if current_fitness[i] < self.fitness[i]:
                    self.fitness[i] = current_fitness[i]
                    self.positions[i] = self.positions[i]
                    if self.fitness[i] < self.best_fitness:
                        self.best_fitness = self.fitness[i]
                        self.best_position = self.positions[i].copy()
            print(f"Epoch {epoch+1}/{self.epoch}, Best Fitness: {self.best_fitness}")
        return self.best_position, self.best_fitness


def save_xgboost_to_h5(model, filepath, feature_names=None):
    """
    将XGBoost模型保存为.h5格式
    
    参数:
    model: 训练好的XGBoost模型
    filepath: 保存路径（.h5文件）
    feature_names: 特征名称列表（可选）
    """
    import os
    import tempfile
    
    try:
        # 获取XGBoost的底层booster
        booster = model.get_booster()
        
        # 获取模型配置
        config = json.loads(booster.save_config())
        
        # 获取特征名称
        if feature_names is None:
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        # 使用临时文件保存模型
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_json:
            temp_json_path = temp_json.name
            booster.save_model(temp_json_path)
        
        # 读取JSON模型文件（明确指定UTF-8编码）
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        
        # 创建HDF5文件
        with h5py.File(filepath, 'w') as f:
            # 保存模型类型
            f.attrs['model_type'] = 'XGBoost'
            f.attrs['model_version'] = xgb.__version__
            
            # 保存模型参数
            params_group = f.create_group('parameters')
            for key, value in model.get_params().items():
                try:
                    if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                        params_group.attrs[key] = value
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        params_group.create_dataset(key, data=np.array(value))
                    elif value is None:
                        params_group.attrs[key] = 'None'
                    else:
                        params_group.attrs[key] = str(value)
                except Exception as e:
                    print(f"警告: 无法保存参数 {key}: {e}")
                    continue
            
            # 保存特征名称
            if feature_names is not None:
                feature_names_group = f.create_group('feature_names')
                for i, name in enumerate(feature_names):
                    feature_names_group.create_dataset(str(i), data=str(name).encode('utf-8'))
            
            # 保存模型结构（JSON格式的字符串）
            model_group = f.create_group('model')
            model_group.create_dataset('json_model', data=json.dumps(model_json).encode('utf-8'))
            
            # 保存模型配置
            config_group = f.create_group('config')
            config_str = json.dumps(config)
            config_group.create_dataset('config_json', data=config_str.encode('utf-8'))
            
            # 保存特征重要性
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                f.create_dataset('feature_importances', data=importance)
            
            # 保存模型的其他属性
            if hasattr(model, 'n_features_in_'):
                f.attrs['n_features_in_'] = int(model.n_features_in_)
            if hasattr(model, 'n_estimators'):
                f.attrs['n_estimators'] = int(model.n_estimators)
        
        # 删除临时JSON文件
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        
        print(f"模型已成功保存为: {filepath}")
        
    except Exception as e:
        print(f"保存模型时出错: {e}")
        # 如果H5保存失败，尝试使用pickle作为备选方案
        import pickle
        pickle_path = filepath.replace('.h5', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"已使用pickle格式保存模型为: {pickle_path}")
        raise


def load_xgboost_from_h5(filepath):
    """
    从.h5文件加载XGBoost模型
    
    参数:
    filepath: .h5文件路径
    
    返回:
    恢复的XGBoost模型
    """
    with h5py.File(filepath, 'r') as f:
        # 读取模型类型
        model_type = f.attrs.get('model_type', b'XGBoost').decode('utf-8') if isinstance(f.attrs.get('model_type'), bytes) else f.attrs.get('model_type')
        
        if model_type != 'XGBoost':
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 读取模型参数
        params = {}
        if 'parameters' in f:
            params_group = f['parameters']
            for key in params_group.attrs:
                value = params_group.attrs[key]
                # 处理字符串 'None' 转换为实际的 None
                if isinstance(value, str) and value == 'None':
                    value = None
                # 处理 verbosity 参数，确保是整数
                if key == 'verbosity' and value is not None:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = 1  # 默认值
                params[key] = value
        
        # 读取特征名称
        feature_names = None
        if 'feature_names' in f:
            feature_names_group = f['feature_names']
            feature_names = [feature_names_group[str(i)][()].decode('utf-8') 
                           for i in range(len(feature_names_group))]
        
        # 读取模型JSON
        if 'model' in f and 'json_model' in f['model']:
            model_json_str = f['model']['json_model'][()].decode('utf-8')
            model_json = json.loads(model_json_str)
            
            # 创建临时JSON文件（明确指定UTF-8编码）
            import tempfile
            import os
            temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            json.dump(model_json, temp_json, ensure_ascii=False)
            temp_json.close()
            
            # 从JSON文件加载模型
            # 清理参数，移除 None 值或设置默认值
            clean_params = {}
            for key, value in params.items():
                if key == 'verbosity' and (value is None or value == 'None'):
                    clean_params[key] = 1  # 默认 verbosity
                elif value is None or value == 'None':
                    # 对于其他 None 参数，跳过（使用 XGBoost 默认值）
                    continue
                else:
                    clean_params[key] = value
            
            model = xgb.XGBRegressor(**clean_params)
            booster = xgb.Booster()
            booster.load_model(temp_json.name)
            model._Booster = booster
            
            # 删除临时文件
            os.unlink(temp_json.name)
            
            # 存储特征名称作为模型的属性（feature_names_in_ 是只读的）
            # 使用私有属性来存储，以便后续使用
            if feature_names is not None:
                # 尝试通过 booster 设置特征名称
                try:
                    # 如果 booster 支持设置特征名称
                    if hasattr(booster, 'feature_names'):
                        booster.feature_names = feature_names
                except:
                    pass
                # 将特征名称存储为模型的私有属性
                model._feature_names = np.array(feature_names)
            else:
                # 如果没有保存的特征名称，尝试从 booster 获取
                try:
                    if hasattr(booster, 'feature_names') and booster.feature_names:
                        model._feature_names = np.array(booster.feature_names)
                except:
                    pass
            
            return model
        else:
            raise ValueError("H5文件中缺少模型数据")


# ============================================================================
# Configuration: Set to True to train a new model, False to skip training
# ============================================================================
TRAIN_MODEL = False  # Set to True to train a new model, False to skip training
MODEL_PATH = "Xgboost.h5"  # Path to the existing model file

# ============================================================================
# Main execution block
# ============================================================================
if __name__ == "__main__":
    # Check if model already exists and training is disabled
    if not TRAIN_MODEL:
        if os.path.exists(MODEL_PATH):
            print(f"✓ Model file '{MODEL_PATH}' already exists.")
            print("Training skipped. To train a new model, set TRAIN_MODEL = True")
            print("You can now use the inference script or app.py to make predictions.")
        else:
            print(f"⚠ Warning: Model file '{MODEL_PATH}' not found.")
            print("Set TRAIN_MODEL = True to train a new model.")
    else:
        # Training mode - load data and train model
        print("Starting model training...")
        # Load data
        df1 = pd.read_excel(r"C:\Users\user\Desktop\1\T25.xlsx", sheet_name='总', index_col=0)

        # 定义数据集列表
        datasets = [
            ("T25 fine droplet size", df1),
        ]

        # 初始化模型及其参数网格
        models = {
            "XGBoost": (xgb.XGBRegressor(objective='reg:squarederror', random_state=42), {
                'n_estimators': [150, 250],
                'learning_rate': [0.01, 0.02],
                'max_depth': [4, 6],
                'subsample': [0.6, 0.8],
                'colsample_bytree': [0.6, 0.8],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [2, 3]
            })
        }
        # 初始化一个空的 DataFrame 来存储所有指标和最佳参数
        all_metrics = pd.DataFrame(
            columns=["Dataset", "Model", "R2 Score (Test)", "MSE (Test)", "RMSE (Test)", "MAE (Test)", "MAPE (Test)",
                     "RSE (Test)", "SD (Test)", "Pbias (Test)", "IA (Test)", "U95 (Test)", "Best Parameters",
                     "R2 Score (Train)", "MSE (Train)", "RMSE (Train)", "MAE (Train)", "MAPE (Train)", "RSE (Train)",
                     "SD (Train)", "Pbias (Train)", "IA (Train)", "U95 (Train)", "R2 Score (Full)", "MSE (Full)", "RMSE (Full)",
                     "MAE (Full)", "MAPE (Full)", "RSE (Full)", "SD (Full)", "Pbias (Full)", "IA (Full)", "U95 (Full)"])

        # 初始化 DataFrame 来存储每个模型的最佳搜索结果
        best_search_results = pd.DataFrame(columns=["Dataset", "Model", "Best Parameters", "Best Score"])

        # U95 计算的置信水平
        confidence = 0.95

        # 训练和评估模型
        for dataset_name, df in datasets:
            y_std = float(np.std(df['Drift Rate (%)'], ddof=1))
            print(f"Dataset: {dataset_name}, Real Data Standard Deviation: {y_std}")

            X = df.drop(columns='Drift Rate (%)')  # 特征矩阵
            y = df['Drift Rate (%)']  # 标签向量

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            metrics = []  # 存储模型评估指标
            model_predictions = {"Real Values": y_test}

            for name, (model, param_grid) in models.items():
                best_params = "N/A"
                best_score = "N/A"

                # 使用自定义SSA进行超参数优化
                if param_grid:
                    # 将参数网格转换为SSA的搜索范围
                    param_names = list(param_grid.keys())
                    lb = []
                    ub = []
                    # 定义参数类型，以便在SSA中进行处理
                    param_types = []

                    for param in param_names:
                        values = param_grid[param]
                        if isinstance(values[0], int):
                            lb.append(min(values))
                            ub.append(max(values))
                            param_types.append('int')
                        elif isinstance(values[0], float):
                            lb.append(min(values))
                            ub.append(max(values))
                            param_types.append('float')
                        elif isinstance(values[0], bool):
                            lb.append(0)
                            ub.append(1)
                            param_types.append('bool')
                        else:
                            # 对于分类参数，需要进行编码
                            unique_values = list(set(values))
                            lb.append(0)
                            ub.append(len(unique_values) - 1)
                            param_types.append(unique_values)  # 保存可能的取值列表

                    # 定义目标函数
                    def objective_function(solution):
                        params = {}
                        for i, value in enumerate(solution):
                            if param_types[i] == 'int':
                                params[param_names[i]] = int(round(value))
                            elif param_types[i] == 'float':
                                params[param_names[i]] = value
                            elif param_types[i] == 'bool':
                                params[param_names[i]] = bool(round(value))
                            else:
                                # 分类参数
                                index = int(round(value))
                                index = max(0, min(index, len(param_types[i]) - 1))
                                params[param_names[i]] = param_types[i][index]
                        model.set_params(**params)
                        # 使用交叉验证的负R2得分作为目标函数
                        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        return -np.mean(scores)  # 最小化负的R2得分，即最大化R2得分

                    # 设置SSA的参数
                    problem = {
                        "fit_func": objective_function,
                        "lb": lb,
                        "ub": ub,
                        "minmax": "min",
                        "verbose": True,
                    }

                    # 初始化并运行SSA优化器
                    epoch = 10
                    population_size = 30

                    # Initialize and run the SSA optimizer
                    optimizer = SparrowSearchAlgorithm(fit_func=objective_function,
                                                       lb=lb,
                                                       ub=ub,
                                                       population_size=population_size,
                                                       epoch=epoch)
                    best_position, best_fitness = optimizer.optimize()

                    # 提取最佳超参数
                    best_params = {}
                    for i, value in enumerate(best_position):
                        if param_types[i] == 'int':
                            best_params[param_names[i]] = int(round(value))
                        elif param_types[i] == 'float':
                            best_params[param_names[i]] = value
                        elif param_types[i] == 'bool':
                            best_params[param_names[i]] = bool(round(value))
                        else:
                            index = int(round(value))
                            index = max(0, min(index, len(param_types[i]) - 1))
                            best_params[param_names[i]] = param_types[i][index]

                    best_score = -best_fitness
                    print(f"最佳超参数: {best_params}")
                    print(f"最佳交叉验证R2得分: {best_score:.4f}")
                    
                    # 使用最佳参数创建并训练模型
                    best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                    best_model.set_params(**best_params)
                    best_model.fit(X_train, y_train)

                    # 保存最佳结果
                    best_search_results = pd.concat(
                        [best_search_results, pd.DataFrame([[dataset_name, name, str(best_params), best_score]],
                                                           columns=best_search_results.columns)], ignore_index=True)

                else:
                    print(f"未使用超参数优化，使用默认参数训练模型")
                    best_model = model.fit(X_train, y_train)

                # 保存模型为.h5格式
                print(f"\n开始保存模型...")
                model_h5_path = f"xgboost_model_{dataset_name.replace(' ', '_')}.h5"
                try:
                    save_xgboost_to_h5(best_model, model_h5_path, feature_names=list(X.columns))
                    print(f"✓ 模型已成功保存为: {model_h5_path}\n")
                except Exception as e:
                    print(f"✗ 保存模型时出错: {e}")
                    raise

                # 创建SHAP解释器并计算SHAP值
                explainer = shap.Explainer(best_model)
                shap_values = explainer(X_test)

                # 从多个样本的SHAP值中移除'Distance'特征
                num_samples = min(1, len(X_test))  # 确保不超过测试集大小
                feature_names = X_test.columns
                if 'Distance' in feature_names:
                    distance_index = list(feature_names).index('Distance')

                    for sample_index in range(num_samples):
                        try:
                            shap_values_filtered = shap_values.values[sample_index].copy()
                            shap_values_filtered = np.delete(shap_values_filtered, distance_index)

                            X_test_filtered = X_test.iloc[sample_index].copy()
                            X_test_filtered = X_test_filtered.drop(labels=['Distance'])

                            # 设置图形属性
                            plt.rcParams.update({'font.size': 8})
                            plt.figure(figsize=(15, 6))
                            # 显示移除 'Distance' 特征后的 SHAP force plot 并保存为 HTML 文件
                            shap_plot_filtered = shap.force_plot(explainer.expected_value, shap_values_filtered, X_test_filtered)

                            # 将 SHAP force plot 保存为 HTML 文件
                            shap.save_html(f"shap_force_plot_sample_{sample_index}.html", shap_plot_filtered)

                            # 显示移除'Distance'特征后的SHAP force plot并保存为PNG文件
                            shap.force_plot(explainer.expected_value, shap_values_filtered, X_test_filtered, matplotlib=True)
                            plt.savefig(f"shap_force_plot_sample_{sample_index}.png")
                            plt.close()
                        except Exception as e:
                            print(f"处理样本 {sample_index} 时出错: {e}")
                            continue