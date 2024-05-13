import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import warnings

warnings.filterwarnings('always')

# If Not SMOTE
THRESHOLD=0.035
# If SMOTE
# THRESHOLD=0.5

def get_clf_eval(y_true, y_pred=None):
    print(confusion_matrix(y_true, y_pred))
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=1), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    print(f"정확도: {accuracy}, 정밀도: {precision}, 재현율: {recall}, F1: {f1}")
    return accuracy, precision, recall, f1

def eval_gini(y_true, y_pred):
    # 실제값과 예측값의 크기가 같은지 확인 (값이 다르면 오류 발생)
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]                      # 데이터 개수
    L_mid = np.linspace(1 / n_samples, 1, n_samples) # 대각선 값
    # 예측값에 대한 지니계수
    pred_order = y_true[y_pred.argsort()]               # y_pred 크기순으로 y_true 값 정렬
    L_pred = np.cumsum(pred_order) / np.sum(pred_order) # 로렌츠 곡선
    G_pred = np.sum(L_mid - L_pred)                     # 예측 값에 대한 지니계수
    # 예측이 완벽할 때 지니계수
    true_order = y_true[y_true.argsort()]               # y_true 크기순으로 y_true 값 정렬
    L_true = np.cumsum(true_order) / np.sum(true_order) # 로렌츠 곡선
    G_true = np.sum(L_mid - L_true)                     # 예측이 완벽할 때 지니계수
    # 정규화된 지니계수
    return G_pred / G_true

# LightGBM용 gini() 함수
def gini(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', eval_gini(labels, preds), True # 반환값

def custom_f1(y_true, y_pred):
    y_pred_clf = np.where(y_pred>=THRESHOLD, 1, 0)
    return f1_score(y_true, y_pred_clf)

class MLModel:
    def __init__(self)->None:
        self.origin_data = None
        self.model = None
        self.train = None
        self.test = None
        self.model_params = None
        self.random_state = 2024

    def __init__(self, model:lgb.LGBMRegressor|xgb.XGBRegressor|RandomForestRegressor|KNeighborsRegressor|SVR|LinearSVR|AdaBoostRegressor|GradientBoostingRegressor|DecisionTreeRegressor, 
                 model_params:dict=None, random_state:int=2024)->None:
        '''
        model argument can be
        1. SVR, LinearSVR
        2. DecisionTreeRegressor
        3. RandomForestRegressor
        4. AdaBoostRegressor
        5. GradientBoostingRegressor
        6. XGBRFRegressor
        7. LGBMRegressor
        8. KNeighborsRegressor
        9. lgb.LGBMRegressor
        10. xgb.XGBRegressor
        random_state is default 2024
        '''
        self.origin_data = None
        self.model = None
        self.train = None
        self.test = None
        self.model_params = model_params
        self.random_state = random_state
        if model_params:
            self.model = model(model_params)
        else:
            self.model = model()

    def load_data(self, file_path:str, validation_size:float=0.2)->pd.DataFrame:
        data = pd.read_csv(file_path)
        x = data.drop(columns=['target'])
        y = data['target']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=validation_size, random_state=self.random_state, stratify=y)

    def fit_model_with_Gridsearch(self, param_grid:dict, n_splits:int=5)->tuple[dict, float]:
        my_f1_scorer = make_scorer(custom_f1, greater_is_better=True)
        fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        if self.model.__module__.split(".")[0]=="sklearn":
            if self.model.__class__==RandomForestRegressor:
                model = RandomForestRegressor(random_state=self.random_state)
            elif self.model.__class__==AdaBoostRegressor:
                model = AdaBoostRegressor(random_state=self.random_state)
            elif self.model.__class__==KNeighborsRegressor:
                model = KNeighborsRegressor()
            elif self.model.__class__==GradientBoostingRegressor:
                model = GradientBoostingRegressor(random_state=self.random_state)
            elif self.model.__class__==SVR:
                model = SVR()
            elif self.model.__class__==LinearSVR:
                model = LinearSVR(random_state=self.random_state)
            elif self.model.__class__==DecisionTreeRegressor:
                model = DecisionTreeRegressor(random_state=self.random_state)
            
        # LGBMRegressor
        if self.model.__class__==lgb.LGBMRegressor:
            model = lgb.LGBMRegressor()
            
        # XGBRegressor
        if self.model.__class__==xgb.XGBRegressor:
            model = xgb.XGBRegressor()
            
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=fold, scoring=my_f1_scorer)
        grid_search.fit(self.x_train, self.y_train)
        return grid_search.best_params_, grid_search.best_score_

    def fit_model(self, model_params:dict, threshold:float, sampler:SMOTE|TomekLinks=None, n_splits:int=5):
        fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for idx, (train_idx, valid_idx) in enumerate(fold.split(self.x_train, self.y_train)):
            print('*'*40, f'폴드 {idx+1} / 폴드 {fold.n_splits}', '*'*40)
            x_train= self.x_train.iloc[train_idx]
            x_valid = self.x_train.iloc[valid_idx]
            y_train = self.y_train.iloc[train_idx]
            y_valid = self.y_train.iloc[valid_idx]
            if sampler:
                x_train, y_train = sampler().fit_resample(x_train, y_train)
            valid_pred_prob, test_pred_prob = None, None
            
            ## scikit-learn
            # 1. SVR, LinearSVR
            # 2. DecisionTreeRegressor
            # 3. RandomForestRegressor
            # 4. AdaBoostRegressor
            # 5. GradientBoostingRegressor
            # 6. XGBRFRegressor
            # 7. LGBMRegressor
            # 8. KNeighborsRegressor
            if self.model.__module__.split(".")[0]=="sklearn":
                if self.model.__class__==RandomForestRegressor:
                    self.model = RandomForestRegressor(**model_params, random_state=self.random_state)
                elif self.model.__class__==AdaBoostRegressor:
                    self.model = AdaBoostRegressor(**model_params, random_state=self.random_state)
                elif self.model.__class__==KNeighborsRegressor:
                    self.model = KNeighborsRegressor(**model_params)
                elif self.model.__class__==GradientBoostingRegressor:
                    self.model = GradientBoostingRegressor(**model_params, random_state=self.random_state)
                elif self.model.__class__==SVR:
                    self.model = SVR(**model_params)
                elif self.model.__class__==LinearSVR:
                    self.model = LinearSVR(**model_params, random_state=self.random_state)
                elif self.model.__class__==DecisionTreeRegressor:
                    self.model = DecisionTreeRegressor(**model_params, random_state=self.random_state)
                
                self.model.fit(x_train, y_train)
                valid_pred_prob = self.model.predict(x_valid)
                test_pred_prob = self.model.predict(self.x_test)
            
            # LGBMRegressor
            if self.model.__class__==lgb.LGBMRegressor:
                self.model = lgb.LGBMRegressor(**model_params)
                self.model.fit(x_train, y_train)
                valid_pred_prob = self.model.predict(x_valid)
                test_pred_prob = self.model.predict(self.x_test)
                '''
                dtrain = lgb.Dataset(x_train, y_train)
                dvalid = lgb.Dataset(x_valid, y_valid)
                dtest = lgb.Dataset(self.x_test)
                self.model = lgb.train(params=model_params,
                                       train_set=dtrain,
                                       num_boost_round=fit_params['num_boost_round'],
                                       valid_sets=dvalid,
                                       feval=gini,
                                       callbacks=[lgb.early_stopping(stopping_rounds=fit_params['stopping_rounds']),
                                                  lgb.log_evaluation(fit_params['fit_params'])])
                valid_pred_prob = self.model.predict(dvalid)
                test_pred_prob = self.model.predict(dtest)
                '''
                
            # XGBRegressor
            if self.model.__class__==xgb.XGBRegressor:
                '''
                dtrain = xgb.DMatrix(x_train, y_train)
                dvalid = xgb.DMatrix(x_valid, y_valid)
                dtest = xgb.DMatrix(self.x_test)
                self.model = xgb.train(params=model_params, 
                                       dtrain=dtrain,
                                       num_boost_round=fit_params['num_boost_round'],
                                       evals=[(dvalid, 'valid')],
                                       maximize=True,
                                       feval=gini,
                                       early_stopping_rounds=fit_params['early_stopping_rounds'],
                                       verbose_eval=fit_params['verbose_eval'])
                best_iter = self.model.best_iteration
                valid_pred_prob = self.model.predict(dvalid, iteration_range=(0, best_iter))
                test_pred_prob = self.model.predict(dtest, iteration_range=(0, best_iter))
                '''
                model = xgb.XGBRegressor(**model_params)
                model.fit(x_train, y_train)
                valid_pred_prob = model.predict(x_valid)
                test_pred_prob = model.predict(self.x_test)

            valid_pred = np.where(valid_pred_prob>=threshold, 1, 0)
            test_pred = np.where(test_pred_prob>=threshold, 1, 0)
            valid_accuracy, valid_precision, valid_recall, valid_f1 = get_clf_eval(y_true=y_valid, y_pred=valid_pred)
            test_accuracy, test_precision, test_recall, test_f1 = get_clf_eval(y_true=self.y_test, y_pred=test_pred)
            accuracy+=valid_accuracy/fold.n_splits
            precision+=valid_precision/fold.n_splits
            recall+=valid_recall/fold.n_splits
            f1+=valid_f1/fold.n_splits
        print(f'평균 validation set 결과')
        print(f"정확도: {accuracy}, 정밀도: {precision}, 재현율: {recall}, F1: {f1}")
        print(f'Test set 결과')
        print(f"정확도: {test_accuracy}, 정밀도: {test_precision}, 재현율: {test_recall}, F1: {test_f1}")
        print(test_pred_prob.max())

    def get_model(self):
        return self.model
    
if __name__=="__main__":
    # model argument can be
    # 1. SVR, LinearSVR
    # 2. DecisionTreeRegressor
    # 3. RandomForestRegressor
    # 4. AdaBoostRegressor
    # 5. GradientBoostingRegressor
    # 6. XGBRFRegressor
    # 7. LGBMRegressor
    # 8. KNeighborsRegressor
    # 9. lgb.LGBMRegressor
    # 10. xgb.XGBRegressor
    model = MLModel(model=AdaBoostRegressor)
    
    # after assign MLModel Class, load_data from local filesystem
    model.load_data('data/train.csv')
    
    # Need To make Search
    best_params, best_scores = model.fit_model_with_Gridsearch(param_grid = {'learning_rate': (0.005, 0.01),
                    'n_estimators' : (30, 60, 90)})
    # max_params = best_params
    print(best_params, best_scores)
    # set parameters to make Model
    '''
    best_params = {
        'colsample_bytree': 1.0,
        'gamma': 8.75239267900182,
        'max_depth': 7,
        'min_child_weight': 7.0,
        'reg_alpha': 7.0,
        'reg_lambda': 1.282397381262159,
        'scale_pos_weight': 1.4538566545153988,
        'subsample': 0.7102431062533778,
        'objective': 'binary:logistic',
        'learning_rate': 0.02,
        'random_state': 1991
    }
    '''
    # fit model by parameters and set threshold to see metric scores
    # Can also use Sampler Like TomekLinks or SMOTE in here, But Need Much Time
    for threshold in np.arange(0.4, 0.6, 0.01):
        model.fit_model(model_params=best_params, threshold=threshold, n_splits=5, sampler=SMOTE)
