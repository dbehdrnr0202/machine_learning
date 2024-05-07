import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

def get_clf_eval(y_test, pred=None, pred_proba_prob=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba_prob)
    print("오차 행렬")
    print(confusion)
    print(f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

def gini(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', eval_gini(labels, preds), True # 반환값

def eval_gini(y_true, y_pred):
    # 실제값과 예측값의 크기가 같은지 확인 (값이 다르면 오류 발생)
    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]                      # 데이터 개수
    L_mid = np.linspace(1 / n_samples, 1, n_samples) # 대각선 값

    # 1) 예측값에 대한 지니계수
    pred_order = y_true[y_pred.argsort()] # y_pred 크기순으로 y_true 값 정렬
    L_pred = np.cumsum(pred_order) / np.sum(pred_order) # 로렌츠 곡선
    G_pred = np.sum(L_mid - L_pred)       # 예측 값에 대한 지니계수

    # 2) 예측이 완벽할 때 지니계수
    true_order = y_true[y_true.argsort()] # y_true 크기순으로 y_true 값 정렬
    L_true = np.cumsum(true_order) / np.sum(true_order) # 로렌츠 곡선
    G_true = np.sum(L_mid - L_true)       # 예측이 완벽할 때 지니계수

    # 정규화된 지니계수
    return G_pred / G_true

class BaseModel:
    def __init__(self) -> None:
        self.data_path = ""
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.submission = pd.DataFrame()
        self.all_data = pd.DataFrame()
        self.model = None

    def load_data(self, data_path):
        # 데이터 경로
        data_path = 'data/'
        self.train = pd.read_csv(data_path + 'train.csv', index_col='id')
        self.test = pd.read_csv(data_path + 'test.csv', index_col='id')
        self.submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')
        self.all_data = pd.concat([self.train, self.test], ignore_index=True)
        self.all_data = self.all_data.drop('target', axis=1) # 타깃값 제거
    
    def preprocessing(self):
        all_features = self.all_data.columns # 전체 피처
        cat_features = [feature for feature in all_features if 'cat' in feature] 
        onehot_encoder = OneHotEncoder() # 원-핫 인코더 객체 생성
        # 인코딩
        encoded_cat_matrix = onehot_encoder.fit_transform(self.all_data[cat_features]) 

        encoded_cat_matrix
        # 추가로 제거할 피처
        drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 
                        'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

        # '1) 명목형 피처, 2) calc 분류의 피처, 3) 추가 제거할 피처'를 제외한 피처
        remaining_features = [feature for feature in all_features 
                            if ('cat' not in feature and 
                                'calc' not in feature and 
                                feature not in drop_features)]
        all_data_sprs = sparse.hstack([sparse.csr_matrix(self.all_data[remaining_features]),
                               encoded_cat_matrix],
                              format='csr')
        num_train = len(self.train) # 훈련 데이터 개수
        # 훈련 데이터와 테스트 데이터 나누기
        self.X = all_data_sprs[:num_train]
        self.X_test = all_data_sprs[num_train:]

        self.y = self.train['target'].values

    def modified_preprocessing(self):
        all_features = self.all_data.columns # 전체 피처
        cat_features = [feature for feature in all_features if 'cat' in feature] 
        onehot_encoder = OneHotEncoder(drop='first') # 불필요한 차원 증가 제거

        encoded_cat_matrix = onehot_encoder.fit_transform(self.all_data[cat_features]) 

        # 추가로 제거할 피처
        drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 
                        'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

        # '1) 명목형 피처, 2) calc 분류의 피처, 3) 추가 제거할 피처'를 제외한 피처
        remaining_features = [feature for feature in all_features 
                            if ('cat' not in feature and 
                                'calc' not in feature and 
                                feature not in drop_features)]
        all_data_sprs = sparse.hstack([sparse.csr_matrix(self.all_data[remaining_features]),
                               encoded_cat_matrix],
                              format='csr')
        num_train = len(self.train) # 훈련 데이터 개수

        # 훈련 데이터와 테스트 데이터 나누기
        self.X = all_data_sprs[:num_train]
        self.X_test = all_data_sprs[num_train:]
        self.y = self.train['target'].values


    def fit_model(self, params:dict, train_params:dict):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1991)
        # OOF 방식으로 훈련된 모델로 검증 데이터 타깃값을 예측한 확률을 담을 1차원 배열
        oof_val_preds = np.zeros(self.X.shape[0]) 
        # OOF 방식으로 훈련된 모델로 테스트 데이터 타깃값을 예측한 확률을 담을 1차원 배열
        oof_test_preds = np.zeros(self.X_test.shape[0])
        for idx, (train_idx, valid_idx) in enumerate(folds.split(self.X, self.y)):
            # 각 폴드를 구분하는 문구 출력
            print('#'*40, f'폴드 {idx+1} / 폴드 {folds.n_splits}', '#'*40)
            
            # 훈련용 데이터, 검증용 데이터 설정 
            X_train, y_train = self.X[train_idx], self.y[train_idx] # 훈련용 데이터
            X_valid, y_valid = self.X[valid_idx], self.y[valid_idx] # 검증용 데이터

            # LightGBM 전용 데이터셋 생성 
            dtrain = lgb.Dataset(X_train, y_train) # LightGBM 전용 훈련 데이터셋
            dvalid = lgb.Dataset(X_valid, y_valid) # LightGBM 전용 검증 데이터셋

            # LightGBM 모델 훈련 
            self.model = lgb.train(params=params,        # 훈련용 하이퍼파라미터
                                train_set=dtrain,     # 훈련 데이터셋
                                num_boost_round=train_params['num_boost_round'], # 부스팅 반복 횟수
                                valid_sets=dvalid,    # 성능 평가용 검증 데이터셋
                                feval=train_params['feval'],           # 검증용 평가지표
                                callbacks=train_params['callbacks'])                  # 100번째마다 점수 출력
            
            # 테스트 데이터를 활용해 OOF 예측
            oof_test_preds += self.model.predict(self.X_test)/folds.n_splits
            
            

            # 모델 성능 평가를 위한 검증 데이터 타깃값 예측
            oof_val_preds[valid_idx] += self.model.predict(X_valid)
            
            ###
            valid_pred_prob = self.model.predict(X_valid)
            valid_pred = np.where(valid_pred_prob>0.5, 1, 0)
            get_clf_eval(y_test=y_valid, pred=valid_pred, pred_proba_prob=valid_pred_prob)
            ###
            
            # 검증 데이터 예측 확률에 대한 정규화 지니계수 
            gini_score = eval_gini(y_valid, oof_val_preds[valid_idx])
            print(f'폴드 {idx+1} 지니계수 : {gini_score}\n')
            self.submission['target'] = oof_test_preds
    
    def save_submission(self, file_path):    
        self.submission.to_csv(file_path)
            
if __name__=="__main__":
    model = BaseModel()
    model.load_data(data_path="data/")
    model.preprocessing()
    model.fit_model(params = {'objective': 'binary',
          'learning_rate': 0.01,
          'force_row_wise': True,
          'random_state': 0},
          train_params={'num_boost_round':1000,
                        'feval':gini,
                        'callbacks':[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]})
    model.save_submission("output/base_submission2.csv")