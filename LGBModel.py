from sklearn.model_selection import train_test_split
from BaseModel import *
from bayes_opt import BayesianOptimization

class LGBModel(BaseModel):
    def __init__(self) -> None:
        self.fixed_params = dict()

    def __init__(self, fixed_params:dict)->None:
        self.fixed_params = fixed_params

    def preprocessing(self):
        all_features = self.all_data.columns # 전체 피처
        cat_features = [feature for feature in all_features if 'cat' in feature] 
        onehot_encoder = OneHotEncoder() # 원-핫 인코더 객체 생성
        # 인코딩
        encoded_cat_matrix = onehot_encoder.fit_transform(self.all_data[cat_features]) 
        self.all_data['num_missing'] = (self.all_data==-1).sum(axis=1)

        # 명목형 피처, calc 분류의 피처를 제외한 피처
        remaining_features = [feature for feature in all_features
                            if ('cat' not in feature and 'calc' not in feature)] 
        # num_missing을 remaining_features에 추가
        remaining_features.append('num_missing')

        # 분류가 ind인 피처
        ind_features = [feature for feature in all_features if 'ind' in feature]

        is_first_feature = True
        for ind_feature in ind_features:
            if is_first_feature:
                self.all_data['mix_ind'] = self.all_data[ind_feature].astype(str) + '_'
                is_first_feature = False
            else:
                self.all_data['mix_ind'] += self.all_data[ind_feature].astype(str) + '_'
        cat_count_features = []
        for feature in cat_features+['mix_ind']:
            val_counts_dict = self.all_data[feature].value_counts().to_dict()
            self.all_data[f'{feature}_count'] = self.all_data[feature].apply(lambda x: 
                                                                val_counts_dict[x])
            cat_count_features.append(f'{feature}_count')

        # 필요 없는 피처들
        drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 
                        'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

        # remaining_features, cat_count_features에서 drop_features를 제거한 데이터
        all_data_remaining = self.all_data[remaining_features+cat_count_features].drop(drop_features, axis=1)

        # 데이터 합치기
        all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining),
                                    encoded_cat_matrix],
                                    format='csr')
        
        num_train = len(self.train) # 훈련 데이터 개수
        # 훈련 데이터와 테스트 데이터 나누기
        self.X = all_data_sprs[:num_train]
        self.X_test = all_data_sprs[num_train:]

        self.y = self.train['target'].values

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, 
                                                      test_size=0.2, 
                                                      random_state=0)

        # 베이지안 최적화용 데이터셋
        self.bayes_dtrain = lgb.Dataset(self.X_train, self.y_train)
        self.bayes_dvalid = lgb.Dataset(self.X_valid, self.y_valid)

    def __eval_function(self, num_leaves, lambda_l1, lambda_l2, feature_fraction,
                  bagging_fraction, min_child_samples, min_child_weight):
        '''최적화하려는 평가지표(지니계수) 계산 함수'''
        # 베이지안 최적화를 수행할 하이퍼파라미터 
        params = {'num_leaves': int(round(num_leaves)),
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'min_child_samples': int(round(min_child_samples)),
                'min_child_weight': min_child_weight,
                'feature_pre_filter': False}
        # 고정된 하이퍼파라미터도 추가
        params.update(self.fixed_params)
        
        print('하이퍼파라미터:', params)    
        
        # LightGBM 모델 훈련
        lgb_model = lgb.train(params=params, 
                            train_set=self.bayes_dtrain,
                            num_boost_round=2500,
                            valid_sets=self.bayes_dvalid,
                            feval=gini,
                            callbacks=[lgb.early_stopping(stopping_rounds=300),   # 조기종료 조건
                                        lgb.log_evaluation(300)])
        # 검증 데이터로 예측 수행
        preds = lgb_model.predict(self.X_valid)
        # 지니계수 계산
        gini_score = eval_gini(self.y_valid, preds)
        print(f'지니계수 : {gini_score}\n')
        return gini_score

    def optimize(self, param_bounds):
        optimizer = BayesianOptimization(f=self.__eval_function,      # 평가지표 계산 함수
                                        pbounds=param_bounds, # 하이퍼파라미터 범위
                                        random_state=0)    
        # 베이지안 최적화 수행
        optimizer.maximize(init_points=3, n_iter=6)
        # 평가함수 점수가 최대일 때 하이퍼파라미터
        max_params = optimizer.max['params']
        # 정수형 하이퍼파라미터 변환
        max_params['num_leaves'] = int(round(max_params['num_leaves']))
        max_params['min_child_samples'] = int(round(max_params['min_child_samples']))
        # 값이 고정된 하이퍼파라미터 추가
        max_params.update(fixed_params)
        return max_params
            
if __name__=="__main__":
# 베이지안 최적화를 위한 하이퍼파라미터 범위
    param_bounds = {'num_leaves': (30, 40),
                    'lambda_l1': (0.7, 0.9),
                    'lambda_l2': (0.9, 1),
                    'feature_fraction': (0.6, 0.7),
                    'bagging_fraction': (0.6, 0.9),
                    'min_child_samples': (6, 10),
                    'min_child_weight': (10, 40)}

    # 값이 고정된 하이퍼파라미터
    fixed_params = {'objective': 'binary',
                    'learning_rate': 0.005,
                    'bagging_freq': 1,
                    'force_row_wise': True,
                    'random_state': 1991}

    model = LGBModel(fixed_params)
    model.load_data(data_path="data/")
    model.preprocessing()
    max_params = model.optimize(param_bounds=param_bounds)

    print(f"최적 파라미터: {max_params}")
    model.fit_model(params = max_params,
          train_params={'num_boost_round':2500,
                        'feval':gini,
                        'n_splits':5,
                        'callbacks':[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(100)]})
    
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    best_threshold = 0
    for threshold in np.arange(0, 0.3, 0.001):
        valid_accuracy, valid_precision, valid_recall, valid_f1 = model.test_model(threshold=threshold)
        if valid_f1>f1:
            f1, best_threshold = valid_f1, threshold
    
    print('#'*80)
    print(f"최적 threshold: {best_threshold}")
    print(model.test_model(threshold=best_threshold))
    
    # model.save_submission("output/lgb_submission2.csv")
    