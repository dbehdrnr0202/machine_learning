from LGBModel import *
import XGBModel as xgb

class XGBModel(LGBModel):
    def __init__(self) -> None:
        super().__init__()
    
    def preprocessing(self):
        all_data = pd.concat([self.train, self.test], ignore_index=True)
        all_data = all_data.drop('target', axis=1) # 타깃값 제거

        all_features = all_data.columns # 전체 피처
        # 명목형 피처
        cat_features = [feature for feature in all_features if 'cat' in feature]

        # 원-핫 인코딩 적용
        onehot_encoder = OneHotEncoder()
        encoded_cat_matrix = onehot_encoder.fit_transform(all_data[cat_features]) 

        # '데이터 하나당 결측값 개수'를 파생 피처로 추가
        all_data['num_missing'] = (all_data==-1).sum(axis=1)

        # 명목형 피처, calc 분류 피처를 제외한 피처
        remaining_features = [feature for feature in all_features
                            if ('cat' not in feature and 'calc' not in feature)] 
        # num_missing을 remaining_features에 추가
        remaining_features.append('num_missing')

        # 분류가 ind인 피처
        ind_features = [feature for feature in all_features if 'ind' in feature]

        is_first_feature = True
        for ind_feature in ind_features:
            if is_first_feature:
                all_data['mix_ind'] = all_data[ind_feature].astype(str) + '_'
                is_first_feature = False
            else:
                all_data['mix_ind'] += all_data[ind_feature].astype(str) + '_'

        cat_count_features = []
        for feature in cat_features+['mix_ind']:
            val_counts_dict = all_data[feature].value_counts().to_dict()
            all_data[f'{feature}_count'] = all_data[feature].apply(lambda x: 
                                                                val_counts_dict[x])
            cat_count_features.append(f'{feature}_count')

        # 필요 없는 피처들
        drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 
                        'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

        # remaining_features, cat_count_features에서 drop_features를 제거한 데이터
        all_data_remaining = all_data[remaining_features+cat_count_features].drop(drop_features, axis=1)

        # 데이터 합치기
        all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining),
                                    encoded_cat_matrix],
                                    format='csr')
        
        num_train = len(self.train) # 훈련 데이터 개수

        # 훈련 데이터와 테스트 데이터 나누기
        self.X = all_data_sprs[:num_train]
        self.X_test = all_data_sprs[num_train:]

        self.y = self.train['target'].values

        # 8:2 비율로 훈련 데이터, 검증 데이터 분리 (베이지안 최적화 수행용)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, 
                                                      test_size=0.2, 
                                                      random_state=0)

        # 베이지안 최적화용 데이터셋
        self.bayes_dtrain = xgb.DMatrix(self.X_train, self.y_train)
        self.bayes_dvalid = xgb.DMatrix(self.X_valid, self.y_valid)
    
    def eval_function(self, max_depth, subsample, colsample_bytree, min_child_weight,
                 reg_alpha, gamma, reg_lambda, scale_pos_weight):
        '''최적화하려는 평가지표(지니계수) 계산 함수'''
        # 베이지안 최적화를 수행할 하이퍼파라미터
        params = {'max_depth': int(round(max_depth)),
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'reg_alpha':reg_alpha,
                'reg_lambda': reg_lambda,
                'scale_pos_weight': scale_pos_weight}
        # 값이 고정된 하이퍼파라미터도 추가
        params.update(fixed_params)
        
        print('하이퍼파라미터 :', params)    
            
        # XGBoost 모델 훈련
        self.model = xgb.train(params=params, 
                            dtrain=self.bayes_dtrain,
                            num_boost_round=2000,
                            evals=[(self.bayes_dvalid, 'bayes_dvalid')],
                            maximize=True,
                            feval=gini,
                            early_stopping_rounds=200,
                            verbose_eval=False)
                            
        best_iter = self.model.best_iteration # 최적 반복 횟수
        # 검증 데이터로 예측 수행
        preds = self.model.predict(self.bayes_dvalid, 
                                iteration_range=(0, best_iter))
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
        max_params['max_depth'] = int(round(max_params['max_depth']))
        # 값이 고정된 하이퍼파라미터 추가
        max_params.update(fixed_params)
        return max_params
    
    def fit_model(self, params: dict, train_params: dict):
        # 층화 K 폴드 교차 검증기 생성
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1991)

        # OOF 방식으로 훈련된 모델로 검증 데이터 타깃값을 예측한 확률을 담을 1차원 배열
        oof_val_preds = np.zeros(self.X.shape[0]) 
        # OOF 방식으로 훈련된 모델로 테스트 데이터 타깃값을 예측한 확률을 담을 1차원 배열
        oof_test_preds = np.zeros(self.X_test.shape[0]) 

        # OOF 방식으로 모델 훈련, 검증, 예측
        for idx, (train_idx, valid_idx) in enumerate(folds.split(self.X, self.y)):
            # 각 폴드를 구분하는 문구 출력
            print('#'*40, f'폴드 {idx+1} / 폴드 {folds.n_splits}', '#'*40)
            
            # 훈련용 데이터, 검증용 데이터 설정
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_valid, y_valid = self.X[valid_idx], self.y[valid_idx]

            # XGBoost 전용 데이터셋 생성 
            dtrain = xgb.DMatrix(X_train, y_train)
            dvalid = xgb.DMatrix(X_valid, y_valid)
            dtest = xgb.DMatrix(self.X_test)
            # XGBoost 모델 훈련
            self.model = xgb.train(params=params, 
                                dtrain=dtrain,
                                num_boost_round=train_params['num_boost_round'],
                                evals=train_params['evals'],
                                maximize=train_params['maximize'],
                                feval=train_params['feval'],
                                early_stopping_rounds=train_params['early_stopping_rounds'],
                                verbose_eval=train_params['verbose_eval'])

            # 모델 성능이 가장 좋을 때의 부스팅 반복 횟수 저장
            best_iter = self.model.best_iteration
            # 테스트 데이터를 활용해 OOF 예측
            oof_test_preds += self.model.predict(dtest, iteration_range=(0, best_iter))/folds.n_splits
            
            # 모델 성능 평가를 위한 검증 데이터 타깃값 예측 
            oof_val_preds[valid_idx] += self.model.predict(dvalid, iteration_range=(0, best_iter))
            
            valid_pred_prob = self.model.predict(dvalid, iteration_range=(0, best_iter))
            valid_pred = np.where(valid_pred_prob>0.5, 1, 0)
            get_clf_eval(y_test=y_valid, pred=valid_pred, pred_proba_prob=valid_pred_prob)

            # 검증 데이터 예측 확률에 대한 정규화 지니계수
            gini_score = eval_gini(y_valid, oof_val_preds[valid_idx])
            print(f'폴드 {idx+1} 지니계수 : {gini_score}\n')

if __name__=="__main__":
    # 베이지안 최적화를 위한 하이퍼파라미터 범위
    param_bounds = {'max_depth': (4, 8),
                    'subsample': (0.6, 0.9),
                    'colsample_bytree': (0.7, 1.0),
                    'min_child_weight': (5, 7),
                    'gamma': (8, 11),
                    'reg_alpha': (7, 9),
                    'reg_lambda': (1.1, 1.5),
                    'scale_pos_weight': (1.4, 1.6)}

    # 값이 고정된 하이퍼파라미터
    fixed_params = {'objective': 'binary:logistic',
                    'learning_rate': 0.02,
                    'random_state': 1991}
    model = XGBModel(fixed_params)
    model.load_data(data_path="data/")
    model.preprocessing()
    max_params = model.optimize(param_bounds=param_bounds)
    print(max_params)
    model.fit_model(params = max_params,
          train_params={'num_boost_round':2000,
                        'maximize':True,
                        'feval':gini,
                        'early_stopping_rounds':200,
                        'verbose_eval':100})