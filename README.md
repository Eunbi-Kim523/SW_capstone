## SW_Capstone Design 2022-2
# 주제 - 딥러닝 기법을 활용한 Walmart 수요예측
## 1.과제개요
 치열한 경쟁 시장에서 기업들은 한정된 자원으로 최대한의 수익을 올리기 위해 다양한 전략을 활용한다. 특히 비효율성을 개선하고 불필요한 낭비와 불확실성은 줄여 자원의 효율화를 도모하기 위해 많은 기업이 ‘수요예측’ 분야에 주목하고 있다. 특히 수요예측을 통해 산출된 자료는 경영활동 계획의 기초가 되기에 경영자 관점에서도 매우 필요한 활동으로 인식된다. 2020년 3월, Kaggle에서 실제 Walmart 매장 판매 데이터를 활용한 M5 forecasting competition을 개최하였다. 해당 대회에서 경희대학교 재학생이 partial pooling(부분 풀링)과 LightGBM을 활용한 direct & recursive forecasting 모델을 결합한 DRFAM 예측 방법을 통해 Accuracy 부문 전체 1위를 기록하였다. 그 후 약 2년이 지난 현재 전통적인 통계적 기법부터 머신러닝, 딥러닝까지 다양한 기법을 통한 수요예측과 모델 개발이 활발히 이루어지고 있다.
 따라서, 본 프로젝트에서는 기존 대회에서 상위권을 기록했던 모델과 방법, 그리고 최근에 새롭게 개발된 방법론을 참고하여 새로운 방식의 딥러닝 Walmart 수요예측 모델을 구축하고 그 결과를 기존 방법과 비교하며 분석하는 것을 목적으로 한다.
## 2. 활용 데이터
 본 과제는 M5 Competition에서 주어진 calendar.csv, sales_train_evalution.csv, sell_price.csv 파일을 전처리하여 활용하였다. 또한, 활용 기법인 TCN(Temporal Convolutional Network)이 딥러닝 모델이기 때문에 모델 학습 및 평가에 많은 시간이 소요될 것을 고려하여 활용 데이터의 범위를 10개 store 중 CA_1 Store 하나로 한정하여 진행되었다. 구체적으로 Python pandas, numpy 라이브러리를 활용하여 전처리를 진행하였다. 3가지 데이터셋에 분산된 정보를 하나의 파일로 결합하고 날짜를 기반으로 year(연), month(월), day(일), week_day(요일) 컬럼을 새롭게 생성하였다. 또한, Sales 데이터를 활용하여 추가로 분석에 사용할 수 있는 지난 특정 기간(7일, 28일, 60일, 90일, 180일) 동안의 판매량 평균 및 표준편차를 구하여 컬럼을 추가 구성하였다.
## 3. 모델 설계 및 학습
 Keras의 TCN 라이브러리를 활용하여 모델을 구축하였다. 모델의 Input layer, Convolution layer, Dense layer, Output layer를 순차적으로 구성하여 모델을 설계하고 3,049개 아이템의 d_1부터 d_1885까지의 1,885일간의 데이터를 Train set, d_1886부터 d_1913까지의 28일간의 데이터를 Validation set으로 하여 모델을 학습시켰다. 그 후 d_1914부터 d_1941까지의 Test set 예측을 수행하고 RMSSE 값을 도출하여 성능을 평가한다. 모든 모델 구현 및 학습 과정은 Google Colaboratory 환경에서 수행되었다.

![image](https://user-images.githubusercontent.com/76906582/207917145-370b863f-d4aa-4652-9651-15124189ba80.png)

## 4. 수행 결과
 모델의 구조 및 파라미터를 변경해가며 성능을 향상시키기 위한 실험 과정을 거쳤고 그 결과 8개의 feature(month(월), day(일), week_day(요일), snap_CA(특별 이벤트), store_id, category_id, department_id, item_id)를 input으로 활용하고 2개의 TCN layer와 Dense layer로 구성된 모델이 가장 좋은 성능을 기록하였다. 모델의 activation function으로는 ReLU, optimizer로는 Adam을 사용하였으며 train loss function = MSE, batch_size = 1000, epoch = 3, dropout = 0으로 설정하였다.
Train loss = 4.7873, Validation loss = 4.0539, test set을 입력하여 예측값을 도출한 결과 MSE = 5.805, R2 score = 0.551, MAE = 1.218 가장 중점이 되는 성능 지표인 RMSSE = 0.432로 나타났다.

![image](https://user-images.githubusercontent.com/76906582/207890742-ee34ab45-c5bf-4369-a6eb-51d9d0695ee0.png)
![image](https://user-images.githubusercontent.com/76906582/207890988-c2c5a320-eff1-4063-8a03-5d388d892c1a.png)

## 5. 기대효과 및 결론
 해당 과제의 TCN 예측 모델은 Walmart 매장별 상품 수요예측을 목적으로 구축되었기 때문에 Walmart와 같은 기업의 수요예측 분야에서 활용될 수 있을 것이다. 특히 모델의 입력 데이터로 매우 기본적인 feature만을 사용하므로 제공 가능한 정보가 한정적인 환경에서도 활용 가능할 것이라 생각한다. 
 본 과제 수행 결과 RMSSE 0.432를 기록하며 기존 M5 대회의 1위 방법론보다 더 뛰어난 모델을 구현하겠다는 목표를 아직 달성하지는 못하였다. 하지만, 충분히 극복 가능한 성능 차이며 특히 큰 값의 판매량을 예측하는 능력을 개선한다면 충분히 더 좋은 성능을 발휘할 수 있을 것이라고 판단된다. 또한, 1위 방법론에 비해 예측에 활용하는 feature의 수가 훨씬 적고 활용 모델 수 역시 현저히 적기 때문에 예측 모델의 효율성 및 활용 가능성은 더 크다고 생각한다.
 따라서, 추후 졸업논문 작성 전까지 BiTCN과 같은 TCN 변형 모델이나 TCN_LSTM과 같은 하이브리드 모델 구현, 다중 모델 활용 등의 다양한 시도를 통해 예측 성능을 강화하고 더 다양한 데이터를 모델 평가에 적용해봄으로써 정확성과 효율성을 모두 갖춘 모델을 완성할 계획이다.
