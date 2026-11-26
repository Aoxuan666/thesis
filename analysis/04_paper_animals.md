# 英文论文 2：animals-16-01417-v2.pdf 完整解析

- 文件：`animals-16-01417-v2.pdf`
- 大小：4,637,774 bytes
- 总页数：28
- 元数据：`{'format': 'PDF 1.7', 'title': 'Monitoring and Early Warning of the Cage-Rearing Broiler Farming Environment Based on an Inspection Robot', 'author': 'Sai Luo, Xiangchao Kong, Xintong Xie, Wanchao Zhang, Pengshen Zheng, He Zhu, Deqi Hao, Jingkun Sun and Changxi Chen', 'subject': 'In large-scale caged broiler houses, the combined effects of ventilation regulation, external meteorological disturbances, and changes in flock growth stages can easily cause the indoor thermal environment to deviate from the target settings, thereby increasing the difficulty of environmental control and production risks. Meanwhile, the broiler house environment exhibits pronounced spatial heterogeneity, while manual inspection is limited by low efficiency, high labor intensity, and delayed detection of anomalies. Therefore, this study proposes an inspection-robot-based method for predicting and providing early warnings for temperature deviation (TD) and temperature–humidity index (THI). In this method, TD is used to characterize the degree of deviation in environmental control, whereas THI is used to characterize the overall thermal environmental risk under the coupled effects of temperature and humidity. For the two prediction tasks, input variables were separately selected based on correlation analysis, and the same Wavelet–ECA–GRU model architecture was adopted for training to achieve short-term prediction over the next 30 min. The results show that the proposed model outperformed baseline models, including LSTM, TCN, and GRU. On the test set, the RMSE and R2 for TD prediction were 0.2886 and 0.9238, respectively, while those for THI prediction were 0.4309 and 0.9287, respectively. Based on the TD classification results and THI risk thresholds, warning strategies for heat deviation, cold deviation, and humid-heat risk were established. The proposed method can identify potential thermal environmental anomalies in caged broiler houses in advance from the perspectives of environmental control deviation and comprehensive thermal environmental risk, thereby assisting farm managers in assessing whether timely adjustments to ventilation, heating, or cooling are needed.', 'keywords': 'inspection robot; cage-reared broiler house; temperature deviation; temperature–humidity index (THI); wavelet decomposition; gated recurrent unit (GRU); efficient channel attention (ECA)', 'creator': 'LaTeX with hyperref', 'producer': 'pdfTeX-1.40.25; modified using OpenPDF 1.4.2', 'creationDate': "D:20260511173135+08'00'", 'modDate': "D:20260511113631+02'00'", 'trapped': '', 'encryption': None}`

- 嵌入图片总数：20（>=120px 有效图 20 张）

## 图表题注索引（自动识别）
- P4: Figure 1. Schematic diagram of the internal layout of the experimental broiler house, the distribution
- P5: Figure 2. Architecture of the robot-based poultry house environmental monitoring, prediction, and
- P6: Table 1. Relevant parameters of the environmental sensors.
- P7: Figure 3. Structure of the Wavelet–ECA–GRU model, with the temperature deviation (TD) prediction
- P9: Figure 4 illustrates the working principle of ECA. Unlike the SE mechanism, ECA
- P10: Figure 4. Schematic diagram of the efficient channel attention mechanism.
- P10: Figure 5. Schematic structure of the GRU.
- P12: Table 2. Suitable ranges of relative humidity and CO2 concentration at different broiler ages.
- P12: Table 3. Suitable temperature ranges at different broiler ages.
- P13: Table 4. Suitable air velocity ranges at different broiler ages.
- P14: Table 5. Thermal environmental risk classification based on the temperature–humidity index (THI).
- P15: Table 6. Description and units of the main variables used in correlation analysis and prediction modeling.
- P15: Table 7. Input features of the TD and THI prediction models.
- P16: Figure 6. Pearson correlation coefficients between candidate variables and TD and THI.
- P17: Figure 7. Comparison between observed and predicted values for temperature–humidity index (THI)
- P18: Figure 8 shows the performance of each model in the TD prediction task. According
- P19: Figure 8. Comparison between observed and predicted values for temperature deviation (TD) predic-
- P19: Tables 8 and 9, respectively.
- P19: Table 8. Performance comparison of different models for temperature–humidity index (THI) prediction.
- P20: Table 9. Performance comparison of different models for temperature deviation (TD) prediction.
- P23: Table 10. Effect of different wavelet basis functions on prediction performance.
- P24: Figure 9. Rolling prediction and early warning results of TD and THI during the test period.
- P24: Table 11. Statistical analysis results of TD and THI early warnings.

## 大尺寸嵌入式图片清单（>=120px）
- Page 4 img#1 (xref=239): 1200x903
- Page 5 img#1 (xref=265): 2181x609
- Page 7 img#1 (xref=299): 1560x772
- Page 10 img#1 (xref=362): 866x693
- Page 10 img#2 (xref=336): 789x429
- Page 16 img#1 (xref=444): 1065x639
- Page 16 img#2 (xref=445): 1056x644
- Page 17 img#1 (xref=467): 1034x549
- Page 17 img#2 (xref=468): 1033x549
- Page 17 img#3 (xref=469): 1034x550
- Page 17 img#4 (xref=470): 1034x550
- Page 17 img#5 (xref=471): 1034x549
- Page 17 img#6 (xref=472): 1034x549
- Page 19 img#1 (xref=517): 1034x549
- Page 19 img#2 (xref=518): 1035x549
- Page 19 img#3 (xref=519): 1034x550
- Page 19 img#4 (xref=520): 1034x550
- Page 19 img#5 (xref=521): 1035x550
- Page 19 img#6 (xref=522): 1034x550
- Page 24 img#1 (xref=583): 1072x1176

## 逐页全文

## Page 1 (chars=3805, imgs=0)

Academic Editor: Chunmei Li
Received: 3 April 2026
Revised: 30 April 2026
Accepted: 3 May 2026
Published: 6 May 2026
Copyright: © 2026 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license.
Article
Monitoring and Early Warning of the Cage-Rearing Broiler
Farming Environment Based on an Inspection Robot
Sai Luo 1,2,†
, Xiangchao Kong 1,†
, Xintong Xie 1,2
, Wanchao Zhang 2
, Pengshen Zheng 2
, He Zhu 2
,
Deqi Hao 2
, Jingkun Sun 2
and Changxi Chen 1,2,*
1
Key Laboratory of Smart Breeding (Co-Construction by Ministry and Province),
Ministry of Agriculture and Rural Affairs, Tianjin 300384, China; luosai_678@163.com (S.L.);
kxc_ouc@163.com (X.K.); xxt17u@163.com (X.X.)
2
College of Computer and Information Engineering, Tianjin Agricultural University, Tianjin 300384, China;
zhangwanchao@tjau.edu.cn (W.Z.); zhuhe_2078@163.com (H.Z.); haodeqi_3178@163.com (D.H.);
ekko201@163.com (J.S.)
*
Correspondence: chenchangxi@tjau.edu.cn; Tel.: +86-15302138777
†
These authors contributed equally to this work.
Simple Summary
Environmental conditions often vary markedly across different locations within caged
broiler houses. Relying solely on pre-installed fixed monitoring points makes it difficult to
detect potential thermal environmental risks in a timely manner. To address this issue, this
study developed an integrated monitoring system consisting of an inspection robot and
fixed monitoring points to enhance environmental monitoring capacity in broiler houses
and enable early identification of potential environmental anomalies. The system can help
farm managers detect emerging thermal environmental problems at an earlier stage and,
in combination with field observations of flock status, determine whether management
interventions such as adjustments to ventilation, heating, or spraying are required. This
approach can improve the timeliness of environmental regulation in broiler houses and
support broiler health and welfare.
Abstract
In large-scale caged broiler houses, the combined effects of ventilation regulation, external
meteorological disturbances, and changes in flock growth stages can easily cause the indoor
thermal environment to deviate from the target settings, thereby increasing the difficulty of
environmental control and production risks. Meanwhile, the broiler house environment
exhibits pronounced spatial heterogeneity, while manual inspection is limited by low
efficiency, high labor intensity, and delayed detection of anomalies. Therefore, this study
proposes an inspection-robot-based method for predicting and providing early warnings
for temperature deviation (TD) and temperature–humidity index (THI). In this method,
TD is used to characterize the degree of deviation in environmental control, whereas
THI is used to characterize the overall thermal environmental risk under the coupled
effects of temperature and humidity. For the two prediction tasks, input variables were
separately selected based on correlation analysis, and the same Wavelet–ECA–GRU model
architecture was adopted for training to achieve short-term prediction over the next 30 min.
The results show that the proposed model outperformed baseline models, including LSTM,
TCN, and GRU. On the test set, the RMSE and R2 for TD prediction were 0.2886 and
0.9238, respectively, while those for THI prediction were 0.4309 and 0.9287, respectively.
Based on the TD classification results and THI risk thresholds, warning strategies for
heat deviation, cold deviation, and humid-heat risk were established. The proposed
method can identify potential thermal environmental anomalies in caged broiler houses
Animals 2026, 16, 1417
https://doi.org/10.3390/ani16091417

## Page 2 (chars=3819, imgs=0)

Animals 2026, 16, 1417
2 of 28
in advance from the perspectives of environmental control deviation and comprehensive
thermal environmental risk, thereby assisting farm managers in assessing whether timely
adjustments to ventilation, heating, or cooling are needed.
Keywords:
inspection robot;
cage-reared broiler house;
temperature deviation;
temperature–humidity index (THI); wavelet decomposition; gated recurrent unit (GRU);
efficient channel attention (ECA)
1. Introduction
The thermal environment is a key factor affecting broiler production performance,
health status, and animal welfare. Modern broilers are characterized by rapid growth, high
metabolic rates, and high heat production per unit body weight, making them particularly
sensitive to adverse environmental conditions such as high temperature and humidity.
When environmental control in broiler houses is inappropriate, broilers may experience
reduced feed intake, inhibited weight gain, elevated body temperature, increased respi-
ratory rate, and impaired immune function; in severe cases, the risk of mortality and
culling may also increase [1,2]. Therefore, timely and accurate perception of broiler house
environmental conditions, together with risk warnings before abnormalities develop, is of
great significance for ensuring healthy broiler production, improving animal welfare, and
enhancing environmental regulation [3–5].
In studies evaluating the thermal environment of livestock and poultry housing, the
temperature–humidity index (THI) has been widely used in heat stress research because it
can comprehensively characterize thermal load under the coupled effects of temperature
and humidity [6,7]. Previous studies have shown that THI is closely associated with
physiological responses, changes in production performance, and heat stress status in
broilers; therefore, it has become one of the important indicators for assessing thermal
environmental risk in broiler production. In addition to THI, comprehensive thermal
environmental indices such as apparent temperature have also been used to characterize
the actual environmental load experienced by broiler flocks [8]. Compared with single
environmental parameters, these indices can, to some extent, comprehensively reflect the
combined effects of temperature, humidity, and other environmental factors on the thermal
environmental status of broilers, and thus have high application value in evaluating the
thermal environment of broiler houses.
During environmental control in broiler houses, the degree to which the actual tem-
perature deviates from the target temperature is also worthy of attention. Previous studies
have evaluated broiler house thermal environments and control performance from the
perspective of deviations between actual environmental temperature and target tempera-
ture, or between actual values and set points [9]. Unlike THI, which focuses on assessing
thermal environmental risk under the coupled effects of temperature and humidity, tem-
perature deviation is more suitable for characterizing the extent to which environmental
regulation deviates from the preset target. In addition to the above indicators, some studies
have also comprehensively evaluated broiler house environmental conditions from the
perspectives of environmental suitability and discomfort, providing a more intuitive basis
for environmental quality assessment and management decision-making [10,11].
The accurate assessment of environmental conditions largely depends on whether
monitoring data can adequately represent the actual environment experienced by broiler
flocks. At present, environmental monitoring in broiler houses is still mainly based on
fixed-point sensors, which collect continuous data through the deployment of devices for
https://doi.org/10.3390/ani16091417

## Page 3 (chars=4397, imgs=0)

Animals 2026, 16, 1417
3 of 28
measuring temperature, humidity, gas concentrations, and other parameters [12,13]. This
approach is convenient for installation and long-term operation. However, the internal
environment of a broiler house is not uniformly distributed, and microenvironmental
conditions often differ among regions, heights, and flock density levels. Therefore, fixed-
point monitoring results may not fully reflect the actual environment to which broilers are
exposed. In recent years, mobile robots and intelligent inspection technologies have begun
to be introduced into environmental monitoring in livestock and poultry houses. Compared
with fixed high-position monitoring, mobile inspection enables dynamic sampling in
the activity areas of broilers, thereby obtaining environmental information with broader
coverage and stronger spatial representativeness [14]. Nevertheless, existing studies have
mainly focused on equipment development and exploratory applications of environmental
monitoring, while how to predict key environmental indicators and identify risks based on
inspection data remains insufficiently investigated [15].
From a modeling perspective, environmental data in broiler houses usually exhibit
strong non-stationarity, obvious local fluctuations, and short-term disturbances. They are
also affected by the coupled effects of multiple environmental factors, which poses consid-
erable challenges for environmental time-series prediction. Previous studies in the fields
of agriculture and livestock housing environments have applied deep learning methods
such as RNN, LSTM, and GRU to predict environmental parameters, including temperature,
humidity, and gas concentrations, and have achieved favorable results [16,17]. On this
basis, some studies have further combined sequence decomposition methods with neural
networks to improve the ability of models to characterize complex sequences by extracting
variation information at different scales [18,19]. Other studies have introduced attention
mechanisms to weight key information in multivariate inputs, thereby improving prediction
accuracy [20–22]. However, most existing studies are still based on fixed-point monitoring
data, and their prediction targets mainly focus on a single environmental variable or a
single comprehensive index. Meanwhile, studies related to mobile inspection have placed
greater emphasis on equipment development and exploratory applications in environmental
monitoring. Compared with the most closely related studies, the present study focuses on
inspection scenarios in caged broiler houses, separately models two indicators—temperature
deviation (TD) and temperature–humidity index (THI)—and further establishes a joint early-
warning strategy to identify broiler house environmental conditions from the perspectives
of both environmental control deviation and thermal environmental risk.
Based on the above considerations, this study developed an application-oriented
monitoring–prediction–warning framework for caged broiler houses under inspection-
based monitoring scenarios, integrating environmental perception, TD/THI short-term
prediction, and dual-indicator early warning for the early identification of thermal envi-
ronmental anomalies. In this framework, inspection-based monitoring is used to improve
the spatial representativeness of environmental perception; TD characterizes the deviation
of the actual indoor temperature from the target temperature and reflects environmental
control deviation; and THI characterizes thermal environmental risk under the combined
effects of temperature and humidity. Compared with prediction studies that focus only on a
single environmental parameter or a single thermal environmental indicator, this study fur-
ther uses the TD and THI prediction results for warnings of heat deviation, cold deviation,
and humid-heat risk, thereby enabling the early identification of thermal environmental
anomalies in broiler houses. When warning information is issued by the system, farm
managers can combine the warning results with field observations of flock status, such
as feed intake, water intake, activity level, and respiratory condition, to comprehensively
assess thermal environmental risk and accordingly adjust ventilation rate, heating intensity,
and cooling measures in a timely manner.
https://doi.org/10.3390/ani16091417

## Page 4 (chars=2878, imgs=1)

Animals 2026, 16, 1417
4 of 28
2. Materials and Methods
2.1. Data Sources and Acquisition System
The experiment was carried out in a commercial poultry farm located in Dezhou,
Shandong Province, China, from 26 May to 29 June 2025, spanning one complete broiler
production cycle of approximately 5 weeks. The tested house accommodated more than
30,000 birds. Structurally, it was a standardized elongated cage-rearing house, measuring
about 101 m in length and 16 m in width, with an eave height of approximately 4.2 m and a
ridge height of approximately 6 m. Inside the house, cages were installed in a longitudinal
arrangement and organized into seven independent rows. Each row consisted of four tiers,
and each tier contained about 73 cage positions. Individual cages measured approximately
1.25 m × 1.0 m × 0.45 m (length × width × height).
To achieve continuous monitoring and dynamic perception of the broiler house en-
vironment, this study established a multi-source environmental data acquisition system
consisting of an inspection robot, fixed indoor monitoring points, and an outdoor mon-
itoring point. All the above devices were developed in-house by Tianjin Agricultural
University (Tianjin, China). As shown in Figure 1, the green rectangles represent the
chicken cages, channel numbers 1–16 indicate the inspection route of the robot, the arrows
indicate examples of the inspection direction, and the purple triangles represent the fixed
monitoring points arranged inside the broiler house. A total of 12 fixed monitoring points
were installed inside the broiler house, located in Channels 1, 4, 5, and 8. In each selected
channel, one monitoring point was placed at the front, middle, and rear sections along
the longitudinal direction of the house. The inspection robot was mainly used to obtain
dynamic environmental information from the broiler activity area, thereby improving the
spatial representativeness of environmental perception in the broiler house. The fixed
indoor monitoring points were mainly used to analyze environmental differences along
both the longitudinal and lateral directions of the house, reflecting the overall spatial
distribution characteristics of the indoor environment. The outdoor monitoring point was
used to synchronously collect outdoor environmental parameters, providing basic data for
analyzing the relationship between indoor and outdoor environmental changes. During
the experiment, the inspection robot conducted scheduled inspections along a preset route
in key areas of the broiler house. The inspection speed was set to 0.3 m/s, and one complete
full-route inspection required approximately 96 min.
Figure 1. Schematic diagram of the internal layout of the experimental broiler house, the distribution
of fixed monitoring points, and the inspection route of the inspection robot.
https://doi.org/10.3390/ani16091417

## Page 5 (chars=3201, imgs=1)

Animals 2026, 16, 1417
5 of 28
As shown in Figure 2, multi-source environmental monitoring data were uploaded
in real time to a remote server via a wireless network, where data storage, timestamp
alignment, state identification, preprocessing, resampling, prediction analysis, and early-
warning information generation were performed. The inspection robot recorded indoor
environmental data, including temperature, relative humidity, air velocity, and carbon
dioxide concentration, at a sampling interval of 3 s. The fixed indoor monitoring points con-
tinuously and synchronously collected the corresponding environmental parameters, while
the outdoor monitoring point continuously collected outdoor environmental parameters,
including temperature, relative humidity, and air velocity. After completing each inspection
task, the inspection robot automatically returned to the charging dock for recharging and
waited for the next scheduled task. During the inspection interval and charging waiting
period, the environmental monitoring devices mounted on the robot remained in opera-
tion. To ensure temporal consistency among environmental information from different
sources, all monitoring terminals attached timestamps during data acquisition. After the
data were uploaded to the server, they were sorted and aligned according to a unified
time reference. During the alignment process, a basic temporal resolution of 3 s was used
to map the data from the inspection robot, fixed indoor monitoring points, and outdoor
monitoring point onto a unified time scale. When slight deviations existed among the
recording times of different devices, nearest-neighbor time matching was adopted to merge
the data into the same time point. After time alignment, the time-aligned 3 s data were
resampled and aggregated using a 5 min time window. Within each 5 min window, valid
records of each environmental variable were averaged to obtain the representative value
for that interval. Invalid records, such as values beyond the sensor measurement range
or duplicate records, were excluded before aggregation. If no valid value was available
for a given variable within a 5 min window, the corresponding aggregated value was
marked as missing and then handled according to the missing-value procedure described
in Section 2.2. This aggregation process reduced the influence of instantaneous noise and
maintained consistency with the time step of the prediction model. The preprocessed time
series were then input into the prediction model for real-time computation, and future
environmental risk warning information was generated based on the prediction results.
Finally, real-time monitoring data, alarm results, prediction results, and early-warning
information were synchronously transmitted to the monitoring platform for displaying the
environmental status of the broiler house and information on abnormal risks. The main
sensor module models, measurement ranges, and accuracies of the monitoring terminals
used in this study are presented in Table 1.
 
Figure 2. Architecture of the robot-based poultry house environmental monitoring, prediction, and
early-warning system.
https://doi.org/10.3390/ani16091417

## Page 6 (chars=3307, imgs=0)

Animals 2026, 16, 1417
6 of 28
Table 1. Relevant parameters of the environmental sensors.
Sensor Index
Unit
Range
Accuracy
Sensor Model
Temperature
◦C
−20~+50 ◦C
±0.2 ◦C
JES-CO2TR-AR1-C2P0
Humidity
%
0~100% RH
±2% RH
JES-CO2TR-AR1-C2P0
Wind speed
m/s
0.1~5 m/s
±(0.2 m/s ± 0.02 × v)
JES-WS-CR1-C2P0
Carbon dioxide
ppm
300~5000 ppm
±50 ppm
JES-CO2TR-AR1-C2P0
2.2. Data Preprocessing
The time-series data collected during the experiment were used to train the prediction
models for temperature deviation (TD) and temperature–humidity index (THI) in the
broiler house. Temperature deviation (TD) was defined as the difference between the
measured indoor temperature and the target temperature, and its expression is given in
Equation (1).
TDt = Tin,t −Ttarget,t
(1)
where Tin,t is the measured indoor temperature at time t, and Ttarget,t is the target tem-
perature corresponding to time t. The temperature–humidity index (THI) was used to
characterize the overall thermal load under the coupled effects of temperature and humidity,
and its calculation formula is given in [23].
THI = (1.8 × T + 32) −[(0.55 −0.0055 × RH) × (1.8 × T −26)]
(2)
where T is the dry-bulb temperature and RH is the relative humidity. To avoid information
leakage in time-series modeling, the dataset was divided into a training set and a test set
at a ratio of 8:2, and feature extraction and normalization were then performed on this
basis. The raw data before preprocessing contained a total of 8458 records. Missing values
occurred only in the CO2 variable, with 191 missing values, accounting for 2.26% of the
records for this variable. In this study, linear interpolation was used to impute the missing
values. After this processing, no missing records remained in the dataset; therefore, no
sample rows were removed.
To eliminate the differences in feature scales and improve model training efficiency,
the min–max normalization method was adopted to map the input features to the interval
[0, 1], as expressed in Equation (3).
z =
x −xmin
xmax −xmin
(3)
where x is the original feature value, xmin and xmax were obtained from the statistics of the
training set and then used for the unified transformation of both the training set and the
test set. The target variables, TD and THI, were also scaled using normalizers fitted only
on the training set, thereby avoiding information leakage from the test set.
Finally, supervised learning samples were constructed using a sliding-window strat-
egy. Specifically, a multivariate sequence of the previous L = 36 time steps (5 min per
step, totaling 180 min) was used as the input to predict the temperature deviation and
temperature–humidity index at the future H = 6 time steps ahead (i.e., a 30-min-ahead
prediction). The 30 min prediction horizon was mainly determined based on the practical
requirements of environmental regulation in broiler houses. It can provide farm managers
with sufficient response time to promptly implement control measures such as ventilation,
heating, or spraying. If the prediction horizon is too short, the advance warning time would
be limited; however, an excessively long prediction horizon would increase the uncertainty
of short-term prediction. The input window was set to 180 min (L = 36), mainly because
https://doi.org/10.3390/ani16091417

## Page 7 (chars=2839, imgs=1)

Animals 2026, 16, 1417
7 of 28
changes in the thermal environment of broiler houses are jointly affected by ventilation
regulation, external disturbances, and heat production from broiler flocks, and therefore
exhibit certain short-term lag effects and thermal inertia. Compared with a shorter window,
a 180 min input window can provide more sufficient historical information to characterize
local fluctuations, variation trends, and the lagged effects of control responses. In contrast,
an excessively long window may introduce historical information that is weakly related to
the current short-term prediction task, thereby increasing model complexity and redundant
noise. Therefore, in this study, 180 min was used as the input window to balance the dy-
namic characteristics of the broiler house environment and the requirements of short-term
prediction. The window was moved along the time series with a step size of one time step
(5 min), thereby generating overlapping samples. This strategy transformed a single long
time series into a large number of supervised learning samples, thus expanding the training
sample size and improving the generalization ability of the model. At the same time, it
preserved local temporal dependencies and control lag effects, which helped the model
learn the short-term dynamic variation patterns of temperature deviation. In addition, the
rolling prediction method with a step size of 1 was consistent with the online deployment
scenario, facilitating real-time rolling prediction and early warning.
2.3. Model Construction
2.3.1. Overall Model Architecture
This study adopted a Wavelet–ECA–GRU hybrid framework for the short-term predic-
tion of temperature deviation (TD) and temperature–humidity index (THI) in broiler houses,
and its overall structure is shown in Figure 3. For each prediction task, the historical series
of the target variable was first decomposed at multiple scales using a sliding-window-based
discrete wavelet transform (DWT) to address its non-stationary fluctuation characteristics.
This process extracted the trend component and detail components at different frequency
bands, thereby enhancing the model’s ability to characterize slow drift and short-term
disturbances. Subsequently, the multi-scale features of the target variable obtained by
wavelet decomposition were concatenated with the remaining input variables to form the
model input sequence.
 
Figure 3. Structure of the Wavelet–ECA–GRU model, with the temperature deviation (TD) prediction
task as an example.
At the feature-weighting stage, an efficient channel attention (ECA) mechanism was
introduced. Based on the statistical information of each input channel within the time win-
dow, this mechanism used a lightweight one-dimensional convolution along the channel
https://doi.org/10.3390/ani16091417

## Page 8 (chars=3641, imgs=0)

Animals 2026, 16, 1417
8 of 28
dimension to model local inter-channel correlations and generate attention weights for each
channel. In this way, the input features were adaptively weighted, thereby adjusting the
relative contribution of different variables in the subsequent temporal modeling process.
At the temporal modeling stage, a gated recurrent unit (GRU) was employed to learn
the temporal dependencies and lag effects in the input sequence, and a fully connected
regression layer was used to output the predicted value of the target variable at the given
forecasting horizon.
2.3.2. Wavelet Decomposition
Broiler house environmental conditions are jointly affected by multiple factors, includ-
ing ventilation regulation, heat exchange, external meteorological conditions, and changes
in flock growth status. As a result, both temperature deviation (TD) and temperature–
humidity index (THI) time series exhibit evident non-stationary characteristics. These
characteristics include not only low-frequency trends that evolve gradually with environ-
mental regulation and growth-stage changes, but also high-frequency components caused
by equipment start–stop events, local disturbances, and fluctuations in the external environ-
ment. A single-scale feature is often insufficient to simultaneously characterize long-term
variation trends and short-term local fluctuations, thereby limiting the ability of prediction
models to learn complex dynamic processes. Therefore, this study introduced the discrete
wavelet transform (DWT) to perform multi-scale decomposition of the target sequence, so
as to extract low-frequency trend information and high-frequency fluctuation features at
different scales, thereby enhancing the representation ability of the subsequent prediction
model for non-stationary time series [24,25].
Considering that the prediction task must satisfy temporal causality, this study
adopted a sliding-window wavelet decomposition strategy. Taking the TD prediction
task as an example, for any time step t, only historical observations up to that time step
were used to construct a window signal with a length of W. When the historical length was
insufficient, left-side padding was performed by replicating boundary values to keep the
window length consistent. Therefore, the historical window signal corresponding to time
step t can be expressed as follows:
St = [TDt−W+1, TDt−W+2, . . . , TDt]
(4)
On this basis, discrete wavelet decomposition was applied to the window signal st,
yielding the level-L approximation coefficient sequence and the detail coefficient sequences
at each level, i.e.,
St
DWT
−−−−→
n
A(L)
t
, D(1)
t
, D(2)
t
, . . . , D(L)
t
o
(5)
where A(L)
t
denotes the level-L approximation coefficient, which mainly reflects the low-
frequency trend information of the sequence, and D(j)
t
denotes the detail coefficient at
level j, which is used to characterize the high-frequency fluctuation component at the
corresponding scale. To ensure strict alignment between the extracted features and the
current time step, the last coefficient from each coefficient sequence was extracted as the
wavelet representation at time t, and a wavelet feature vector was constructed.
Fwav
t
=
h
A(L)
t
, D(1)
t
, D(2)
t
, . . . , D(L)
t
i
(6)
The extracted wavelet features were then concatenated with the other screened en-
vironmental variables and used as inputs to the subsequent ECA–GRU model. For the
THI prediction task, the same sliding-window wavelet decomposition procedure was
adopted, with only the decomposition target changed from the TD historical series to the
https://doi.org/10.3390/ani16091417

## Page 9 (chars=3323, imgs=0)

Animals 2026, 16, 1417
9 of 28
THI historical series. In this study, the number of wavelet decomposition levels was set to
3, and the sliding window length was set to 56. To compare the effects of different wavelet
basis functions on prediction performance, db2, db4, and sym4 were selected as candidate
wavelet bases and further compared in subsequent experiments.
2.3.3. ECA Channel Attention
Attention mechanisms are essentially a resource allocation strategy and have been
widely applied in neural networks to enhance the representation of key information. Chan-
nel attention mechanisms can improve feature representation by modeling inter-channel
dependencies. However, SE-style channel attention mechanisms usually rely on dimen-
sionality reduction to capture channel-wise relationships, which may weaken the direct
modeling of dependencies among the original channels and reduce the preservation of
complete cross-channel interaction information. In contrast, ECA avoids dimensionality re-
duction and captures local cross-channel interactions through lightweight one-dimensional
convolution, thereby improving feature recalibration efficiency while maintaining low
model complexity [26,27].
Figure 4 illustrates the working principle of ECA. Unlike the SE mechanism, ECA
does not use fully connected layers to perform dimensionality reduction mapping of
channel features. Instead, it directly models channel dependencies through a local cross-
channel interaction strategy and achieves efficient channel weight allocation by means of
lightweight one-dimensional convolution. To adapt to the channel interaction range under
different feature dimensions, ECA adaptively selects the convolution kernel size according
to the number of channels, which can be expressed as
k = ψ(C) =

log2(C) + b
γ

odd
(7)
Among them, C denotes the number of feature channels, b and γ are hyperparameters,
and |·|odd indicates that the result is mapped to the nearest odd integer. Referring to the
default settings of the original ECA-Net, this study fixed γ = 2 and b = 1, and adaptively
determined the one-dimensional convolution kernel size k according to the number of
channels C, without conducting an additional manual search for k. The ablation experi-
ments in the original ECA-Net showed that adaptive convolution kernel selection generally
performs better than using a fixed kernel size, while also avoiding manual parameter
tuning through cross-validation [27]. Therefore, this study adopted this configuration to
reduce interference caused by additional hyperparameter selection and to balance model
complexity with feature representation capability.
Furthermore, ECA adaptively recalibrates the input features by assigning different
weights to different feature channels. The channel weight can be expressed as
ωi = σ
 
k
∑
j=1
αjyj
i
!
(8)
where ωi denotes the attention weight of the i-th channel, σ(·) is the Sigmoid activation
function, αj denotes the shared parameters of the convolution kernel, and yj
i denotes the
information response of the current channel and its local neighboring channels. Through
this mechanism, channels corresponding to key variables receive higher responses, whereas
redundant or strongly interfering feature channels are relatively suppressed.
https://doi.org/10.3390/ani16091417

## Page 10 (chars=1465, imgs=2)

Animals 2026, 16, 1417
10 of 28
Figure 4. Schematic diagram of the efficient channel attention mechanism.
2.3.4. Gated Recurrent Unit (GRU)
The gated recurrent unit (GRU) is a type of recurrent neural network (RNN) with a
gating mechanism. By introducing an update gate and a reset gate, GRU regulates the
retention of historical information and the writing of new information, thereby alleviat-
ing the gradient vanishing problem that commonly occurs in traditional RNNs during
long-sequence modeling. Compared with LSTM, GRU has a simpler structure and fewer
parameters, allowing it to improve training efficiency while maintaining strong temporal
modeling capability. Therefore, it is suitable for environmental sequence prediction tasks
characterized by non-stationary fluctuations [28,29]. The basic structure of GRU is shown
in Figure 5. At time step t, its calculation process can be expressed as follows:
zt = σ(Wz ∗xt + Uz ∗ht−1)
(9)
rt = σ(Wr ∗xt + Ur ∗ht−1)
(10)
eht = tanh(W ∗xt + U ∗(rt ⊙ht−1))
(11)
ht = (1 −zt) ⊙ht−1 + zt ⊙eht
(12)
Figure 5. Schematic structure of the GRU.
Among them, xt denotes the input feature vector at time step t, and ht denotes the
hidden state at the current time step. zt and rt represent the update gate and reset gate,
respectively, while eh denotes the candidate hidden state. W and U are learnable weight
matrices, σ is the sigmoid function, and ⊙denotes element-wise multiplication.
https://doi.org/10.3390/ani16091417

## Page 11 (chars=2618, imgs=0)

Animals 2026, 16, 1417
11 of 28
For an input sequence of length L, [xt−L+1, . . . , xt], the GRU outputs the corresponding
hidden-state sequence {ht−L+1, . . . , ht}. In this study, the hidden state at the last time step,
ht, was used as the sequence representation, and the predicted value of the target variable
was output through a fully connected layer as follows:
ˆyt+p = Woht + bo
(13)
where ˆyt+p denotes the predicted value at the future step p, and Wo and bo denote the
weight matrix and bias term of the output layer, respectively.
2.3.5. Experimental Environment and Parameter Settings
The experimental hardware configuration included an Intel i7-14650HX CPU with
a clock speed of 2.20 GHz, an RTX 4060 GPU, and 16 GB of memory. During model
training, the Adam optimizer was used with a learning rate of 0.001, a batch size of 32,
and a maximum of 200 training epochs, together with an early stopping strategy. The
input window length was set to 36 time steps, and the prediction horizon was set to 6
time steps. The hidden dimension of the GRU layer was set to 128, and the number of
network layers was set to 1. For both the TD and THI prediction tasks, three-level wavelet
decomposition was adopted, and modeling and training were conducted separately under
the same Wavelet–ECA–GRU framework.
2.3.6. Evaluation Indicators
To quantitatively evaluate the predictive performance of different models, mean
squared error (MSE), mean absolute error (MAE), root mean square error (RMSE), and the
coefficient of determination (R2) were adopted as evaluation indicators. Their calculation
formulas are given in Equations (14)–(17).
MSE = 1
N
N
∑
i=1
(yi −ˆyi)2
(14)
MAE = 1
N
N
∑
i=1
|yi −ˆyi|
(15)
RMSE =
v
u
u
t 1
N
N
∑
i=1
(yi −ˆyi)2
(16)
R2 = 1 −
N
∑
i=1
(yi −ˆyi)2
N
∑
i=1
(yi −y)2
(17)
where N denotes the sample size, yi denotes the actual value, ˆyi denotes the predicted
value, and y denotes the mean of the actual values.
2.4. Real-Time Monitoring Module for the Rearing Environment
To improve the representativeness of broiler house environmental perception and the
timeliness of anomaly identification, a real-time broiler house environmental monitoring
module was developed based on the inspection robot platform.
During the operation of the monitoring module, the inspection robot and fixed envi-
ronmental data acquisition devices synchronously collected environmental parameters at
the preset frequency and uploaded the data to the server in real time through a wireless
network. On this basis, the system incorporated broiler age information and retrieved the
https://doi.org/10.3390/ani16091417

## Page 12 (chars=2917, imgs=0)

Animals 2026, 16, 1417
12 of 28
suitable environmental ranges corresponding to different growth stages to determine the
status of the current temperature, relative humidity, air velocity, and CO2 concentration in
the broiler house. Combined with fixed-point monitoring data, the system further analyzed
environmental distribution differences along the front–rear and left–right directions of the
broiler house, thereby enabling dynamic monitoring and visual display of environmental
changes in the broiler house. When collecting environmental parameters, the inspection
robot synchronously recorded the cage number and channel number to mark the spatial
source of the real-time monitoring data. When a monitored value exceeded the suitable
range corresponding to the relevant age stage, the system generated a real-time monitoring
alarm and retained the location information corresponding to the abnormal data.
According to the environmental requirements of broilers at different growth stages,
suitable ranges were defined for temperature, relative humidity, air velocity, and CO2
concentration [30,31]. Specifically, the threshold values for relative humidity and CO2
concentration are listed in Table 2, the temperature thresholds are listed in Table 3, and
the air velocity thresholds are listed in Table 4. When a monitored value fell within the
suitable range, the corresponding environmental parameter was judged to be in a normal
state; when the monitored value was below the lower limit or above the upper limit, it was
judged to be abnormal. Let the monitored value of an environmental parameter at time t be
x(t), and let the lower and upper limits corresponding to the broiler age at time t be L(at)
and U(at), respectively, where at denotes the broiler age corresponding to time t. Then, the
environmental state determination rule can be expressed as
S(t) =







Low,
x(t) < L(at)
Normal,
L(at) ≤x(t) ≤U(at)
High,
x(t) > U(at)
(18)
where S(t) denotes the environmental state label at time t. For CO2 concentration, only
an upper threshold was defined; when the monitored value exceeded the suitable upper
limit, the state was determined as CO2 over-limit. Based on the above rules, the system was
able to perform real-time state identification of the broiler house environment and provide
basic support for subsequent early warning generation and environmental regulation
decision-making.
Table 2. Suitable ranges of relative humidity and CO2 concentration at different broiler ages.
Age (Days)
Suitable Range of Humidity
Suitable CO2 Concentration Range
1–10
65–70%
CO2 ≤1500 ppm
11–30
60–65%
CO2 ≤1500 ppm
31–45
55–60%
CO2 ≤1500 ppm
46–
50–55%
CO2 ≤1500 ppm
Table 3. Suitable temperature ranges at different broiler ages.
Age (Days)
Suitable Temperature Range
1–3
34–35 ◦C
4–7
32–33 ◦C
8–15
30–31 ◦C
16–23
28–29 ◦C
24–31
26–27 ◦C
32–39
24–25 ◦C
40–
22–23 ◦C
https://doi.org/10.3390/ani16091417

## Page 13 (chars=3716, imgs=0)

Animals 2026, 16, 1417
13 of 28
Table 4. Suitable air velocity ranges at different broiler ages.
Age (Days)
Suitable Air Velocity Range
1–14
0–0.20 m/s
15–21
0–0.51 m/s
22–28
0–1.02 m/s
29–
1.5–2.5 m/s
2.5. Real-Time Rolling Prediction and Dual-Indicator Early Warning Strategy
To meet the requirements for online monitoring and early intervention of environmen-
tal anomalies in broiler houses, this study developed a real-time rolling prediction and
dual-indicator early-warning strategy for temperature deviation (TD) and temperature–
humidity index (THI). This strategy was based on the environmental time series collected
by the inspection robot along a preset route. The raw 3 s inspection data were aggregated
using a 5 min time window, and the aggregated sequences were used as model inputs.
During each rolling prediction process, the model used the historical information from the
most recent 36 time steps to predict TD and THI over the next 30 min. The prediction results
were used to characterize the short-term variation trend of the thermal environment in the
broiler house. On this basis, this study combined the prediction results of TD and THI to
establish dual-indicator early-warning rules, generating warning information from two
perspectives: environmental control deviation and comprehensive thermal environmental
risk. This strategy can assist farm managers in promptly assessing abnormal thermal
environmental conditions in the broiler house in combination with field observations of
flock status, such as feed intake, water intake, activity level, and respiratory condition, and
in implementing control measures such as ventilation, heating, or cooling.
During real-time operation, multi-source environmental data collected by the inspec-
tion robot, indoor fixed monitoring points, and the outdoor monitoring point were first
aligned according to their timestamps and then resampled and aggregated into an equally
spaced 5 min time series. At any time step t, an input sequence was constructed based on
the most recent 36 historical time steps and fed into the trained Wavelet–ECA–GRU models
to obtain the predicted TD and THI values at 6 time steps ahead, i.e.,
ˆyTD
t+H = fTD

XTD
t

, ˆyTHI
t+H = fTHI

XTHI
t

(19)
where fTD and fTHI denote the TD prediction model and the THI prediction model, re-
spectively, and ˆyTD
t+H and ˆyTHI
t+H denote the predicted values of temperature deviation and
temperature–humidity index at the future time point, respectively. During online deploy-
ment, the input window was continuously updated forward by one time step, thereby
forming a rolling prediction process consistent with the actual sampling interval. The
rolling prediction was implemented at the broiler-house level using the aggregated 5 min
time series. Accordingly, each prediction update provided a short-term estimate of the
overall environmental trend and warning status of the monitored house, rather than a
location-specific prediction for each spatial point within the house. During online deploy-
ment, a new prediction was generated whenever a new 5 min aggregated sample became
available. Practical tests showed that approximately 0.5 s was required to complete model
input construction, prediction calculation, threshold judgment, and warning-level output
after each new aggregated sample was generated. This processing time was considerably
shorter than the 5 min rolling update interval and therefore did not affect the continuous
updating of warning results.
Based on the rolling prediction results, threshold judgment and risk warning were
performed separately for TD and THI. Specifically, the TD prediction results were used to
https://doi.org/10.3390/ani16091417

## Page 14 (chars=3688, imgs=0)

Animals 2026, 16, 1417
14 of 28
identify whether there would be a risk of cold deviation or heat deviation in environmental
control in the broiler house, whereas the THI prediction results were used to identify whether
there would be a humid-heat environmental risk. The TD warning threshold was set with
reference to the requirements for ventilation and temperature control in broiler houses
reported in Reference [32], which indicated that, during transition ventilation and tunnel
ventilation, the indoor temperature should not deviate from the target temperature by more
than ±2.5 ◦C. Therefore, when the predicted TD value was greater than 2.5 ◦C, a potential
heat deviation risk was identified; when the predicted TD value was lower than −2.5 ◦C, a
potential cold deviation risk was identified; otherwise, the condition was judged as normal.
For THI, this study referred to the THI-based thermal environmental risk classification
reported by temperature and humidity [2], in which THI values were classified into comfort,
alert, danger, and emergency categories. To facilitate continuous threshold judgment in the
proposed early-warning system, the corresponding intervals were expressed as THI < 70,
70 ≤THI < 76, 76 ≤THI < 81, and THI ≥81, respectively. Kim et al. [2] also showed that
higher THI conditions were associated with decreased feed intake and body weight gain,
as well as increased respiratory rate and rectal temperature, indicating that THI is closely
related to physiological responses and production performance changes caused by heat
stress in broilers. The specific classification criteria are shown in Table 5.
Table 5. Thermal environmental risk classification based on the temperature–humidity index (THI).
THI Range
Thermal Environment Status
THI < 70
Comfort
70 ≤THI < 76
Warning
76 ≤THI < 81
Danger
THI ≥81
Emergency
On this basis, the system separately generated warning information according to
the rolling prediction results of TD and THI. When the predicted TD was lower than the
cold-deviation threshold or higher than the heat-deviation threshold, the system issued
a cold-deviation warning or a heat-deviation warning, respectively. When the predicted
THI reached the alert level or above, the system issued a humid-heat risk warning at the
corresponding level. If the warning conditions for both TD and THI were met simultane-
ously, the system issued both types of warning information at the same time. In this way,
the proposed strategy can identify environmental anomalies in broiler houses in advance
from two dimensions: environmental control deviation and comprehensive thermal envi-
ronmental risk, and can support more targeted regulation measures such as ventilation,
heating, and cooling.
3. Results and Discussion
3.1. Correlation Analysis and Feature Selection (Pearson)
In studies on environmental prediction in livestock and poultry houses, Pearson corre-
lation analysis is commonly used for the preliminary screening of input variables to reduce
redundant variables and improve the relevance of model inputs. For example, previous
studies on apparent temperature prediction in broiler houses have used Pearson correlation
coefficients to screen input parameters [8]. Studies on environmental parameter prediction
in pig houses have also used Pearson correlation analysis to evaluate the correlations
between candidate environmental variables and prediction targets, and to determine model
input variables accordingly [22]. Therefore, in this study, Pearson correlation coefficients
were used to conduct a preliminary analysis of the linear correlations between candidate
variables and TD and THI.
https://doi.org/10.3390/ani16091417

## Page 15 (chars=3192, imgs=0)

Animals 2026, 16, 1417
15 of 28
In this study, the Pearson correlation coefficient (PCC) was used for correlation analysis.
The value range of the Pearson correlation coefficient r is [−1, 1], where r > 0 indicates a
positive correlation between variables, r < 0 indicates a negative correlation, and a larger
| r | indicates a stronger linear correlation between variables. To quantitatively interpret
the degree of correlation, the correlations were classified according to the absolute value
of the Pearson correlation coefficient. When | r |< 0.30, the correlation between variables
was considered negligible; when 0.30 ≤| r |< 0.50, a low correlation was considered
to exist; when 0.50 ≤| r |< 0.70, a moderate correlation was considered to exist; when
0.70 ≤| r |< 0.90, a high correlation was considered to exist; and when | r |≥0.90, a
very high correlation was considered to exist [33]. The Pearson correlation coefficient was
calculated as follows:
r =
∑n
i=1 (xi −x)(yi −y)
q
∑n
i=1 (xi −x)2∑n
i=1 (yi −y)2
(20)
where xi and yi denote the observed values of variables x and y at the i-th sample, re-
spectively; x and y denote the sample means of the corresponding variables, respectively;
and n denotes the number of samples. Table 6 lists the names, meanings, and units of the
main parameters involved in the correlation analysis and prediction modeling in this study,
thereby providing a unified description for subsequent variable screening and model input
design. Figure 6 shows the Pearson correlation coefficients between temperature deviation
(TD), temperature–humidity index (THI), and each candidate variable. In the figure, red
indicates positive correlations, blue indicates negative correlations, and the values beside
the bars represent the Pearson correlation coefficients between the corresponding variables
and the target variable. The results show that the absolute Pearson correlation coefficients
between some candidate variables and TD or THI were lower than 0.50, indicating weak
linear correlations with the target variables and relatively limited direct contributions to
the prediction results. In contrast, Age, Tout, CO2 and Ttarget showed moderate or stronger
correlations with TD and were therefore selected as input features for the TD prediction
model. For THI, Age, Tin and Ttarget also showed moderate or stronger correlations with
the target variable and were therefore included as inputs to the THI prediction model. The
input parameters of the TD and THI prediction models are shown in Table 7.
Table 6. Description and units of the main variables used in correlation analysis and prediction modeling.
Parameter
Description
Unit
Age
Age of broiler chickens
d
TD
Temperature deviation
◦C
THI
Temperature–Humidity
Index
-
CO2
Indoor CO2 concentration
ppm
Tin
Indoor temperature
◦C
Ttarget
Target temperature
◦C
Tout
Outdoor temperature
◦C
RHin
Indoor relative humidity
%
RHout
Outdoor relative humidity
%
Vout
Outdoor wind speed
m/s
Vin
Indoor wind speed
m/s
Table 7. Input features of the TD and THI prediction models.
Prediction Target
Input Parameters
TD
Age, TD, Tout, CO2, Ttarget
THI
Age, THI, Tin, Ttarget
https://doi.org/10.3390/ani16091417

## Page 16 (chars=3096, imgs=2)

Animals 2026, 16, 1417
16 of 28
Figure 6. Pearson correlation coefficients between candidate variables and TD and THI.
3.2. Comparison of Predictive Performance Among Different Models
To visually demonstrate the temporal representation characteristics of different models
in short-term prediction, the same continuous time segment from the test set was selected,
and the measured and predicted values for the THI and TD prediction tasks were visualized
separately. Figures 7 and 8 show the subfigure-based comparisons of six models for the
THI and TD prediction tasks, respectively, where the blue line represents the measured
values and the orange line represents the predicted values. The subfigures correspond,
from left to right and from top to bottom, to: (a) LSTM, (b) TCN, (c) GRU, (d) ECA–GRU,
(e) Wavelet–GRU, and (f) Wavelet–ECA–GRU.
3.2.1. Comparison of THI Prediction Results
As shown in Figure 7, the fitting ability of different models differed considerably in
the THI prediction task. Overall, all models were able to reflect the general variation trend
of the THI series to some extent; however, their prediction performance differed in intervals
involving local rapid fluctuations, abrupt changes, and high-value variations.
As shown in Figure 7a,b, although LSTM and TCN could capture the overall variation
direction of THI, obvious amplitude compression and local lag were still observed in
intervals with rapid fluctuations or abrupt changes. This indicates that the ability of these
two models to respond to non-stationary disturbances remains limited in the current task.
One possible reason is that, although LSTM has strong long-term memory capability, its
relatively complex gating structure may lead to delayed updating of effective information in
short-term high-frequency fluctuation scenarios, thereby weakening the model’s response
to rapid changes. TCN mainly relies on fixed convolutional receptive fields to extract
local temporal patterns. When a sequence simultaneously contains trend components,
periodic variations, and abrupt disturbances, its ability to represent cross-scale dynamic
relationships is relatively limited.
By comparison, as shown in Figure 7c, GRU better preserved the peak–valley structure
and variation rhythm of the THI series during most periods, indicating that it was more
effective in modeling short-term temporal dependencies in the current task. This is related
to the relatively compact gated update mechanism of GRU, which can retain necessary
historical state information while more directly incorporating changes in current inputs.
Therefore, in scenarios with limited sample size and a focus on short-term prediction, GRU
is more likely to capture the dynamic variation patterns in broiler house environmental
sequences. However, in some high-value fluctuation intervals, the original GRU still
showed a certain degree of underestimation, suggesting that when relying only on temporal
memory, the model remains limited in recovering amplitudes under the superposition of
complex disturbances.
https://doi.org/10.3390/ani16091417

## Page 17 (chars=1858, imgs=6)

Animals 2026, 16, 1417
17 of 28
 
 
 
 
 
 
Figure 7. Comparison between observed and predicted values for temperature–humidity index (THI)
prediction: (a) LSTM, (b) TCN, (c) GRU, (d) ECA–GRU, (e) Wavelet–GRU, and (f) Wavelet–ECA–GRU.
After introducing the ECA mechanism into GRU, as shown in Figure 7d, the model
responded more sufficiently in key fluctuation intervals, and the prediction offset near
some extreme values was alleviated. By modeling local interactions among channels, ECA
strengthened feature representations more relevant to THI variations and reduced the
interference of redundant information to some extent, enabling the model to focus more on
input information that contributed substantially to the prediction results.
Furthermore, as shown in Figure 7e, Wavelet–GRU tracked abrupt-change intervals
and high-frequency disturbances more stably after wavelet decomposition was introduced.
The main reason is that wavelet decomposition first decomposes the original THI series
into subsequences in different frequency bands, allowing long-term smooth trends and
short-term local fluctuations to be separated in the feature space. GRU then learns the
temporal dependencies of components at different scales. This processing reduces the
difficulty of directly modeling the original mixed signal and weakens the interference of
high-frequency disturbances in trend modeling, making it more suitable for handling the
non-stationarity and multi-scale fluctuation characteristics commonly present in broiler
house environmental sequences.
Overall, as shown in Figure 7f, Wavelet–ECA–GRU achieved better local fitting per-
formance in most key intervals. Its advantages were reflected not only in the accuracy of
overall trend fitting, but also in the stable representation of local fluctuations, extreme-value
https://doi.org/10.3390/ani16091417

## Page 18 (chars=4173, imgs=0)

Animals 2026, 16, 1417
18 of 28
changes, and turning-point characteristics. This improvement mainly benefited from the
synergistic effects of wavelet decomposition, the ECA mechanism, and GRU. Wavelet
decomposition was used to alleviate the non-stationarity and scale-mixing problems of
the input sequence, ECA was used to adaptively enhance key feature channels, and GRU
was used to capture temporal dynamic dependencies in the sequence. The combination of
these three components effectively improved the representation capability and prediction
stability of the model for complex broiler house environmental sequences.
3.2.2. Comparison of TD Prediction Results
Figure 8 shows the performance of each model in the TD prediction task. According
to the actual sequence trend, TD gradually increased from a relatively small deviation to
a higher level in the early stage. In the middle stage, TD switched several times among
different deviation levels and was accompanied by rapid decreases within short periods. In
the later stage, several prominent deviation peaks and rapid declining processes appeared.
These changes may be related to the accumulation of indoor temperature deviation from
the target temperature, the decline after the intervention of control measures, and the
combined influence of external environmental fluctuations.
Overall, all models were able to reconstruct the general variation process of the TD
sequence. As shown in Figure 8a,b, although LSTM and TCN could reflect the overall
variation direction of TD, obvious deviations still occurred in several intervals with large
deviations and during stages with rapid changes in deviation. The main limitation of
LSTM was not merely local underestimation; rather, it responded slowly during transitions
from one stable level to another. This suggests that, while using historical information,
LSTM tends to retain the inertia of the previous state, thereby weakening its immediate
representation of rapid switching processes. In contrast, the deviation of TCN was more
systematic, as it showed a relatively persistent overall overestimation in multiple intervals.
This indicates that although the local convolution-based learning mechanism can extract
short-term structures, it is insufficient for stably characterizing baseline changes among
different state intervals in the TD sequence.
By comparison, as shown in Figure 8c, GRU maintained the variation rhythm of TD
more effectively during most periods, indicating that it was more effective in modeling
short-term dependencies and state updates. This is related to the relatively compact gated
structure of GRU, which can retain necessary historical information while more flexibly
incorporating current input changes. Therefore, GRU is more suitable for processing envi-
ronmental sequences such as TD, which involve frequent state transitions. However, the
original GRU still showed some underestimation in certain high-value intervals, suggesting
that relying solely on temporal memory remains limited in recovering local extreme values
and rapid increasing processes.
On this basis, as shown in Figure 8d,e, ECA–GRU and Wavelet–GRU improved TD
prediction from different perspectives. ECA–GRU enhanced key input information through
channel recalibration, making the model more sensitive near peaks and in local disturbance
intervals; however, local amplification also occurred at some positions with abrupt changes.
Wavelet-GRU separated the slowly varying components and rapid short-term fluctuations in
the sequence through wavelet decomposition, thereby reducing the interference of complex
mixed signals in direct modeling. As a result, it showed more balanced performance in
intervals involving increasing deviation, decline, and obvious local fluctuations.
Overall, as shown in Figure 8f, Wavelet–ECA–GRU exhibited more stable prediction
performance in key intervals, especially during local rapid fluctuations and state-switching
stages, where both amplitude deviation and phase lag were reduced. This improvement
may be attributed to the synergistic effect of wavelet decomposition and the ECA mecha-
https://doi.org/10.3390/ani16091417

## Page 19 (chars=1198, imgs=6)

Animals 2026, 16, 1417
19 of 28
nism. The former helps separate slow-varying trends and fast-changing disturbances in the
sequence, while the latter strengthens the extraction of key feature information, thereby
improving the model’s ability to represent complex dynamic processes.
 
 
 
 
 
 
Figure 8. Comparison between observed and predicted values for temperature deviation (TD) predic-
tion: (a) LSTM, (b) TCN, (c) GRU, (d) ECA–GRU, (e) Wavelet–GRU, and (f) Wavelet–ECA–GRU.
3.3. Performance Evaluation of Different Prediction Models
To further quantitatively compare the performance differences among the models in
short-term prediction tasks, the prediction results on the test set were evaluated using
MSE, MAE, RMSE, and R2. The results of the models for THI and TD are presented in
Tables 8 and 9, respectively.
Table 8. Performance comparison of different models for temperature–humidity index (THI) prediction.
Model
MSE
MAE
RMSE
R2
LSTM
0.4341
0.4955
0.6588
0.8332
TCN
0.4829
0.5687
0.6949
0.8144
GRU
0.3660
0.5064
0.6050
0.8594
ECA-GRU
0.2393
0.3867
0.4892
0.9081
Wavelet-GRU
0.2466
0.4052
0.4966
0.9053
Wavelet-ECA-GRU
0.1857
0.3326
0.4309
0.9287
https://doi.org/10.3390/ani16091417

## Page 20 (chars=3444, imgs=0)

Animals 2026, 16, 1417
20 of 28
Table 9. Performance comparison of different models for temperature deviation (TD) prediction.
Model
MSE
MAE
RMSE
R2
LSTM
0.1714
0.3297
0.4140
0.8431
TCN
0.2154
0.3765
0.4641
0.8029
GRU
0.1628
0.3259
0.4035
0.8510
ECA-GRU
0.1287
0.2653
0.3587
0.8822
Wavelet-GRU
0.1216
0.2706
0.3487
0.8887
Wavelet-ECA-GRU
0.0833
0.2077
0.2886
0.9238
3.3.1. Comparison of THI Prediction Performance
As shown in Table 8, the performance differences among the models in the THI
prediction task were relatively obvious, among which the Wavelet–ECA–GRU model
achieved the best results. Its MSE, MAE, and RMSE were 0.1857, 0.3326, and 0.4309,
respectively, and the R2 value reached 0.9287. All of these metrics were better than those of
the other comparison models, indicating that this model had higher prediction accuracy
and stronger fitting ability for THI prediction.
Compared with the baseline GRU model, the MSE, MAE, and RMSE of the Wavelet–
ECA–GRU model decreased by 49.26%, 34.32%, and 28.78%, respectively, while R2 in-
creased from 0.8594 to 0.9287. This indicates that, in the THI prediction task, wavelet
decomposition and the ECA mechanism could also form an effective complementary com-
bination, thereby enhancing the model’s ability to represent complex temporal features.
Both ECA–GRU and Wavelet–GRU also outperformed the baseline GRU. Specifically, for
ECA–GRU, MSE, MAE, and RMSE decreased by 34.62%, 23.64%, and 19.14%, respec-
tively, and R2 increased to 0.9081. For Wavelet–GRU, the three error metrics decreased
by 32.62%, 19.99%, and 17.92%, respectively, and R2 increased to 0.9053. These results
indicate that introducing either the attention mechanism or wavelet decomposition alone
could improve THI prediction performance, while their combined use produced a more
pronounced improvement.
From the results of the baseline models, LSTM and TCN showed relatively higher
overall errors and lower R2 values, indicating that these models still had certain limitations
in handling THI series characterized by non-stationarity and complex fluctuations. By
contrast, the improved GRU-based models were better able to adapt to the multi-scale
variations and key feature differences in the THI series, thereby exhibiting better predic-
tive performance.
Overall, the Wavelet–ECA–GRU model achieved the best results in both TD and THI
prediction tasks, indicating that the combination of wavelet decomposition and the ECA
mechanism can effectively improve the short-term prediction capability of the GRU model
for broiler house environmental series.
3.3.2. Comparison of TD Prediction Performance
As shown in Table 9, the performance differences among the models in the TD pre-
diction task were relatively evident. Overall, the GRU-based models outperformed LSTM
and TCN, among which the Wavelet–ECA–GRU model achieved the best prediction re-
sults. Its MSE, MAE, and RMSE were 0.0833, 0.2077, and 0.2886, respectively, and the R2
value reached 0.9238. These results indicate that the model was able to more accurately
capture the variation process of the TD series, showing advantages in both error control
and fitting ability.
Using the baseline GRU as a reference, the MSE, MAE, and RMSE of Wavelet–ECA–
GRU decreased by 48.83%, 36.27%, and 28.48%, respectively, while R2 increased from
0.8510 to 0.9238. This indicates that simultaneously introducing wavelet decomposition
https://doi.org/10.3390/ani16091417

## Page 21 (chars=4348, imgs=0)

Animals 2026, 16, 1417
21 of 28
and the ECA mechanism into the GRU model significantly improved its ability to model
the TD series. Meanwhile, both ECA–GRU and Wavelet–GRU also performed better
than the baseline GRU. Specifically, for ECA–GRU, MSE, MAE, and RMSE decreased by
20.95%, 18.59%, and 11.10%, respectively, and R2 increased to 0.8822. For Wavelet–GRU,
the three error metrics decreased by 25.31%, 16.97%, and 13.58%, respectively, and R2
increased to 0.8887. These results indicate that either introducing the attention mechanism
or incorporating multi-scale decomposition features could improve TD prediction accuracy,
while the combined use of both produced a more pronounced improvement.
By contrast, the overall performance of LSTM and TCN was relatively weaker. In
particular, TCN performed worse than the GRU-based models on all four metrics, indicating
that its ability to capture the local fluctuations and temporal dependency characteristics
of the TD series in this study was insufficient. Overall, the Wavelet–ECA–GRU model
achieved the best performance in both the THI and TD prediction tasks, indicating that
simultaneously introducing wavelet decomposition and the ECA mechanism into the
GRU model can further enhance its short-term prediction capability for broiler house
environmental series.
Considering the results of both the THI and TD prediction tasks, Wavelet–ECA–GRU
achieved the best performance among the comparison models established in this study.
This indicates that the advantage of the proposed model was not reflected only in a single
error metric, but also in multiple aspects, including overall fitting ability, average error
control, and suppression of large deviations. These results suggest that the combination of
wavelet decomposition, the ECA mechanism, and GRU can improve the short-term predic-
tion capability of broiler house environmental sequences at different levels. Specifically,
GRU captures temporal dynamic dependencies, wavelet decomposition helps alleviate the
interference caused by non-stationarity, multi-scale fluctuations, and local high-frequency
disturbances in the original environmental sequences, and the ECA mechanism further
enhances the extraction of key feature information. These three components are not simply
superimposed; rather, they form strong complementary effects in terms of input represen-
tation, key information enhancement, and temporal modeling. Therefore, the combined
model outperformed the models that introduced only one of these improvement strategies.
Previous studies have indicated that sequence decomposition methods can improve the
representation of complex environmental time-series characteristics by extracting variation
information at different scales [18]. In addition, deep-learning and hybrid prediction mod-
els have also been applied to multi-parameter environmental prediction in poultry and
pig houses [19–22]. In line with these methodological trends, the improved performance
of the Wavelet–ECA–GRU model in this study may be attributed to the complementary
effects of multi-scale decomposition, the channel attention mechanism, and GRU-based
temporal modeling. However, unlike studies that mainly focus on a single environmental
variable or a single comprehensive thermal index, this study further links the short-term
prediction results with TD- and THI-based warning information, thereby extending model
comparison to decision-oriented environmental management in caged broiler houses.
From the differences between the THI and TD prediction results, the improved models
showed a consistent performance improvement trend in both tasks, indicating that the
proposed method has a certain degree of stability and is not effective only for a specific
indicator. Meanwhile, the magnitudes of improvement differed between the two tasks, sug-
gesting that THI and TD place different emphases on model capability. As a comprehensive
index formed by the coupling of temperature and humidity, THI is related not only to the
numerical state of the environment itself, but also closely to the thermal comfort level of
broilers. Previous studies have shown that when THI or heat load increases, broilers usually
dissipate heat by increasing respiratory rate, spreading their wings, increasing water intake,
https://doi.org/10.3390/ani16091417

## Page 22 (chars=4302, imgs=0)

Animals 2026, 16, 1417
22 of 28
and reducing activity. If high-temperature and high-humidity conditions persist, they
may further lead to reduced feed intake, elevated body temperature, impaired growth
performance, and decreased overall comfort [2]. Therefore, when the model predicts that
THI will continue to increase or remain at a high level, farm managers should promptly
check and adjust ventilation, cooling, and water supply measures in combination with
observed heat-stress behaviors of the flock, so as to reduce the adverse effects of hot and
humid environments on broiler comfort and production performance.
In comparison, TD more directly reflects the degree to which the actual temperature
deviates from the target temperature, and therefore can characterize dynamic deviations
during environmental regulation in broiler houses. A continuous increase in TD indicates
that the broiler house temperature is gradually becoming higher than the target level.
This is usually associated with insufficient ventilation, low air velocity, or limited cooling
effectiveness, which weakens the ability of broilers to dissipate heat through convection
and evaporation. Under such conditions, the flock is more likely to be exposed to a heat-
deviation environment and may gradually show responses such as increased panting, more
frequent lying behavior, reduced feed intake, and decreased activity [34]. When TD contin-
ues to decrease and the actual temperature becomes lower than the target temperature, it
may indicate that the flock is exposed to a cold-deviation condition. This is usually related
to insufficient heating or inappropriate ventilation strategies. Previous studies have shown
that cold stress can adversely affect broiler health and welfare and may induce immune
imbalance, reduced antioxidant capacity, and physiological disorders. Behaviorally, broilers
usually reduce heat loss by huddling, curling up, and decreasing activity to maintain body
temperature stability [35]. Accurate prediction of local abrupt changes, state-transition
intervals, and short-term rapid fluctuations in TD can be used to determine whether the
environmental control system in the broiler house deviates from the set state. When TD con-
tinues to increase, the operation of the ventilation and cooling systems should be checked
first. When TD continues to decrease and falls below the target temperature, heating and
ventilation strategies should be adjusted in time, thereby reducing the adverse effects of
heat and cold stress on broiler health and welfare.
It should be noted that the model performance evaluation results in this study are
still limited by the data source, experimental scenario, and sample coverage. First, the
data used in this study were obtained from a single production cycle and a single broiler
house facility. Although this allowed the prediction performance of the model to be verified
under relatively consistent environmental conditions, it also limited the applicability of
the findings to broader production conditions. Under different environmental conditions,
the patterns of temperature and humidity variation inside broiler houses may differ sub-
stantially. Previous studies have shown that seasonal conditions, broiler house structure,
and ventilation mode can affect indoor airflow as well as the distribution characteristics of
temperature and humidity [36,37]. In addition, differences in stocking density and venti-
lation control level may further alter the indoor thermal environment and environmental
regulation load. These factors may change the distribution range, fluctuation amplitude,
and state-transition characteristics of TD and THI, thereby affecting the stability and appli-
cability of the model in different production scenarios. Therefore, further validation under
different production cycles, seasons, and broiler house conditions is still needed in future
studies to more comprehensively evaluate the practical application performance of the
model in commercial production.
3.4. Effect of Wavelet Basis Functions on Prediction Performance
To further analyze the effect of wavelet basis function selection on model prediction
performance, three commonly used wavelet bases, namely db2, db4, and sym4, were
https://doi.org/10.3390/ani16091417

## Page 23 (chars=3719, imgs=0)

Animals 2026, 16, 1417
23 of 28
selected for comparative experiments while keeping all other model parameters unchanged.
The prediction results of the model for TD and THI under different wavelet basis functions
are presented in Table 10.
Table 10. Effect of different wavelet basis functions on prediction performance.
Prediction Target
Wavelet Basis
MAE
RMSE
R2
TD
db2
0.2399
0.3173
0.9073
TD
db4
0.2077
0.2886
0.9238
TD
sym4
0.2226
0.3057
0.9139
THI
db2
0.3681
0.4730
0.9118
THI
db4
0.3326
0.4309
0.9287
THI
sym4
0.3636
0.4638
0.9152
As shown in Table 10, the selection of the wavelet basis function had a certain influence
on the prediction accuracy of the model. For the TD prediction task, the model using
the db4 wavelet basis achieved the lowest MAE and RMSE values of 0.2077 and 0.2886,
respectively, while obtaining the highest R2 value of 0.9238. Compared with db2 and sym4,
db4 showed superior performance in terms of error control and goodness of fit. For the THI
prediction task, db4 also achieved the best prediction performance, with MAE, RMSE, and
R2 values of 0.3326, 0.4309, and 0.9287, respectively, outperforming the other two wavelet
basis functions.
Considering the MAE, RMSE, and R2 metrics for both TD and THI prediction tasks,
the db4 wavelet basis exhibited the best overall performance among the three wavelet basis
functions, enabling higher prediction accuracy and better fitting performance in the current
prediction tasks. Therefore, db4 was selected as the wavelet basis function for subsequent
model construction.
3.5. Analysis of Early Warning Results
To evaluate the application performance of the proposed dual-indicator early warning
method during practical operation, the rolling prediction results of TD and THI, together
with the corresponding warning states during the test period, were visualized and analyzed
in combination with statistical results, as shown in Figure 9 and Table 11. As can be seen
from Figure 9a,c, the predicted TD curve was generally consistent with the actual curve in
terms of the overall variation trend, and it was able to reflect the dynamic variation charac-
teristics of temperature deviation in the broiler house during the test period. In particular,
during overheating periods, the prediction results were able to track the actual fluctua-
tions relatively well, and the corresponding warning-state timeline was largely consistent
with the actual warning results. Table 11 further shows that, among the 596 test samples,
the actual TD states were dominated by the normal state, whereas the overheating state
accounted for a relatively small proportion, and the predicted distribution was generally
consistent with the actual distribution. Specifically, the actual and predicted proportions of
the normal state were 93.62% and 93.29%, respectively, while those of the overheating state
were 6.38% and 6.71%, respectively. Meanwhile, the deviation in overheating frequency
was 5.26%, and the consistency of state distribution reached 99.66%. This indicates that,
under the current test conditions, the model was able to characterize the overall distribution
pattern of temperature deviation in the broiler house and showed good consistency with the
classification results for the small number of heat-deviation cases during the test period. It
should be noted that, because heat-deviation samples accounted for only a small proportion
of the dataset, the current results are still insufficient to fully validate the model’s ability
to identify overheating under frequent, persistent, or more severe temperature-deviation
conditions. Further validation is therefore needed under different seasons, broiler house
https://doi.org/10.3390/ani16091417

## Page 24 (chars=1689, imgs=1)

Animals 2026, 16, 1417
24 of 28
facilities, stocking densities, and ventilation control conditions to evaluate the model’s
ability to identify TD states.
Figure 9. Rolling prediction and early warning results of TD and THI during the test period.
Table 11. Statistical analysis results of TD and THI early warnings.
Metric
Result
TD normal state proportion (Actual/Predicted)
93.62%/93.29%
TD overheating state proportion (Actual/Predicted)
6.38%/6.71%
TD overheating frequency deviation
5.26%
TD state distribution consistency
99.66%
THI warning level proportion (Actual/Predicted)
100%/100%
THI classification distribution consistency
100%
As shown in Figure 9b,d, the predicted THI curve was also generally consistent with
the actual curve, and all test samples fell within the warning level. Table 11 shows that, for
the 596 samples, both the actual and predicted classifications were at the warning level,
resulting in a classification distribution consistency of 100%. It should be emphasized
that, because all THI classifications corresponding to the current test samples were at the
alert level, with no samples falling into the comfort, danger, or emergency levels, the 100%
consistency rate in classification distribution shown in Table 11 only indicates that the
model prediction results were consistent with the threshold-based judgment under the
current single risk level. It does not demonstrate that the model already has the ability to
identify multi-level THI risks.
Considering the TD and THI results together, these two indicators describe the en-
vironmental status of the broiler house from the perspectives of temperature deviation
https://doi.org/10.3390/ani16091417

## Page 25 (chars=4119, imgs=0)

Animals 2026, 16, 1417
25 of 28
and temperature–humidity coupled thermal load, respectively, and therefore show a cer-
tain degree of complementarity. Specifically, TD reflects the degree to which the actual
temperature deviates from the target temperature, whereas THI characterizes the thermal
environmental risk level under the combined effects of temperature and humidity. Based
on the current test results, the dual-indicator early-warning method can reflect changes in
the environmental status corresponding to TD and THI during the test period. Given that
heat-deviation events accounted for only a small proportion of the current TD test samples
and that the THI samples were concentrated within a single alert level, future studies will
further conduct tests under different seasons, broiler house facilities, stocking densities,
and ventilation control conditions to evaluate the application performance of this method
under more severe temperature-deviation conditions and multi-level THI risk scenarios.
4. Conclusions
This study addressed the practical requirements for online environmental monitoring
and risk warning in caged broiler houses by developing a multi-source environmental
monitoring and short-term early-warning framework integrating an inspection robot, fixed
indoor monitoring points, and an outdoor monitoring point. For the two indicators, tem-
perature deviation (TD) and temperature–humidity index (THI), 30 min rolling prediction
models based on Wavelet–ECA–GRU were established separately, and a dual-indicator
early-warning strategy was designed to identify environmental anomalies in broiler houses
from two perspectives: environmental regulation deviation and temperature–humidity
coupled thermal risk.
The results showed that, under the data conditions of this study, Wavelet–ECA–
GRU achieved favorable prediction performance in both the TD and THI prediction tasks,
outperforming the comparison models, including LSTM, TCN, GRU, ECA–GRU, and
Wavelet–GRU. Specifically, the MSE, MAE, RMSE, and R2 values for TD prediction were
0.0833, 0.2077, 0.2886, and 0.9238, respectively, while those for THI prediction were 0.1857,
0.3326, 0.4309, and 0.9287, respectively. These results indicate that the integration of multi-
scale decomposition, channel attention, and GRU-based temporal modeling can effectively
characterize short-term fluctuations, local disturbances, and variation trends in broiler
house environmental sequences.
From the perspective of practical application, TD and THI reflect different aspects of
thermal environmental management in broiler houses. TD directly characterizes the degree
to which the actual temperature deviates from the target temperature and can be used to
identify potential heat-deviation or cold-deviation risks during environmental control. THI,
in contrast, comprehensively reflects the thermal environmental load under the combined
effects of temperature and humidity and can be used to identify humid-heat risk. The
combination of these two indicators can reduce the limitations associated with relying
only on a single temperature variable or a single comprehensive thermal environmental
index, allowing the warning results to cover both environmental control deviation and
temperature–humidity coupled risk. In practical production, this method can alert farm
managers in advance to changes in the thermal environment of broiler houses and help
them adjust ventilation rate, heating intensity, and cooling measures in a timely manner
in combination with field observations of flock status, such as feed intake, water intake,
activity level, and respiratory condition.
Further analysis of the warning results showed that, during the current test period,
the TD-based warning results were relatively close to the actual state distribution and
were able to reflect changes in environmental control deviation in the broiler house. For
THI-based warnings, because the test samples were mainly concentrated at the warning
level, the model prediction results were consistent with the threshold-based judgment
https://doi.org/10.3390/ani16091417

## Page 26 (chars=4252, imgs=0)

Animals 2026, 16, 1417
26 of 28
under the current single risk level. However, its identification capability in multi-level
risk scenarios still requires further validation. These results indicate that the TD and THI
dual-indicator prediction and early-warning method based on multi-source environmental
data has certain practical feasibility and can serve as an auxiliary approach for the early
identification of environmental anomalies in caged broiler houses.
Nevertheless, this study still has several limitations. The data were collected from
a single broiler house, a single batch, and one production cycle, resulting in relatively
limited scenario coverage. The distribution of risk levels in the warning test samples
was also imbalanced. In particular, the THI samples did not cover additional risk levels
such as comfort, danger, and emergency; therefore, the current results are not sufficient
to fully demonstrate the model’s comprehensive identification capability under complex
environmental conditions and multi-level risk scenarios. Future studies will incorporate
multi-season, multi-house, and multi-batch data to further validate the generalization
ability and warning stability of the model, and to optimize the applicability of thresholds
and warning strategies under different environmental conditions.
Author Contributions: S.L.: Writing—Original draft, Software, Methodology, Investigation, Valida-
tion, Data curation, Writing—review and editing. X.K.: Writing—review and editing, Data curation,
Investigation, Supervision. X.X.: Investigation, Data curation, Methodology. W.Z.: Investigation,
Data curation. P.Z.: Investigation, Data curation, Methodology. H.Z.: Data curation, Methodology.
D.H.: Investigation, Data curation, Methodology. J.S.: Investigation, Data curation. C.C.: Resources,
Supervision, Methodology, Investigation, Funding acquisition. All authors have read and agreed to
the published version of the manuscript.
Funding: (1) “Supported by National Key R&D Program of China” (2023YFD2000800); (2) “Supported
by China Agriculture Research System of MOF and MARA” (CARS-41); (3) “Supported by the 2025
Young Scientific and Technological Talents Development Fund Program of Tianjin Agricultural
University” (2025QNKJ12).
Institutional Review Board Statement: Ethical review and approval were not required for this study
because the research involved only non-invasive environmental data collection in a broiler house
using an inspection robot and fixed monitoring devices, without any direct handling, intervention, or
manipulation of the animals.
Informed Consent Statement: Informed consent was obtained from all owners prior to the inclusion
of their animals in this study.
Data Availability Statement: The data are not publicly available at this time due to ongoing related
research but are available from the corresponding author upon reasonable request.
Acknowledgments: The authors would like to acknowledge the financial support provided by the
National Key R&D Program of China (Grant No. 2023YFD2000800), the 2025 Young Scientific and
Technological Talents Development Fund Program of Tianjin Agricultural University (Grant No.
2025QNKJ12), and the China Agriculture Research System of MOF and MARA (CARS-41).
Conflicts of Interest: The authors declare no conflicts of interest.
References
1.
Lara, L.J.; Rostagno, M.H. Impact of Heat Stress on Poultry Production. Animals 2013, 3, 356–369. [CrossRef]
2.
Kim, H.R.; Seong, P.; Seol, K.-H.; Park, J.-E.; Kim, H.; Park, W.; Cho, J.H.; Lee, S.D. Effects of Heat Stress on Growth Performance,
Physiological Responses, and Carcass Traits in Broilers. J. Therm. Biol. 2025, 127, 103994. [CrossRef] [PubMed]
3.
Brugaletta, G.; Teyssier, J.-R.; Rochell, S.J.; Dridi, S.; Sirri, F. A Review of Heat Stress in Chickens. Part I: Insights into Physiology
and Gut Health. Front. Physiol. 2022, 13, 934381. [CrossRef] [PubMed]
4.
Quinteiro-Filho, W.M.; Ribeiro, A.; Ferraz-de-Paula, V.; Pinheiro, M.L.; Sakai, M.; Sá, L.R.M.; Ferreira, A.J.P.; Palermo-Neto, J.
Heat Stress Impairs Performance Parameters, Induces Intestinal Injury, and Decreases Macrophage Activity in Broiler Chickens.
Poult. Sci. 2010, 89, 1905–1914. [CrossRef]
https://doi.org/10.3390/ani16091417

## Page 27 (chars=5526, imgs=0)

Animals 2026, 16, 1417
27 of 28
5.
Oluwagbenga, E.M.; Fraley, G.S. Heat Stress and Poultry Production: A Comprehensive Review. Poult. Sci. 2023, 102, 103141.
[CrossRef]
6.
Purswell, J.L.; Dozier, W.A., III; Olanrewaju, H.A.; Davis, J.D.; Xin, H.; Gates, R.S. Gates Effect of Temperature-Humidity Index
on Live Performance in Broiler Chickens Grown From 49 To 63 Days of Age. In Proceedings of the 2012 IX International Livestock
Environment Symposium (ILES IX); American Society of Agricultural and Biological Engineers: St. Joseph, MI, USA, 2012.
7.
Omomowo, O.O.; Falayi, F.R. Temperature-Humidity Index and Thermal Comfort of Broilers in Humid Tropics. CIGR J. 2021, 23,
101–110.
8.
Zheng, P.; Zhang, W.; Gao, B.; Ma, Y.; Chen, C. Multi-Step Apparent Temperature Prediction in Broiler Houses Using a Hybrid
SE-TCN–Transformer Model with Kalman Filtering. Sensors 2025, 25, 6124. [CrossRef] [PubMed]
9.
Ryder, A.A.; Feddes, J.J.R.; Zuidhof, M.J. Field Study to Relate Heat Stress Index to Broiler Performance. J. Appl. Poult. Res. 2004,
13, 493–499. [CrossRef]
10.
Martinez, A.A.G.; Nääs, I.D.A.; De Carvalho-Curi, T.M.R.; Abe, J.M.; Da Silva Lima, N.D. A Heuristic and Data Mining Model for
Predicting Broiler House Environment Suitability. Animals 2021, 11, 2780. [CrossRef]
11.
Da Silva, N.C.; de Nääs, I.A.; Barros, J.d.S.G.; Moura, D.J. de Modeling Broiler Discomfort Under Commercial Housing: Seasonal
Trends and Predictive Insights for Precision Livestock Farming. Poultry 2025, 4, 38. [CrossRef]
12.
Li, M.; Zhou, Z.; Zhang, Q.; Zhang, J.; Suo, Y.; Liu, J.; Shen, D.; Luo, L.; Li, Y.; Li, C. Multivariate Analysis for Data Mining to
Characterize Poultry House Environment in Winter. Poult. Sci. 2024, 103, 103633. [CrossRef]
13.
Qi, F.; Zhao, X.; Shi, Z.; Li, H.; Zhao, W. Environmental Factor Detection and Analysis Technologies in Livestock and Poultry
Houses: A Review. Agriculture 2023, 13, 1489. [CrossRef]
14.
Balthazar, G.D.R.; Silveira, R.M.F.; Soares, T.C.; Da Silva, I.J.O. RobôFrango: Proof of Concept of a Mobile Biosensor Robot for
Environmental Monitoring in Broiler Houses. Results Eng. 2025, 28, 107796. [CrossRef]
15.
Park, M.; Britton, D.; Daley, W.; McMurray, G.; Navaei, M.; Samoylov, A.; Usher, C.; Xu, J. Artificial Intelligence, Sensors, Robots,
and Transportation Systems Drive an Innovative Future for Poultry Broiler and Breeder Management. Anim. Front. Rev. Mag.
Anim. Agric. 2022, 12, 40–48. [CrossRef] [PubMed]
16.
Lee, S.; Lee, I.; Yeo, U.; Kim, J.; Kim, R. Machine Learning Approach to Predict Air Temperature and Relative Humidity inside
Mechanically and Naturally Ventilated Duck Houses: Application of Recurrent Neural Network. Agriculture 2022, 12, 318.
[CrossRef]
17.
Chen, X.; Yang, L.; Xue, H.; Li, L.; Yu, Y. A Machine Learning Model Based on GRU and LSTM to Predict the Environmental
Parameters in a Layer House, Taking CO2 Concentration as an Example. Sensors 2023, 24, 244. [CrossRef]
18.
Xu, Z.; Zou, X.; Yin, Z.; Zhang, S.; Song, Y.; Zhang, J.; Lu, J. Prediction model of ammonia concentration in yellow-feather broilers
house during winter based on EEMD-GRU. INMATEH Agric. Eng. 2020, 61, 59–70. [CrossRef]
19.
Zang, J.; Ye, S.; Xu, Z.; Wang, J.; Liu, W.; Bai, Y.; Yong, C.; Zou, X.; Zhang, W. Prediction Model of Carbon Dioxide Concentration
in Pig House Based on Deep Learning. Atmosphere 2022, 13, 1130. [CrossRef]
20.
Ji, H.; Teng, G. Multistep Prediction of Temperature and Humidity in Poultry Houses Based on the GFF-Transformer Model. Eng.
Agric. 2025, 12, 803. [CrossRef]
21.
Li, X.; Shen, J.; Yang, X.; Nie, P.; Li, Y.; Li, Z.; Tan, Y.; Li, Z. Construction of Time Series Prediction and Dynamic Evaluation Model
for Environmental Factors in Pregnant Sow Houses in Winter. Smart Agric. Technol. 2025, 12, 101125. [CrossRef]
22.
Wu, Z.; Xu, K.; Chen, Y.; Liu, Y.; Song, W. A New Prediction Model Based on Deep Learning for Pig House Environment. Sci. Rep.
2024, 14, 31141. [CrossRef]
23.
You, G.; Dai, J.; Tian, Y.; Wang, Y.; Zhu, Q.; Li, D.; Zeng, Z.; Zhao, X. Study on the Suitable Environmental Temperature and
Humidity for Hy-Line Brown Layer Breeders in Summer. J. Yunnan Agric. Univ. 2020, 35, 990–995.
24.
Xu, D.; Zeng, Q.; Wang, W.; Zhang, X.; Wang, Y.; Zang, H. A Deep Learning Runoff Prediction Model Based on Wavelet
Decomposition and Dynamic Feature Fusion. Sci. Rep. 2025, 15, 37235. [CrossRef] [PubMed]
25.
Kio, A.E.; Xu, J.; Gautam, N.; Ding, Y. Wavelet Decomposition and Neural Networks: A Potent Combination for Short Term Wind
Speed and Power Forecasting. Front. Energy Res. 2024, 12, 1277464. [CrossRef]
26.
Liang, H.; Wu, J.; Zhang, H.; Yang, J. Two-Stage Short-Term Power Load Forecasting Based on RFECV Feature Selection Algorithm
and a TCN–ECA–LSTM Neural Network. Energies 2023, 16, 1925. [CrossRef]
27.
Wang, Q.; Wu, B.; Zhu, P.; Li, P.; Zuo, W.; Hu, Q. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks.
In Proceedings of the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); IEEE: Seattle, WA, USA, 2020;
pp. 11531–11539. [CrossRef]
28.
Cho, K.; van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; Bengio, Y. Learning Phrase Representations
Using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in
Natural Language Processing (EMNLP); Moschitti, A., Pang, B., Daelemans, W., Eds.; Association for Computational Linguistics:
Doha, Qatar, 2014; pp. 1724–1734.
https://doi.org/10.3390/ani16091417

## Page 28 (chars=2226, imgs=0)

Animals 2026, 16, 1417
28 of 28
29.
Chung, J.; Gulcehre, C.; Cho, K.; Bengio, Y. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
arXiv 2014, arXiv:1412.3555. [CrossRef]
30.
GB/T 19664-2005; Production Technique Criterion for Commercial Broiler. Standardization Administration of China: Beijing,
China, 2005.
31.
NY/T 1871-2010; Feeding Management Regulations of Yellow-Feathered Chicken. Ministry of Agriculture of the People’s Republic
of China: Beijing, China, 2010.
32.
DB1304/T 373-2021; Technical Regulation for Three-Dimensional Cage Rearing of Broilers. Handan Municipal Administration for
Market Regulation: Handan, China, 2021.
33.
Hinkle, D.E.; Wiersma, W.; Jurs, S.G. Applied Statistics for the Behavioral Sciences, 5th ed.; Houghton Mifflin: Boston, MA, USA,
2003.
34.
Cartoni Mancinelli, A.; Baldi, G.; Soglia, F.; Mattioli, S.; Sirri, F.; Petracci, M.; Castellini, C.; Zampiga, M. Impact of Chronic Heat
Stress on Behavior, Oxidative Status and Meat Quality Traits of Fast-Growing Broiler Chickens. Front. Physiol. 2023, 14, 1242094.
[CrossRef]
35.
Vasconcelos, M.D.C.; Sousa, L.S.; Lopes, T.S.B.; Gonçalves, L.M.; De Souza, A.B.; Avelar, N.M.; Oliveira, J.M.F.; Leme, F.D.O.P.;
Lara, L.J.C.; Araújo, I.C.S. Impact of Increased Pre-Start Diet Density on Broiler Chick Behavior, Corticosterone Levels, and
Performance Responses under Cold Stress during Early Life. J. Therm. Biol. 2024, 124, 103974. [CrossRef] [PubMed]
36.
Deng, S.; Li, Z.; Wei, Y.; Wang, Y.; Li, B.; Zheng, W. Assessing Temperature Distribution inside Commercial Stacked Cage Broiler
Houses in Winter. Animals 2024, 14, 2638. [CrossRef]
37.
Küçüktopçu, E.; Cemek, B.; Simsek, H. Modeling Environmental Conditions in Poultry Production: Computational Fluid
Dynamics Approach. Animals 2024, 14, 501. [CrossRef]
Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.
https://doi.org/10.3390/ani16091417
