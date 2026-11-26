# 英文论文 1：agriculture-16-01845.pdf 完整解析

- 文件：`agriculture-16-01845.pdf`
- 大小：34,928,931 bytes
- 总页数：31
- 元数据：`{'format': 'PDF 1.7', 'title': 'Development and Application of an Inspection-Robot-Based Digital Twin Platform for Cage-Reared Broilers', 'author': 'Sai Luo, Wanchao Zhang, Deqi Hao, He Zhu, Jingkun Sun, Jiaze Sun and Changxi Chen', 'subject': 'With the expansion of broiler production and the transition toward intelligent and labor-saving management, conventional manual inspection is limited by high labor intensity, unintuitive spatial representation of abnormalities, and inefficient on-site verification. To address these limitations, this study developed an abnormality monitoring system for cage-reared broiler houses by integrating an inspection robot, a digital twin environment, and cloud-based data services. A parameterized three-dimensional model and semantic cage anchors were established according to the dimensions of the physical broiler house and the cage arrangement rules. Robot simultaneous localization and mapping (SLAM) poses, inspection aisles, camera identifiers, and cage arrangement parameters were combined to calculate the semantic locations of dead-bird events and map them within the digital twin environment. Open-mouth breathing, infrared abnormalities, and acoustic abnormalities were additionally visualized at the candidate-cage, local-region, or inspection-aisle level according to the completeness of the available localization information. The system also enabled virtual–physical synchronization of the robot’s position, orientation, and operating status, as well as remote interactive control through a WebGL-based interface. Field tests conducted over approximately 100 days showed that model optimization reduced the triangle count, vertex count, and file size by 48.34%, 30.47%, and 44.81%, respectively, while shortening the initial WebGL scene loading time from 3.84 to 3.05 s. When 500 abnormality markers were displayed simultaneously, the optimized scene maintained an average frame rate of 67.83 fps. Among 2035 dead-bird events, 1954 were correctly localized in terms of cage row, tier, and group, yielding a cage-level localization accuracy of 96.0%. A total of 6412 robot control-command records were evaluated, achieving an overall execution success rate of 99.50%, with mean feedback times ranging from 1.0 to 1.2 s. These results demonstrate that the proposed system provides an integrated workflow for abnormality event acquisition, cage-level localization, three-dimensional visualization, and inspection robot management.', 'keywords': 'digital twin; inspection robot; cage-reared broiler house; abnormalitymonitoring; three-dimensional visualization', 'creator': 'LaTeX with hyperref', 'producer': 'pdfTeX-1.40.25; modified using OpenPDF 1.4.2', 'creationDate': "D:20260827193334+08'00'", 'modDate': "D:20260827134753+02'00'", 'trapped': '', 'encryption': None}`

- 嵌入图片总数：12（>=120px 有效图 12 张）

## 图表题注索引（自动识别）
- P4: Figure 1. An overview of the experimental broiler house, inspection robot, and multisource
- P5: Figure 1. Field conditions of the experimental broiler house and operation of the inspection robot.
- P5: Table 1. Abnormality event types, upstream data sources, and spatial mapping levels in the digital
- P5: Table 2. Measurement ranges and sampling intervals of the environmental parameters monitored by
- P6: Figure 2. Overall architecture of the proposed system.
- P7: Figure 3. Multisource data transmission and digital twin interactive control flow of the system.
- P8: Figure 4. Base 3D model of the digital twin broiler house.
- P10: Figure 5. Parameterized generation of the cage system and establishment of semantic anchors in
- P13: Figure 6. Semantic cage-location calculation and three-dimensional mapping process for abnormality
- P15: Figure 7.
- P15: Figure 7. Affine coordinate transformation process and virtual–physical synchronization mechanism
- P16: Table 3.
- P16: Table 3. Core hardware, software, and measurement settings used for WebGL scene performance
- P20: Figure 8. Interface of the Unity WebGL-based digital twin broiler-house monitoring platform. The
- P20: Table 4. Comparison of digital twin broiler-house model optimization and WebGL deployment
- P21: Figure 9. Changes in the average WebGL scene frame rate before and after optimization under
- P22: Table 5. Semantic cage-location calculation and field verification results for representative dead-bird
- P23: Figure 10. Cage-level three-dimensional marking and detail retrieval for a dead-bird event. (a) Abnor-
- P23: Figure 10b. Through the linkage between abnormality markers and event details, dead-bird
- P24: Table 6. Coordinates and fitting residuals of the SLAM–Unity affine calibration points.
- P25: Table 7. Leave-one-out cross-validation results for the SLAM–Unity affine transformation.
- P25: Figure 11. Interface for virtual–physical synchronization and robot status visualization.
- P26: Figure 12. Interactive control interface for the inspection robot. The asterisk indicates the selected
- P27: Table 8. Interactive control-command evaluation results for the inspection robot.

## 大尺寸嵌入式图片清单（>=120px）
- Page 5 img#1 (xref=266): 2642x2111
- Page 6 img#1 (xref=281): 4260x8265
- Page 7 img#1 (xref=293): 4984x2412
- Page 8 img#1 (xref=311): 5935x1556
- Page 10 img#1 (xref=323): 5204x4589
- Page 13 img#1 (xref=345): 1752x3632
- Page 15 img#1 (xref=381): 1978x3629
- Page 20 img#1 (xref=408): 3721x2503
- Page 21 img#1 (xref=428): 2428x1795
- Page 23 img#1 (xref=440): 4448x4415
- Page 25 img#1 (xref=466): 1323x1931
- Page 26 img#1 (xref=470): 1726x3137

## 逐页全文

## Page 1 (chars=3711, imgs=0)

Academic Editor: Ning Yang
Received: 18 July 2026
Revised: 19 August 2026
Accepted: 25 August 2026
Published: 27 August 2026
Copyright: © 2026 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license.
Article
Development and Application of an Inspection-Robot-Based
Digital Twin Platform for Cage-Reared Broilers
Sai Luo 1,2, Wanchao Zhang 1,2, Deqi Hao 1,2, He Zhu 1,2, Jingkun Sun 1,2, Jiaze Sun 1,2 and Changxi Chen 1,2,*
1
Key Laboratory of Smart Breeding (Co-Construction by Ministry and Province), Ministry of Agriculture and
Rural Affairs, Tianjin 300384, China; 2409010101@stu.tjau.edu.cn (S.L.); zhangwanchao@tjau.edu.cn (W.Z.);
2409048132@stu.tjau.edu.cn (D.H.); 2409010106@stu.tjau.edu.cn (H.Z.); 2409028116@stu.tjau.edu.cn (J.S.);
2509028129@stu.tjau.edu.cn (J.S.)
2
College of Computer and Information Engineering, Tianjin Agricultural University, Tianjin 300384, China
*
Correspondence: chenchangxi@tjau.edu.cn; Tel.: +86-15302138777
Abstract
With the expansion of broiler production and the transition toward intelligent and labor-
saving management, conventional manual inspection is limited by high labor intensity,
unintuitive spatial representation of abnormalities, and inefficient on-site verification. To
address these limitations, this study developed an abnormality monitoring system for cage-
reared broiler houses by integrating an inspection robot, a digital twin environment, and
cloud-based data services. A parameterized three-dimensional model and semantic cage
anchors were established according to the dimensions of the physical broiler house and
the cage arrangement rules. Robot simultaneous localization and mapping (SLAM) poses,
inspection aisles, camera identifiers, and cage arrangement parameters were combined to
calculate the semantic locations of dead-bird events and map them within the digital twin
environment. Open-mouth breathing, infrared abnormalities, and acoustic abnormalities
were additionally visualized at the candidate-cage, local-region, or inspection-aisle level
according to the completeness of the available localization information. The system also
enabled virtual–physical synchronization of the robot’s position, orientation, and operating
status, as well as remote interactive control through a WebGL-based interface. Field tests
conducted over approximately 100 days showed that model optimization reduced the
triangle count, vertex count, and file size by 48.34%, 30.47%, and 44.81%, respectively, while
shortening the initial WebGL scene loading time from 3.84 to 3.05 s. When 500 abnormality
markers were displayed simultaneously, the optimized scene maintained an average frame
rate of 67.83 fps. Among 2035 dead-bird events, 1954 were correctly localized in terms
of cage row, tier, and group, yielding a cage-level localization accuracy of 96.0%. A total
of 6412 robot control-command records were evaluated, achieving an overall execution
success rate of 99.50%, with mean feedback times ranging from 1.0 to 1.2 s. These results
demonstrate that the proposed system provides an integrated workflow for abnormality
event acquisition, cage-level localization, three-dimensional visualization, and inspection
robot management.
Keywords:
digital twin; inspection robot; cage-reared broiler house; abnormality
monitoring; three-dimensional visualization
1. Introduction
With the increasing scale, intensification, and intelligence of broiler production, com-
mercial poultry farms are gradually transitioning toward labor-saving, automated, and
Agriculture 2026, 16, 1845
https://doi.org/10.3390/agriculture16171845

## Page 2 (chars=4207, imgs=0)

Agriculture 2026, 16, 1845
2 of 31
refined management. Precision livestock farming uses sensors, machine vision, acoustic
monitoring, the Internet of Things (IoT), and intelligent data analysis to continuously moni-
tor animal behavior, health status, production performance, and farming environments,
thereby providing new technical approaches for real-time management and abnormality
warning in large-scale livestock production [1–4]. In poultry production, these technologies
have been applied to environmental monitoring, flock health assessment, animal welfare
analysis, and production process management, promoting a transition from experience-
based management to data-driven management [5–7].
Commercial cage-reared broiler houses are generally characterized by high stocking
densities, long house layouts, large numbers of cage locations, and densely arranged cages
comprising multiple rows, tiers, and groups. Conventional inspection primarily relies on
workers walking along inspection aisles to observe birds at different cage tiers and record
abnormal locations. Such manual monitoring is time-consuming and labor-intensive and
may depend heavily on inspection frequency and personnel experience [8].
In multi-tier cage systems, cage meshes may obscure the visible contours of birds [9],
while feeding and drinking equipment may further obstruct the field of view [10]. These
factors make it difficult to consistently inspect all cage locations within a short period.
After abnormalities such as dead birds are detected, workers must further identify the
corresponding cage row, tier, and group, thereby increasing the workload required for
on-site verification and intervention.
Machine vision, deep learning, infrared thermography, and acoustic monitoring have
been widely applied to poultry behavior and health monitoring, including behavior recog-
nition, diseased-bird detection, open-mouth breathing detection, thermal-status assessment,
and abnormal-sound recognition [11–20]. Although these studies have improved auto-
mated abnormality screening, they have mainly focused on recognition performance rather
than the spatial organization and intuitive presentation of detected abnormalities within
commercial cage-reared broiler houses.
Mobile inspection robots equipped with visible-light cameras, thermal imaging de-
vices, acoustic sensors, and environmental sensors have also been investigated for poultry-
house inspection and dead-bird detection [21–28]. These systems can reduce manual
inspection workload and support repeated sensing along predefined routes, but existing
studies have primarily emphasized autonomous inspection, abnormality detection, and
robot operation rather than semantic localization and three-dimensional representation of
abnormality events.
Despite substantial progress in poultry abnormality recognition and robotic inspection,
existing systems generally output results as detection images, confidence scores, abnor-
mality lists, two-dimensional robot map coordinates, or back-end data fields [21,24–27].
Simultaneous localization and mapping (SLAM) coordinates can represent the continuous
position of a robot within a map but cannot directly correspond to the cage-row, tier, and
group identifiers used in practical farm management. Farm managers must therefore inter-
pret these coordinates or images by additionally considering the inspection aisle, camera
installation position, and physical layout of the poultry house. Moreover, the complete-
ness of localization information differs among abnormality events. Visible-light dead-bird
events can generally be associated with robot positions, camera viewing sides, and cage-tier
information, whereas infrared, acoustic, and some behavioral screening results may only
be localized to candidate cages, local regions, or inspection aisles. Therefore, a hierarchical
spatial representation method is required to accommodate the different data characteristics
of abnormality events.
Digital twin technology provides a unified virtual representation of physical structures,
sensing data, equipment states, and operational information and has been increasingly
https://doi.org/10.3390/agriculture16171845

## Page 3 (chars=3891, imgs=0)

Agriculture 2026, 16, 1845
3 of 31
applied to smart agriculture and livestock production [29–37]. Existing studies have
mainly focused on environmental monitoring, animal and equipment state modeling,
production-process simulation, and resource management. However, practical deployment
in commercial livestock environments remains constrained by multisource data integration,
spatial correspondence between physical and virtual environments, model interoperability,
and browser-based rendering performance [32–35]. In multi-tier cage-reared broiler houses,
additional challenges include semantic cage-row–tier–group localization, hierarchical rep-
resentation of abnormality events with different localization completeness, coordinate
transformation between robot SLAM maps and three-dimensional scenes, and integration
of abnormality information, robot states, and remote control within a unified platform.
To address these limitations, this study developed an inspection-robot-based digi-
tal twin platform for a commercial cage-reared broiler house. The platform integrates
a parameterized three-dimensional environment, semantic cage-location mapping, hi-
erarchical visualization of abnormality events, SLAM–Unity coordinate transformation,
virtual–physical robot synchronization, cloud-based data services, and remote interactive
control. The system was evaluated through field data collected during an approximately
100-day production cycle.
The main contributions of this study are as follows:
(1) A parameterized digital twin environment was developed for a commercial cage-
reared broiler house. Geometric model optimization, shared model resources, and parame-
terized generation were used to reduce WebGL loading and rendering demands.
(2) A hierarchical spatial mapping method based on semantic cage anchors was
established. The method enables cage-level localization of dead-bird events and represents
events with incomplete localization information at the candidate-cage, local-region, or
inspection-aisle level.
(3) A virtual–physical robot synchronization method based on two-dimensional affine
transformation was developed to synchronize the robot’s position, orientation, and operat-
ing status within the digital twin environment.
(4) A cloud-based digital twin monitoring and interactive control system was devel-
oped and evaluated in terms of model runtime performance, dead-bird cage localization,
virtual–physical robot synchronization, and robot control execution.
2. Materials and Methods
2.1. Study Site, Inspection Robot and Data Sources
Field experiments were conducted at a commercial cage-reared white-feather broiler
farm in Gaoyou, Yangzhou, Jiangsu Province, China, from 27 September 2025 to 5 January
2026, covering a production cycle of approximately 100 days. During the experimental
period, the broiler house remained under normal production and management conditions.
The inspection robot performed routine data acquisition, status reporting, and inspection
tasks within the inspection aisles.
The experiments were conducted in a standardized enclosed cage-reared broiler house
measuring 110 m in length and 16 m in width. The cages were arranged in 7 rows, 4 tiers,
and 63 groups, and each cage unit measured 1.25 m × 1.00 m × 0.45 m. The broiler house
accommodated approximately 30,000 birds, with approximately 20 broilers housed in each
cage during the later stage of the production cycle. The main internal components included
cage systems, inspection aisles, ventilation equipment, lighting equipment, and other
auxiliary farming facilities. The cage arrangement, aisle distribution, and robot operating
areas provided the physical spatial basis for the subsequent construction of the digital twin
environment, spatial mapping of abnormality events, and virtual–physical synchronization
of the robot.
https://doi.org/10.3390/agriculture16171845

## Page 4 (chars=4377, imgs=0)

Agriculture 2026, 16, 1845
4 of 31
Field inspections were performed using a mobile inspection robot independently de-
veloped by our research team at Tianjin Agricultural University (Tianjin, China), equipped
with a simultaneous localization and mapping (SLAM)-based navigation chassis. The
robot integrated visible-light imaging devices, infrared thermal imaging devices, acous-
tic acquisition devices, environmental sensors, a lifting mechanism, and communication
modules. It could operate along predefined inspection routes within the broiler house
and perform multisource data acquisition, status reporting, and task execution. The
visual and thermal imaging devices were Hikvision dual-spectrum cameras (HM-TD2B28T-
7/T1; Hangzhou Micro-Imaging Software Co., Ltd., Hangzhou, China), with an image
resolution of 1920 × 1080 pixels. Acoustic data were acquired using a Hikvision omnidi-
rectional microphone (DS-65VA300U PRO; Hangzhou Micro-Imaging Software Co., Ltd.,
Hangzhou, China). Visible-light images were primarily used to identify dead-bird events
and open-mouth breathing behavior, infrared thermal images were used to identify thermal
abnormalities, and acoustic data were used to identify abnormal sounds. The environmen-
tal sensors collected air temperature, relative humidity, air velocity, light intensity, CO2
concentration, NH3 concentration, PM10, and total suspended particulate matter (TSP).
Temperature, relative humidity, and CO2 were measured using a JES-CO2TR-AR1-C2P0
sensor module (Shandong Jianda Renke Electronic Technology Co., Ltd., Jinan, China),
while air velocity was measured using a JES-WS-CR1-C2P0 sensor (Shandong Jianda Renke
Electronic Technology Co., Ltd., Jinan, China). The robot also uploaded operational in-
formation, including its SLAM map coordinates, heading angle, velocity, lifting-platform
height, battery level, charging status, and task progress.
The abnormality events integrated into the system included dead-bird events, open-
mouth breathing events, infrared abnormalities, and acoustic abnormalities. All recognition
results were generated by existing upstream abnormality detection modules developed by
the research team. In this study, these detection models were not retrained, optimized, or
compared in terms of recognition performance. Instead, their outputs, including abnor-
mality type, occurrence time, confidence score, image or audio evidence, and associated
location information, were used as event inputs for the digital twin system. The study fo-
cused on the spatial organization, three-dimensional visualization, and information linkage
of abnormality events. The model architectures, training procedures, and offline perfor-
mance of the upstream recognition modules have been reported in a previous study [38].
Dead-bird events were used for the quantitative evaluation of cage-level localization
performance. Throughout the entire production cycle, the cage locations calculated by the
system for individual dead-bird events were compared with the corresponding manual
on-site verification records. Open-mouth breathing events, infrared abnormalities, and
acoustic abnormalities were mainly incorporated as supplementary screening information
to evaluate the system’s ability to uniformly receive, spatially mark, and retrieve the details
of multiple types of abnormality data. Their upstream recognition accuracy and cage-level
localization accuracy were not quantitatively evaluated in this study.
Environmental data and robot status data were collected at intervals of 1 s, while
visible-light and infrared images were acquired at a rate of approximately one frame every
1.4 s. Data from all sources retained their acquisition timestamps, device identifiers, and
associated robot status information to support direct association between abnormality
events and the corresponding robot poses, semantic cage-location calculation, and updates
of the three-dimensional scene. The field conditions of the experimental broiler house, cage
layout, inspection aisles, and operating environment of the inspection robot are shown in
Figure 1. An overview of the experimental broiler house, inspection robot, and multisource
data is provided in Table 1, while the measurement ranges and sampling intervals of the
environmental sensors are presented in Table 2.
https://doi.org/10.3390/agriculture16171845

## Page 5 (chars=1630, imgs=1)

Agriculture 2026, 16, 1845
5 of 31
Figure 1. Field conditions of the experimental broiler house and operation of the inspection robot.
(a) Front exterior view of the experimental broiler house, where the number “9” indicates the on-site
broiler-house identifier; (b) side exterior view of the experimental broiler house; (c) internal cage
arrangement and inspection aisles; (d) inspection robot operating within an inspection aisle.
Table 1. Abnormality event types, upstream data sources, and spatial mapping levels in the digital
twin system.
Abnormality Event Type
Raw Data Source
Upstream Detection Output
Spatial Mapping Level
Dead-bird event
Visible-light images
Event type, confidence score,
image evidence, timestamp,
and associated position
Cage level
Open-mouth breathing
event
Visible-light images
Event type, confidence score,
image evidence, timestamp,
and associated position
Candidate-cage or
local-region level
Infrared abnormality event
Infrared thermal images
Anomaly region, thermal
image, timestamp, and
associated position
Candidate-cage or
local-region level
Acoustic abnormality
event
Acoustic data
Event type, timestamp,
audio evidence, and
associated position
Aisle or local-area level
Table 2. Measurement ranges and sampling intervals of the environmental parameters monitored by
the inspection robot.
Environmental
Parameter
Unit
Measurement
Range
Sampling Interval
Temperature
◦C
−20–50
1 s
Relative humidity
%
0–100
1 s
Wind speed
m s−1
0.1–5
1 s
CO2
ppm
300–5000
1 s
Illuminance
lx
0–20,000
1 s
NH3
ppm
0–100
1 s
PM10
µg m−3
0–1000
1 s
TSP
mg m−3
0–10
1 s
https://doi.org/10.3390/agriculture16171845

## Page 6 (chars=996, imgs=1)

Agriculture 2026, 16, 1845
6 of 31
2.2. System Architecture and Data Interaction Mechanism
This study developed a digital twin-based three-dimensional visualization and inter-
active control system for abnormality monitoring in cage-reared broiler houses. The system
uses the physical broiler house and inspection robot as its physical foundation, multisource
sensing data as its driving source, and the digital twin environment as the carrier for spatial
mapping of abnormality events, linkage of environmental conditions, and virtual–physical
synchronization of the robot. A cloud platform is further employed to support the unified
management of abnormality events, environmental information, inspection tasks, and
alarm records. The overall system architecture is shown in Figure 2, and the multisource
data transmission and digital twin interaction control process is illustrated in Figure 3.
Figure 2. Overall architecture of the proposed system.
https://doi.org/10.3390/agriculture16171845

## Page 7 (chars=3074, imgs=1)

Agriculture 2026, 16, 1845
7 of 31
Figure 3. Multisource data transmission and digital twin interactive control flow of the system.
2.2.1. Overall System Architecture
The system adopted a five-layer architecture comprising the physical layer, data layer,
model and processing layer, digital twin layer, and cloud platform application layer. The
physical layer consisted of the broiler house, cage array, inspection aisles, inspection robot,
and onboard sensing devices and was responsible for field data acquisition and robot
command execution. The data layer received and organized multisource sensing data,
robot status information, and control requests. The model and processing layer parsed and
stored these data and encapsulated abnormality recognition results, robot poses, device
information, and evidence files into structured event records.
The digital twin layer provided parameterized cage generation, semantic cage index-
ing, hierarchical spatial mapping of abnormality events, and synchronized visualization
of the inspection robot. The cloud platform application layer supported abnormality
management, environmental and robot-status monitoring, historical record retrieval, and
inspection-task control. The Unity WebGL environment was embedded into the cloud
platform interface to provide three-dimensional visualization and interactive control.
2.2.2. Multisource Data Transmission and Digital Twin Interactive Control Process
The system data transmission and interactive control process is shown in Figure 3 and
comprises four main stages: robot data uploading, cloud-based processing and storage,
service data delivery, and control-command transmission with execution feedback. The
cloud platform functions as the central hub for data storage, result services, and message
forwarding, connecting the inspection robot, the cloud platform application interface, and
the Unity WebGL digital twin interface.
The inspection robot uploads multimodal sensing data, environmental monitoring
data, SLAM poses, operating-status information, and task execution feedback to the cloud
platform. Structured data, such as robot poses, environmental parameters, battery level,
and task status, are transmitted as messages, whereas file-based data, including visible-
light images, infrared thermal images, and audio recordings, are transmitted through file
uploading or media streaming. Each data record retains the robot identifier, acquisition
timestamp, device identifier, and associated pose to support multisource data association,
abnormality event generation, and subsequent traceability.
The cloud platform performs data parsing, raw data storage, and structured event
generation. Abnormality recognition results are associated with robot poses, acquisition
devices, and evidence files to form unified abnormality event records. Abnormality events,
environmental data, and robot status information are then provided to the cloud platform
application interface and Unity WebGL environment through application programming
https://doi.org/10.3390/agriculture16171845

## Page 8 (chars=3219, imgs=1)

Agriculture 2026, 16, 1845
8 of 31
interfaces (APIs), WebSocket communication, and message-forwarding services. The digital
twin interface generates three-dimensional markers according to the cage-level or regional
localization results of abnormality events and updates the virtual robot and related status
information based on the robot pose, heading angle, and operating status.
When a user initiates an inspection, pause, resume, return-to-charge, or manual
control request through the WebGL interface, the cloud platform validates the request,
schedules the task, encapsulates the command, and transmits the control instruction to the
robot. After executing the corresponding task, the robot uploads its latest pose, task status,
and execution result to the cloud platform, which synchronously updates the application
interface and digital twin environment. This establishes a closed-loop control process
comprising interface interaction, cloud-based forwarding, robot execution, status feedback,
and scene updating.
2.3. Digital Twin Scene Construction
2.3.1. Three-Dimensional Modeling in Blender
A three-dimensional model of the experimental broiler house was constructed in
Blender (v5.2.0 LTS) based on field measurements and cage arrangement rules. The model
was developed at the actual scale of 110 m × 16 m and included the principal broiler-house
structure, standard cage units, inspection aisles, ventilation and lighting equipment, and
other major facilities. Particular attention was paid to preserving the geometric dimensions,
tier relationships, and key spatial boundaries of the cage system because these structures
served as the primary spatial reference for abnormality localization. The broiler house
contained 7 cage rows, 4 tiers, and 63 groups. Owing to the large number of repeated
cage structures, a modular modeling strategy was adopted, in which a standard four-tier
cage-group model was created as the basic unit for subsequent parameterized generation
of the complete cage array in Unity.
During modeling, noncritical decorative components, small connectors, and invisible
structures with limited relevance to abnormality localization were simplified, whereas
cage boundaries, inspection-aisle positions, and robot operating areas were retained. The
resulting models were exported in FBX format and imported into Unity for subsequent
parameterized cage generation, semantic anchor establishment, and WebGL deployment.
The base model of the digital twin broiler house is shown in Figure 4.
Figure 4. Base 3D model of the digital twin broiler house.
2.3.2. Three-Dimensional Model Optimization for WebGL Deployment
The complete digital twin broiler house contains large numbers of repetitive cages,
cage frames, and auxiliary facilities. Excessive complexity of the base model would increase
the burden associated with resource downloading, initial loading, and real-time rendering
on the WebGL page. However, cage boundaries, inspection aisles, and abnormality-marker
attachment regions also support semantic cage binding and spatial localization. Apply-
ing a uniform high-ratio simplification to all objects could therefore deform key spatial
https://doi.org/10.3390/agriculture16171845

## Page 9 (chars=4102, imgs=0)

Agriculture 2026, 16, 1845
9 of 31
boundaries or distort abnormality-marker positions. Accordingly, a differentiated model
optimization strategy was adopted based on the relationship between each model object
and its spatial localization function.
Standard cage units, principal cage boundaries, inspection aisles, robot operating areas,
and abnormality-marker attachment regions were classified as localization-critical objects.
Because these objects directly affect cage-identifier binding, semantic anchor establishment,
and virtual robot position visualization, their principal outlines, dimensional proportions,
and key spatial boundaries were retained. Only local details that did not affect localization
or visual recognition were removed.
Objects such as walls, roofs, auxiliary equipment enclosures, small connectors, and
invisible surfaces were mainly used to maintain the visual completeness of the scene
and had limited relevance to cage-level spatial mapping. Therefore, a greater degree of
geometric simplification was applied to these objects. The main optimization procedures
included reducing the number of vertices and polygons, removing invisible faces, merging
locally coplanar structures, and simplifying noncritical connection details.
In terms of resource management, repeated cage instances generated in Unity reused
the same mesh and material resources, while facilities with similar visual properties were
assigned shared materials to reduce resource redundancy and rendering-state switching
overhead. Necessary color distinctions and visual hierarchies were retained for abnormality
markers, inspection paths, and key interactive objects to ensure adequate spatial recognition
during browser-based interaction.
Through these procedures, the optimized model reduced geometric complexity and
resource redundancy while preserving the overall broiler-house layout, the spatial relation-
ships among cage rows and tiers, the positions of inspection aisles, and key localization
boundaries. The triangle count, vertex count, model file size, and initial WebGL load-
ing time before and after optimization were evaluated using the methods described in
Section 2.6.1.
2.3.3. Parameterized Generation of the Cage System and Establishment of Semantic Cage
Anchors in Unity
To efficiently construct the complete cage array and avoid cumulative positioning
errors associated with manual duplication, the optimized standard four-tier cage-group
model was used as the basic unit for parameterized generation in Unity. The configurable
parameters included the number of cage rows, tiers, and groups, spatial intervals, genera-
tion directions, and initial coordinates. According to the actual layout of the experimental
broiler house, the cage array was generated with 7 rows, 4 tiers, and 63 groups. The
parameter configuration interface and generated cage array are shown in Figure 5a,b.
Semantic cage anchors were established simultaneously during cage generation to
decouple abnormality-event localization from the geometric meshes of the cage models.
Each anchor corresponded to a specific cage location and was positioned at a fixed vertical
offset from the associated cage center to improve marker visibility and reduce occlusion.
Each anchor stored a unique cage identifier in the format “Cxx-Txx-Gxxx”; for example,
“C03-T02-G025” represents row 3, tier 2, and group 25. The correspondence between
semantic anchors and cage identifiers is illustrated in Figure 5c.
During initialization, the system constructed a semantic index using the complete
cage identifier as the key and the corresponding anchor object as the value. When an
abnormality event containing a valid cage identifier was received, the corresponding anchor
was retrieved directly and an abnormality marker was generated at the target location.
This approach reduced the dependence of abnormality mapping on model names, mesh
structures, and local geometric modifications while maintaining a stable correspondence
between cage identifiers and three-dimensional locations.
https://doi.org/10.3390/agriculture16171845

## Page 10 (chars=2219, imgs=1)

Agriculture 2026, 16, 1845
10 of 31
Figure 5. Parameterized generation of the cage system and establishment of semantic anchors in
Unity. (a) Parameter configuration interface in Unity; (b) top view of the generated cage array;
(c) semantic anchors and corresponding identifiers of a four-tier cage unit.
2.4. Semantic Cage-Location Calculation and Hierarchical Spatial Mapping of Abnormality Events
Based on SLAM Poses
The cages in a cage-reared broiler house are arranged regularly in multiple rows, tiers,
and groups. During inspection tasks, the upstream abnormality recognition modules can
generate event information for dead birds, open-mouth breathing, infrared abnormalities,
and acoustic abnormalities. Raw abnormality events generally contain the abnormality
type, occurrence time, acquisition device identifier, confidence score, and image or audio
evidence; however, they do not directly indicate the specific spatial location of the event
within the broiler house.
To establish correspondence between abnormality events and physical cage locations, a
semantic cage-location calculation and hierarchical spatial mapping method was developed
based on the robot SLAM position stored with the event record, inspection aisle identifier,
camera identifier, and cage arrangement parameters. For abnormality events with a valid
robot position, an explicit camera viewing side, and cage-tier information, the system
sequentially determines the cage-row, tier, and group identifiers, generates a complete cage
identifier, and maps the event to the semantic cage anchor established in Section 2.3. For
acoustic abnormality events lacking explicit viewing-side or tier information, the system
does not forcibly assign a single cage identifier. Instead, such events are represented at the
inspection-aisle or local-region level according to the aisle and local position at which the
acoustic data were acquired.
The method mainly comprises the association of abnormality events with robot inspec-
tion states, determination of cage-row and tier identifiers, calculation of the cage-group
identifier, semantic cage encoding, semantic anchor retrieval, and generation of three-
https://doi.org/10.3390/agriculture16171845

## Page 11 (chars=4033, imgs=0)

Agriculture 2026, 16, 1845
11 of 31
dimensional abnormality markers. It should be noted that this section focuses on converting
robot SLAM map information into discrete semantic cage locations. The affine transfor-
mation from continuous SLAM map coordinates to continuous Unity scene coordinates is
primarily used for virtual robot synchronization and is described in Section 2.5.
2.4.1. Association of Abnormality Events with Robot Inspection States
During task execution, the inspection robot continuously acquired and uploaded
information including its SLAM position, inspection aisle identifier, camera identifier,
task status, and acquisition time. When an abnormality event was generated by the
upstream recognition module, the corresponding robot position and associated inspection
information were packaged together with the abnormality information and stored as part
of the same event record.
Each abnormality event record contained the event timestamp, robot SLAM position,
inspection aisle identifier, camera identifier, confidence score, and associated evidence. If
key localization fields, including the robot position, inspection aisle identifier, or camera
identifier, were missing or invalid, the event was excluded from automatic cage-location
calculation and labeled as requiring manual verification.
The inspection robot travels back and forth within the same aisle. After reaching the
end of an aisle, the robot can reverse while maintaining the same body orientation. In this
case, only the direction of motion changes, whereas the physical installation positions and
viewing sides of the left- and right-side cameras remain unchanged. Therefore, forward
travel and reverse return do not alter the mapping relationship between the cameras and
cage rows, and the robot motion direction is not included in the calculation of cage-row or
tier identifiers.
2.4.2. Calculation of Cage-Row, Tier, and Group Identifiers
The experimental broiler house contained seven cage rows and eight inspection aisles.
For inspection aisle (A), the left-side cameras observed cage row (A), with this relationship
applying to aisles 1–7; on aisle 8, the left side corresponded to the wall area. The right-side
cameras observed cage row (A-1), with this relationship applying to aisles 2–8; on aisle 1,
the right side corresponded to the wall area. Therefore, only the left-side cameras (cameras
1–4) were used in aisle 1, only the right-side cameras (cameras 5–8) were used in aisle 8,
and all cameras (cameras 1–8) could be used in aisles 2–7.
Cameras 1–4 mounted on the left side of the robot were used to observe the cage rows
on the left side of the inspection aisle, whereas cameras 5–8 mounted on the right side were
used to observe the cage rows on the right side. The system retrieved the corresponding
cage-row identifier according to the inspection aisle identifier and the camera viewing
side. When the viewing side of a camera corresponded to a wall area, no valid cage-row
identifier was generated.
The camera identifier was also used to determine the target cage tier. Cameras 1–4
on the left side corresponded to tiers 1–4 of the left-side cage row, respectively, whereas
cameras 5–8 on the right side corresponded to tiers 1–4 of the right-side cage row, respec-
tively. After the target cage row and tier had been determined, the target cage-group
identifier was calculated from the robot’s SLAM position at the time of the abnormality
event. For each inspection aisle, the inspection start and end positions used for route
planning were determined by manually moving the robot to the inspection start boundary
adjacent to the first cage group and the inspection end boundary adjacent to the last cage
group and recording the corresponding SLAM coordinates. The transverse coordinates
of inspection aisles 1–8 were 0.016, 2.008, 4.192, 6.418, 8.469, 10.503, 12.606, and 14.693 m,
respectively, while the longitudinal coordinates of the start and end positions were −1.502
https://doi.org/10.3390/agriculture16171845

## Page 12 (chars=3551, imgs=0)

Agriculture 2026, 16, 1845
12 of 31
and −80.252 m, respectively. For each aisle, the recorded start coordinate was used as the
cage-group calculation origin, denoted by (xstart, ystart), and the corresponding start–end
segment defined the valid longitudinal range for cage-group calculation. Let the robot
position at the time of the abnormality event be denoted by (xcurrent, ycurrent). The planar
distance between the current robot position and the cage-group calculation origin was
calculated as follows:
d =
q
(xcurrent −xstart)2 + (ycurrent −ystart)2
(1)
Let W denote the effective spacing between adjacent cage groups along the arrange-
ment direction of the inspection aisle. For the experimental broiler house, W was set to
1.25 m, corresponding to the longitudinal length of each cage group. The target cage-group
identifier G was calculated as follows:
G =
 d
W

(2)
where ⌈·⌉denotes the ceiling function. Each cage row in the experimental broiler house
contained 63 cage groups; therefore, the valid range of (G) was 1–63. If the calculated value
exceeded this valid range or the robot position lay outside the valid longitudinal range
bounded by the recorded start and end positions of the corresponding inspection aisle, no
specific cage-group identifier was generated, and the abnormality event was labeled as
requiring manual verification.
The same cage-group calculation origin and group-spacing parameter were used when
the robot moved forward or returned in reverse within the same inspection aisle. Because
the cage-group identifier was determined by the robot’s actual position in the SLAM map,
a change in the direction of motion did not alter the cage-group identifier corresponding to
the same spatial position.
2.4.3. Semantic Cage-Anchor Retrieval and Three-Dimensional Mapping of
Abnormality Events
After valid cage-row, tier, and group identifiers had been obtained, the system gener-
ated a complete cage identifier according to the unified cage-row–tier–group coding rule.
Each cage identifier was represented as a zero-padded fixed-length string. For example,
“C03-T02-G025” denotes the cage located in row 3, tier 2, and group 25. This identifier
also served as the unified association index among abnormality event records, cloud-based
operational data, and objects within the Unity digital twin environment.
During initialization of the digital twin environment, the system traversed all semantic
cage anchors established during parameterized cage generation and constructed a semantic
cage index using the complete cage identifier as the key and the corresponding anchor
object as the value. For the experimental broiler house, which contained 7 cage rows, 4 tiers,
and 63 groups, a total of 1764 semantic anchors were established, forming a spatial index
covering all cage locations.
When the system received an abnormality event with a valid cage identifier, it retrieved
the corresponding semantic anchor using that identifier and generated a three-dimensional
abnormality marker near the target anchor. To reduce occlusion by cage meshes, feed
troughs, and other structures, a vertical offset was applied to the marker relative to the
anchor, and a guide line or positional indicator was used to point toward the target
cage location. This method eliminated the need for abnormality markers to depend on
specific model names or fixed mesh coordinates, because target-position retrieval and
three-dimensional mapping were performed through the unified cage identifier.
https://doi.org/10.3390/agriculture16171845

## Page 13 (chars=2469, imgs=1)

Agriculture 2026, 16, 1845
13 of 31
Each abnormality marker was associated one-to-one with its corresponding cloud-
based abnormality event record. The linked information included the event identifier,
abnormality type, occurrence time, confidence score, image or audio evidence, robot
position, environmental parameters recorded at the corresponding time, and processing
status. When a user selected an abnormality marker in the three-dimensional environment,
the WebGL interface retrieved the corresponding record using the event identifier and
displayed the related information in a detail panel. This enabled linked access among the
spatial location of the abnormality, supporting evidence, and operational records. The
semantic cage-location calculation and three-dimensional mapping process for abnormality
events with complete localization information is shown in Figure 6.
Figure 6. Semantic cage-location calculation and three-dimensional mapping process for abnormality
events with complete localization information. The superscript “*” denotes the robot inspection-state
record associated with the abnormality event at time ti.
Different spatial representation methods were adopted according to the completeness
of the localization information available for each abnormality type. For dead-bird events
with a valid robot position, inspection aisle identifier, and camera identifier, the system
generated a complete cage identifier and created a cage-level three-dimensional marker
at the corresponding semantic anchor. Open-mouth breathing and infrared abnormalities
were mapped to candidate cages or local regions according to the available localization
fields. When localization information was insufficient or uncertain, the event was not
forcibly assigned to a unique cage location. Because acoustic abnormalities generally
lacked explicit viewing-side and cage-tier information, they were primarily represented
at the inspection-aisle or local-region level according to the aisle and local position of the
robot at the time of data acquisition.
When key localization fields in an abnormality event record were missing or invalid,
the robot position was outside the valid range, the calculated cage-group identifier exceeded
the allowable range, or semantic anchor retrieval failed, the system retained the original
event record and labeled it as requiring manual verification rather than forcibly generating a
https://doi.org/10.3390/agriculture16171845

## Page 14 (chars=3572, imgs=0)

Agriculture 2026, 16, 1845
14 of 31
cage-level three-dimensional marker. Farm managers could review the corresponding time,
evidence, and robot status information through the abnormality event list and conduct
further verification according to the on-site conditions.
2.5. Virtual–Physical Synchronization of the Inspection Robot Based on Affine Transformation
To synchronize the physical inspection robot with its virtual counterpart, a spatial cor-
respondence was established between the robot SLAM map and the Unity environment. Be-
cause the two coordinate systems differ in origin, orientation, and scale, a two-dimensional
affine transformation was used to map robot positions from the SLAM map plane to the
Unity horizontal plane. The transformed coordinates, together with robot heading and
operating-status information, were used for virtual–physical synchronization.
2.5.1. Affine Calibration Between the SLAM Map and Unity Scene Coordinate Systems
Six fixed positions with explicit spatial correspondences between the SLAM map
and the Unity environment were selected as calibration points. To cover the principal
spatial extent of the broiler house, one point was selected at the front and rear ends of
inspection aisles 1, 5, and 8. These aisles represented one side, the central region, and
the opposite side of the broiler house in the transverse direction, while the front and rear
points covered different longitudinal regions. Each calibration point consisted of a pair
of SLAM map coordinates and corresponding Unity scene coordinates, and the six points
were not collinear.
A two-dimensional affine transformation was used to establish the coordinate rela-
tionship between the SLAM map and the Unity environment:
Xu= a1Xm+a2Ym+tx
(3)
Zu= a3Xm+a4Ym+tz
(4)
where Xm and Ym denote the current robot coordinates in the SLAM map; Xu and Zu
denote the corresponding coordinates in the Unity environment; a1, a2, a3 and a4 are affine
transformation parameters representing the scale, rotation, and directional relationships
between the two coordinate systems; and tx and tz are translation parameters compensating
for differences between the coordinate origins.
The six affine transformation parameters were jointly estimated using the least-squares
method based on the six calibration-point pairs. After parameter estimation, the SLAM
coordinates of the six calibration points were substituted into the affine transformation
model, and the transformed Unity coordinates were compared with their corresponding
reference coordinates. The planar Euclidean distance was used as the fitting residual:
ei =
r
X′
u,i −Xu,i
2
+

Z′
u,i −Zu,i
2
(5)
where X′
u,i and Z′
u,i are the Unity coordinates of the (i)-th calibration point obtained through
the affine transformation, and Xu,i and Zu,i are the corresponding reference coordinates
in the Unity environment. The consistency of the affine model with the calibration point
set was evaluated using the individual fitting residuals, mean fitting residual, root-mean-
square fitting residual, and maximum fitting residual.
Because all six calibration points were used to estimate the final affine transforma-
tion parameters, these residuals characterize the fitting consistency of the model with the
complete calibration-point set rather than prediction errors at locations excluded from
parameter estimation. To further assess predictive stability using the available field mea-
surements, leave-one-out cross-validation (LOOCV) was performed. In each iteration, one
https://doi.org/10.3390/agriculture16171845

## Page 15 (chars=2594, imgs=1)

Agriculture 2026, 16, 1845
15 of 31
calibration point was excluded and the affine transformation parameters were re-estimated
using the remaining five points. The excluded point was then transformed using the
resulting model and compared with its corresponding Unity reference coordinates, which
were not used in that iteration of parameter estimation. This procedure was repeated six
times so that each calibration point served once as the held-out validation point, and the
planar Euclidean distance was calculated as the held-out prediction error.
The practical applicability of the final affine transformation was further evaluated by
observing the synchronization of the virtual robot’s position, orientation, and operating
status while the physical robot moved through different inspection aisles. The affine
coordinate transformation and virtual–physical robot synchronization process is shown in
Figure 7.
Figure 7. Affine coordinate transformation process and virtual–physical synchronization mechanism
of the inspection robot.
2.5.2. Synchronization of the Virtual Robot Pose and Operating Status
After affine calibration, the SLAM coordinates periodically uploaded by the inspection
robot were transformed into the corresponding Unity coordinates and used to update the
position of the virtual robot. The robot heading angle was used to synchronize its body
orientation, with angular compensation applied according to the correspondence between
the principal inspection directions in the SLAM map and Unity coordinate systems.
The system distinguished the robot’s direction of movement from its body orientation.
When the robot moved backward within the same inspection aisle without changing
its heading, only the virtual position was updated and the model orientation remained
unchanged. When the physical robot turned and its heading angle changed, the virtual
robot orientation was updated accordingly. This avoided incorrectly visualizing reverse
motion as a change in robot orientation.
Robot operating information, including velocity, lifting-platform height, battery level,
charging status, current task, and task progress, was synchronously displayed in the
digital twin interface. Through continuous SLAM-pose transformation and status up-
dating, the virtual robot maintained correspondence with the physical robot in position,
orientation, and operating state, providing spatial and operational support for abnormality-
event review, subsequent on-site verification, inspection-task management, and remote
operation monitoring.
https://doi.org/10.3390/agriculture16171845

## Page 16 (chars=3737, imgs=0)

Agriculture 2026, 16, 1845
16 of 31
2.6. System Evaluation Methods
To evaluate the performance of the developed system in digital twin broiler-house
construction and browser-based deployment, semantic cage-location calculation and three-
dimensional mapping of dead-bird events, virtual–physical synchronization of the inspec-
tion robot, and interactive control, four groups of evaluations were conducted: model opti-
mization and WebGL runtime performance evaluation, field validation of cage-level local-
ization for dead-bird events, affine calibration and virtual–physical robot synchronization
evaluation, and interactive control-command evaluation. Open-mouth breathing, infrared
abnormalities, and acoustic abnormalities were incorporated primarily as supplementary
screening information to evaluate the unified organization, hierarchical spatial visualization,
and detail-retrieval functions of multiple types of abnormality data. Their upstream recog-
nition accuracy and cage-level localization accuracy were not quantitatively evaluated.
2.6.1. Evaluation of Model Optimization and WebGL Runtime Performance
To evaluate the optimization effectiveness of the digital twin broiler-house model, the
triangle count, vertex count, model file size, and initial WebGL scene loading time were
measured before and after model optimization. The triangle and vertex counts were used to
characterize model geometric complexity, whereas the model file size was used to represent
the volume of scene resources. Initial loading time was defined as the interval from when a
user entered the WebGL page to when the principal scene models had been loaded and the
page could normally support rotation, zooming, panning, and interactive operations.
The pre- and post-optimization tests were conducted using the same Unity version,
WebGL build configuration, terminal hardware, browser version, page resolution, network
environment, and initial scene viewpoint to minimize the influence of differences in test
conditions. The scenes before and after optimization also used the same cage arrangement,
abnormality-marker models, and functional configurations.
To evaluate the effect of increasing numbers of abnormality markers on WebGL run-
time performance, 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, and 500 abnormality markers
were loaded into the scenes before and after optimization. The markers were generated
according to the same rules, and identical generation ranges, scene viewpoints, and page
operating states were used in both sets of tests.
After the markers had been loaded under each test condition, the scene was allowed to
operate stably for 2 s. The scene frame rate was then continuously recorded for 10 s, and the
mean value during the measurement period was taken as the average frame rate for that
condition. Each condition was tested 10 times, and the mean of the repeated measurements
was reported. During testing, scene rotation, zooming, panning, abnormality-marker
selection, event-detail retrieval, and robot status-panel updating were also examined to
confirm that these functions operated normally.
The hardware, software, and measurement settings used in the evaluation are listed in
Table 3.
Table 3. Core hardware, software, and measurement settings used for WebGL scene performance
testing.
Parameter
Specification
CPU
Intel(R) Core(TM) i7-14650HX (2.20 GHz)
GPU
NVIDIA GeForce RTX 4060 Laptop GPU
Operating system
Windows 11, 64-bit
Web browser
Microsoft Edge, Version 144.0.3719.92
Unity and WebGL build configuration
Unity 2021.3.44f1c1
Page resolution and display refresh rate
2560 × 1600 pixels; 165.04 Hz
Stabilization time and measurement duration
2 s; 10 s
https://doi.org/10.3390/agriculture16171845

## Page 17 (chars=4010, imgs=0)

Agriculture 2026, 16, 1845
17 of 31
2.6.2. Evaluation of Cage-Level Localization and Three-Dimensional Mapping of
Dead-Bird Events
Dead-bird events with clearly documented on-site locations throughout the entire
production cycle were used for the quantitative evaluation of cage-level localization perfor-
mance. Farm personnel manually verified each dead-bird event included in the evaluation
and recorded the actual cage-row, tier, and group identifiers of the affected bird. The com-
plete cage identifier calculated by the system was then compared with the corresponding
manual verification record.
An event was considered correctly localized at the cage level only when the cage-row,
tier, and group identifiers calculated by the system were all consistent with the manually
verified values. Cage-level localization accuracy was calculated as the percentage of
correctly localized dead-bird events relative to the total number of valid dead-bird events
included in the evaluation.
For events in which the system-calculated cage location differed from the manual
verification record, the sources of error were further analyzed based on the robot SLAM
pose, inspection aisle identifier, camera identifier, and the spatial relationship between the
abnormality location and adjacent cage-group boundaries. Each error was then classified
as a cage-row, tier, or cage-group identification error. For cage-group identification errors,
the difference between the system-calculated and manually verified group identifiers
was further quantified to determine whether the event was assigned to the preceding or
following adjacent cage group or involved a displacement greater than one group.
For dead-bird events with successfully generated cage identifiers, the system was
further examined to determine whether it could retrieve the corresponding semantic anchor
using the cage identifier and generate a three-dimensional abnormality marker at the target
location. The system was also assessed for its ability to display the abnormality type,
occurrence time, confidence score, abnormality image, complete cage identifier, robot pose,
environmental parameters recorded at the corresponding time, and processing status after
the user selected the abnormality marker.
Open-mouth breathing and infrared abnormalities were displayed at the candidate-
cage or local-region level according to the available camera identifier, robot position,
and other localization fields. Acoustic abnormalities were primarily represented at the
inspection-aisle or local-region level. For these supplementary abnormalities, only data
integration, spatial visualization, and detail-retrieval functions were evaluated.
2.6.3. Evaluation of Affine Fitting, Leave-One-out Validation, and Virtual–Physical
Robot Synchronization
To evaluate the affine fitting between the SLAM map coordinate system and the
Unity scene coordinate system, six calibration points were selected at the front and rear
ends of inspection aisles 1, 5, and 8. All six pairs of SLAM map coordinates and Unity
scene reference coordinates were used to estimate the parameters of the two-dimensional
affine transformation.
After parameter estimation, the SLAM map coordinates of each calibration point were
substituted into the affine transformation model to obtain the corresponding transformed
Unity coordinates. These coordinates were then compared with the reference coordinates
in the Unity environment. The planar Euclidean distance between the transformed and
reference coordinates was used as the fitting residual. The fitting residual of each cal-
ibration point, as well as the mean, root-mean-square, and maximum fitting residuals,
were calculated.
Because all six calibration points were used to estimate the final affine transformation
parameters, the corresponding residuals were used to characterize the fitting consistency
of the affine model with the complete calibration-point set and were not interpreted as
https://doi.org/10.3390/agriculture16171845

## Page 18 (chars=4193, imgs=0)

Agriculture 2026, 16, 1845
18 of 31
independent external validation errors. To provide an additional out-of-sample assessment
based on the available field measurements, LOOCV was performed. For each iteration,
one point pair was completely excluded, and the remaining five point pairs were used to
re-estimate all six affine transformation parameters. The SLAM coordinates of the excluded
point were then transformed using the model obtained from the other five points and
compared with its corresponding Unity reference coordinates, which were not used in
that iteration of parameter estimation. The planar Euclidean distance was calculated as
the held-out prediction error. After all six iterations, the mean error, root-mean-square
error (RMSE), and maximum error were calculated to characterize the predictive stability
of the affine transformation. The practical applicability of the coordinate transformation
was functionally evaluated by examining continuous position updating, orientation syn-
chronization, and operating-status visualization while the robot moved through different
inspection aisles.
During actual inspection operations, the virtual robot was examined to determine
whether it appeared in the inspection aisle and local position corresponding to those of
the physical robot and whether its position was continuously updated according to the
uploaded SLAM coordinates. When the robot changed its direction of motion within the
same inspection aisle without changing its body heading, the virtual robot was expected to
update only its position while maintaining its model orientation. When the robot physically
turned at the end of an aisle and its heading angle changed, the virtual robot was expected
to adjust its orientation accordingly.
In addition to position and orientation, the WebGL digital twin interface was examined
to determine whether robot velocity, lifting-platform height, battery level, charging status,
current task, and task progress could be synchronously updated in the status panel. These
observations were used to comprehensively evaluate the synchronization between the
operating status of the physical robot and its representation in the virtual environment.
2.6.4. Evaluation of Interactive Control of the Inspection Robot
To evaluate the interactive control performance between the digital twin interface
and the physical inspection robot, control-command records were collected for inspection
task control, waypoint navigation and motion control, return-to-charge operation, and
emergency stopping. The evaluated command records were accumulated during field
operation rather than generated through purposefully repeated validation trials with equal
sample sizes across command categories. Inspection task control included task initiation,
pausing, and resumption, whereas waypoint navigation and motion control included way-
point navigation and related movement operations executed during inspection. Because
waypoint-navigation and motion commands were repeatedly invoked during routine in-
spection routes, their accumulated number was substantially larger than those of task-level
and safety-related commands. Emergency-stop commands were invoked only for safety
verification or abnormal-state handling and were therefore much less frequent.
After a user submitted a control request through the WebGL interface, the request was
first transmitted to the cloud platform back end. Based on the current operating status of
the robot, the cloud platform validated the command parameters, assessed the task state,
and encapsulated the control command before forwarding it to the robot. After executing
the requested operation, the robot returned its task status, current position, execution
progress, and execution result to the cloud platform, and the updated information was
synchronously displayed on the digital twin interface.
A control operation was considered successful when the robot completed the intended
action or entered the target task state and returned valid execution feedback. An operation
was considered unsuccessful when the robot failed to complete the intended action, failed
https://doi.org/10.3390/agriculture16171845

## Page 19 (chars=3973, imgs=0)

Agriculture 2026, 16, 1845
19 of 31
to enter the target task state, or did not return valid feedback. The execution success rate of
a control command was defined as the proportion of successful executions relative to the
total number of evaluated command records for that command type.
Control feedback time was defined as the interval between receipt of the control
request by the cloud platform server and receipt of the first valid execution feedback from
the robot. Recording both request and feedback timestamps on the cloud platform server
reduced the influence of clock asynchrony among the digital twin interface, cloud platform,
and robot-side devices on the timing results.
The execution stability and feedback timeliness of remote inspection robot control
through the digital twin interface were comprehensively evaluated by calculating the
number of command records, number of successful executions, execution success rate, and
average feedback time for each control function.
Using the above evaluation methods, the system was assessed in terms of digital
twin model optimization and WebGL runtime performance, cage-level localization and
three-dimensional mapping of dead-bird events, virtual–physical synchronization of the
inspection robot, and remote interactive control. The corresponding results are presented
in Section 3.
3. Results
To validate the applicability of the developed digital twin system to abnormality mon-
itoring in a cage-reared broiler house, the results were presented from three perspectives:
integration and deployment of the digital twin monitoring platform, cage-level spatial
mapping and information linkage of abnormality events, and virtual–physical synchro-
nization and interactive control of the inspection robot. The evaluation focused on whether
inspection data from the physical broiler house could be received, organized, and delivered
through the cloud platform and subsequently used to drive abnormality visualization, robot
status synchronization, and interactive control within the WebGL digital twin environment.
3.1. Digital Twin Broiler-House Construction and WebGL Deployment
Based on the actual dimensions of the experimental broiler house, the cage arrange-
ment, inspection aisles, and layout of the major facilities, the digital twin broiler-house
environment, cloud-based data services, and Unity WebGL three-dimensional interface
were integrated and deployed. The main broiler-house structure, cage array, inspection
aisles, ventilation and lighting facilities, and robot operating areas within the digital twin
environment were organized according to their spatial relationships in the physical broiler
house. The cages were parametrically generated according to the actual configuration of
7 rows, 4 tiers, and 63 groups, ensuring that the cage-row relationships, tier structure, group
arrangement, and aisle positions in the virtual environment corresponded to those in the
physical broiler house.
After scene construction and system function integration, the Unity environment was
published in WebGL format and embedded into the cloud platform interface, forming the
digital twin broiler-house monitoring platform shown in Figure 8. The three-dimensional
broiler-house environment served as the main visualization area and synchronously dis-
played the virtual robot position, inspection task status, equipment operating information,
and number of abnormality events. The platform also provided functions including scene-
view switching, abnormality filtering and localization, robot-following visualization, and
interactive control. By integrating the spatial structure of the broiler house, robot operating
status, abnormality event information, and task control functions into a unified interface,
the platform provided a visual environment for subsequent spatial mapping of abnormality
events and virtual–physical synchronization of the inspection robot.
https://doi.org/10.3390/agriculture16171845

## Page 20 (chars=2584, imgs=1)

Agriculture 2026, 16, 1845
20 of 31
Figure 8. Interface of the Unity WebGL-based digital twin broiler-house monitoring platform. The
asterisk indicates the currently selected operating mode, the arrows indicate movement directions,
and the square indicates the stop command. Colors are mainly used for interface distinction, with
red highlighting stop and emergency controls.
During parameterized generation of the cage array, the system simultaneously estab-
lished semantic anchors covering all cage locations. Each cage location was assigned a
unique cage-row–tier–group identifier. For example, “C03-T02-G025” denotes the cage
located in row 3, tier 2, and group 25. A one-to-one correspondence was established be-
tween each cage identifier and its associated semantic anchor, providing a unified index
linking abnormality event records, cloud-based operational data, and three-dimensional
objects in Unity. Therefore, the constructed environment not only represented the overall
spatial structure of the broiler house but also supported cage-level retrieval, localization,
and marker visualization of abnormality events.
Because the complete broiler-house environment contained large numbers of repetitive
cages, cage frames, and auxiliary facilities, noncritical models were optimized while key
structures, including the cage array, inspection aisles, major equipment locations, robot
operating areas, and abnormality localization boundaries, were retained. Repeated cage
instances also reused shared mesh and material resources. The geometric complexity,
resource size, and initial loading performance before and after model optimization are
presented in Table 4.
Table 4. Comparison of digital twin broiler-house model optimization and WebGL deployment
performance.
Indicator
Before
Optimization
After Optimization
Reduction Rate
Number of triangles
802,597
414,619
48.34%
Number of vertices
406,296
282,504
30.47%
Model file size (MB)
15.4
8.5
44.81%
Initial loading time (s)
3.84
3.05
20.57%
After optimization, the triangle count decreased from 802,597 to 414,619, representing
a reduction of 48.34%; the vertex count decreased from 406,296 to 282,504, corresponding to
a reduction of 30.47%; the model file size decreased from 15.4 MB to 8.5 MB, representing
a reduction of 44.81%; and the initial WebGL scene loading time decreased from 3.84 s to
3.05 s, corresponding to a reduction of 20.57%. These results indicate that model optimiza-
tion effectively reduced scene geometric complexity and resource size and shortened the
https://doi.org/10.3390/agriculture16171845

## Page 21 (chars=3174, imgs=1)

Agriculture 2026, 16, 1845
21 of 31
initial browser loading time while preserving the overall broiler-house layout, cage-row
and tier relationships, and key spatial localization boundaries.
To further evaluate the effect of increasing abnormality-marker loads on scene per-
formance, 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, and 500 abnormality markers were
loaded into the scenes before and after optimization. Changes in the average frame rate
under different marker-load conditions are shown in Figure 9.
Figure 9. Changes in the average WebGL scene frame rate before and after optimization under
different abnormality-marker loads.
As the number of abnormality markers increased, the average frame rates of both the
pre- and post-optimization scenes decreased. However, the optimized scene maintained a
relatively higher frame rate under all test conditions. With 50 abnormality markers, the
optimized scene achieved an average frame rate of 165.23 fps. When the number of markers
increased to 200, 300, and 500, the average frame rates were 136.29, 96.63, and 67.83 fps,
respectively. Under the 500-marker condition, the average frame rate increased from
63.00 fps before optimization to 67.83 fps after optimization, representing an improvement
of 7.67%.
According to production records collected during the experimental period, the num-
ber of unprocessed dead-bird abnormality markers simultaneously retained in a single
broiler house was generally low under normal production conditions. Therefore, 50 abnor-
mality markers were sufficient to cover typical daily visualization requirements, whereas
100–500 markers were mainly used to simulate high-load conditions, including concen-
trated abnormality occurrence, delayed event processing, and the accumulation of his-
torical abnormality records. Under all test conditions, scene rotation, zooming, panning,
abnormality-marker selection, event-detail retrieval, and robot status updating functioned
normally. These results indicate that the optimized WebGL environment could meet
routine abnormality visualization and interaction requirements while retaining a certain
performance margin as the number of abnormality markers increased substantially.
3.2. Semantic Cage-Location Calculation and Three-Dimensional Mapping of Abnormality Events
Based on SLAM Poses
To validate semantic cage-location calculation and three-dimensional mapping of ab-
normality events, dead-bird events with clearly documented field locations throughout the
entire production cycle were used for quantitative evaluation. The events included in the
test covered different inspection aisles, camera viewing sides, and cage tiers, enabling assess-
ment of cage-level localization and spatial mapping under different operating conditions.
Based on the robot SLAM pose, inspection aisle identifier, and camera identifier
associated with each dead-bird event, the system calculated the corresponding cage-row,
tier, and group identifiers and generated a complete cage identifier in the format “Cxx-Txx-
Gxxx”. The identifier was then used to retrieve the corresponding semantic anchor in the
https://doi.org/10.3390/agriculture16171845

## Page 22 (chars=3642, imgs=0)

Agriculture 2026, 16, 1845
22 of 31
Unity digital twin environment, thereby converting the robot’s continuous position into a
discrete semantic cage location and its associated three-dimensional position.
During field testing, personnel manually verified each detected dead-bird event with
a clearly recorded location and documented the actual cage row, tier, and group in which
the affected bird was located. The system-calculated results were then compared with the
manual verification records. An event was considered correctly localized at the cage level
only when the calculated cage-row, tier, and group identifiers were all consistent with the
corresponding manual records. The system-calculated and manually verified results for
selected representative events are presented in Table 5.
Table 5. Semantic cage-location calculation and field verification results for representative dead-bird
events.
Event ID
Inspection Round
Manually Verified Cage ID
Computed Cage ID
Result
E001
R01
C03-T02-G025
C03-T02-G025
Correct
E002
R01
C05-T04-G038
C05-T04-G038
Correct
E003
R02
C02-T01-G017
C02-T01-G017
Correct
E004
R02
C06-T03-G042
C06-T03-G041
Incorrect
E005
R03
C04-T02-G056
C04-T02-G056
Correct
As shown in Table 5, the system-calculated cage locations were consistent with the
manual verification results for most representative events. Among the 81 incorrectly
localized events, all errors involved a one-group displacement between adjacent cage
groups. Specifically, 59 events were assigned to the following cage group, whereas 22 events
were assigned to the preceding cage group. No errors involving displacement greater than
one cage group were observed, and no cage-row or cage-tier identification errors occurred.
For example, the manually verified cage location of event E004 was C06-T03-G042, whereas
the system-calculated result was C06-T03-G041. The cage-row and tier identifiers were
correct, and only the cage-group identifier differed by one.
These one-group localization errors occurred mainly near the boundaries between
adjacent cage groups. When the robot was positioned close to a group boundary, local
fluctuations in the SLAM pose and boundary sensitivity during the discretization of contin-
uous spatial positions could cause the calculated result to be assigned to the preceding or
following group. In contrast, the cage-row and tier identifiers were primarily determined
by the fixed inspection aisle–camera mapping relationships and were therefore less affected
by short-term fluctuations in robot position.
Based on event-by-event comparisons between the system-calculated and manually
verified cage locations throughout the entire production cycle, 1954 of the 2035 dead-
bird events included in the evaluation had cage-row, tier, and group identifiers that were
all consistent with the manual verification records. The resulting cage-level localization
accuracy for dead-bird events was 96.0%. This result reflects the combined practical
performance of event-record construction, cage-row and tier determination, cage-group
calculation, and complete cage-identifier generation.
After a valid cage identifier had been generated, the system used it as an index to
retrieve the corresponding semantic anchor in the Unity environment and generated a three-
dimensional abnormality marker near the target cage. To improve spatial recognizability,
the marker was displayed above the cage location and pointed toward the target cage using
a guiding line or positional indicator. The cage-level three-dimensional mapping result for
a dead-bird event is shown in Figure 10a.
https://doi.org/10.3390/agriculture16171845

## Page 23 (chars=2435, imgs=1)

Agriculture 2026, 16, 1845
23 of 31
Figure 10. Cage-level three-dimensional marking and detail retrieval for a dead-bird event. (a) Abnor-
mality marker and cage-level localization in the digital twin environment; (b) abnormality evidence,
cage-location information, and associated environmental parameters.
When a user selected a three-dimensional abnormality marker, the WebGL inter-
face retrieved the corresponding record from the cloud platform using the event identi-
fier and displayed information including the abnormality type, occurrence time, confi-
dence score, complete cage identifier, abnormality image, robot SLAM pose, environmen-
tal parameters recorded at the corresponding time, and processing status, as shown in
Figure 10b. Through the linkage between abnormality markers and event details, dead-bird
abnormalities were transformed from isolated detection images or back-end records into
spatial event objects with explicit cage locations, abnormality evidence, and associated
environmental information.
For dead-bird events with valid cage identifiers, the system completed semantic-
anchor retrieval, three-dimensional abnormality-marker generation, and event-detail asso-
ciation, thereby enabling information linkage from semantic cage-location calculation to
spatial visualization in the digital twin environment. The position of each three-dimensional
marker was consistent with that of the corresponding semantic cage anchor. After selecting
a marker, users could further inspect the abnormality evidence and associated environ-
mental information, providing spatial guidance for locating the target cage and conducting
on-site verification.
In addition to dead-bird events, the platform also integrated upstream auxiliary screen-
ing results for open-mouth breathing, infrared abnormalities, and acoustic abnormalities.
Open-mouth breathing and infrared abnormalities were displayed at the candidate-cage
or local-region level according to the robot position, camera identifier, and other available
localization information associated with each event. Acoustic abnormalities generally
lacked explicit camera viewing-side and cage-tier information and were therefore repre-
sented mainly at the inspection-aisle or local-region level according to the robot’s aisle and
local position at the time of acquisition. This hierarchical visualization strategy prevented
https://doi.org/10.3390/agriculture16171845

## Page 24 (chars=3932, imgs=0)

Agriculture 2026, 16, 1845
24 of 31
auxiliary abnormalities from being incorrectly assigned to a unique cage when the available
localization information was insufficient.
It should be noted that only dead-bird events were manually verified throughout the
entire production cycle and used to calculate cage-level localization accuracy. Open-mouth
breathing, infrared abnormalities, and acoustic abnormalities were used primarily to evalu-
ate the platform’s ability to uniformly integrate multiple types of auxiliary screening results
and support hierarchical spatial visualization and event-detail retrieval. Their upstream
recognition accuracy and cage-level localization performance were not quantitatively eval-
uated. Therefore, the reported localization accuracy of 96.0% applies only to the dead-bird
events included in the field verification.
3.3. Virtual–Physical Synchronization and Interactive Control of the Inspection Robot
3.3.1. Affine Calibration and Virtual–Physical Robot Synchronization
To establish the spatial correspondence between the SLAM map coordinate system of
the inspection robot and the Unity digital twin coordinate system, six calibration points
with clearly defined spatial correspondences were selected. One point was positioned at
the front and rear ends of inspection aisles 1, 5, and 8, respectively. These three aisles were
located on one side, in the middle, and on the opposite side of the broiler house along its
width, whereas the front and rear calibration points covered different longitudinal regions.
The calibration points were therefore distributed across the broiler-house floor plane.
The robot was sequentially moved to each calibration position, and the corresponding
SLAM map coordinates were recorded. The reference coordinates of the corresponding
positions were then obtained from the Unity digital twin environment. The six pairs of
SLAM map and Unity scene coordinates were used to construct an overdetermined system
of equations, and the parameters of the two-dimensional affine transformation were jointly
estimated using the least-squares method. After parameter estimation, the SLAM map
coordinates of each calibration point were substituted back into the affine transformation
model to obtain the corresponding transformed Unity coordinates. The planar Euclidean
distance between each transformed coordinate and its reference coordinate in the Unity
environment was calculated as the fitting residual. The calibration-point coordinates and
fitting results are presented in Table 6.
Table 6. Coordinates and fitting residuals of the SLAM–Unity affine calibration points.
Calibration Point
SLAM Map
Coordinates (m)
Reference Unity
Coordinates (m)
Transformed Unity
Coordinates (m)
Fitting Residual
(m)
P1
(−0.170, −0.450)
(5.417, −2.971)
(5.422765, −2.945523)
0.026122
P2
(−0.190, −82.060)
(35.159, −162.141)
(35.153453, −162.122922)
0.018909
P3
(8.590, −0.310)
(12.232, −2.020)
(12.232374, −2.076325)
0.056326
P4
(8.550, −80.970)
(41.602, −159.353)
(41.601129, −159.402160)
0.049168
P5
(14.720, −0.180)
(16.992, −1.437)
(16.985870, −1.405608)
0.031985
P6
(14.730, −81.780)
(46.730, −160.592)
(46.736408, −160.561462)
0.031203
As shown in Table 6, the fitting residuals of the six calibration points ranged from
0.018909 to 0.056326 m. Based on these residuals, the mean, root-mean-square error (RMSE),
and maximum fitting residuals were 0.035619, 0.037918, and 0.056326 m, respectively.
Because all six points were used to estimate the final affine transformation, these values
characterize the fitting consistency of the model with the complete calibration-point set
rather than its predictive performance for points excluded from parameter estimation.
To further assess the predictive stability of the affine transformation using the available
field measurements, leave-one-out cross-validation (LOOCV) was performed. In each
https://doi.org/10.3390/agriculture16171845

## Page 25 (chars=2928, imgs=1)

Agriculture 2026, 16, 1845
25 of 31
iteration, one calibration point was excluded from parameter estimation, and the affine
transformation was re-estimated using the remaining five points. The excluded point was
then treated as a held-out validation sample, and its SLAM coordinates were transformed
using the model obtained from the other five points. The predicted Unity coordinates were
compared with the corresponding reference coordinates. The LOOCV results are presented
in Table 7.
Table 7. Leave-one-out cross-validation results for the SLAM–Unity affine transformation.
Held-Out Point
Reference Unity Coordinates
(m)
Predicted Unity Coordinates
(m)
Held-Out Prediction Error (m)
P1
(5.417, −2.971)
(5.431804, −2.905579)
0.067075
P2
(35.159, −162.141)
(35.144674, −162.094308)
0.048840
P3
(12.232, −2.020)
(12.232564, −2.104906)
0.084907
P4
(41.602, −159.353)
(41.600697, −159.426545)
0.073557
P5
(16.992, −1.437)
(16.978319, −1.366939)
0.071384
P6
(46.730, −160.592)
(46.744449, −160.523144)
0.070355
The held-out prediction errors ranged from 0.048840 to 0.084907 m. The mean predic-
tion error was 0.069353 m, the RMSE was 0.070178 m, and the maximum prediction error
was 0.084907 m. Although the held-out prediction errors were higher than the fitting resid-
uals obtained using the complete calibration-point set, all six errors remained below 0.1 m.
These results indicate that the affine transformation retained relatively stable predictive
performance when individual calibration points were excluded from parameter estimation.
During operation, the virtual robot was displayed in the inspection aisle and local
position corresponding to those of the physical robot, and its position was continuously up-
dated according to the SLAM coordinates. When the robot moved backward while its body
heading angle remained unchanged, only the position of the virtual robot was updated,
whereas its model orientation remained unchanged. When the physical robot turned at the
end of an aisle and its heading angle changed, the virtual robot synchronously adjusted its
orientation. This prevented backward movement from being incorrectly visualized as a
robot turn.
In addition to position and orientation, information including robot speed, lifting-
platform height, battery level, charging status, current task, and task progress was syn-
chronously updated in the Unity WebGL status panel. The corresponding results are
shown in Figure 11. Taken together, the calibration-point fitting results, LOOCV results,
and actual operating performance demonstrate that the established coordinate mapping
and status synchronization mechanism supported continuous position updating, orien-
tation visualization, and operating-status display of the virtual robot within the digital
twin environment.
Figure 11. Interface for virtual–physical synchronization and robot status visualization.
https://doi.org/10.3390/agriculture16171845

## Page 26 (chars=2378, imgs=1)

Agriculture 2026, 16, 1845
26 of 31
3.3.2. Interactive Control Test Results for the Inspection Robot
Following completion of virtual–physical robot synchronization, the interactive control
functions provided through the Unity WebGL digital twin interface were further evaluated.
The system included automatic inspection and manual control modes. The automatic
inspection mode was mainly used to start, pause, and resume inspection tasks. The manual
control mode supported forward movement, backward movement, turning, stopping, and
target-position navigation, together with return-to-charge and emergency-stop functions.
The robot interactive control interface is shown in Figure 12.
Figure 12. Interactive control interface for the inspection robot. The asterisk indicates the selected
operating mode; the X mark closes the panel; the arrows and square indicate movement directions
and stopping, respectively. Red highlights stop and emergency controls, while the other colors are
used for visual distinction.
During the execution of inspection tasks or manual control commands, the cloud
platform continuously received data including the robot position, heading angle, operating
speed, lifting-platform height, battery level, charging status, task status, and task progress
and transmitted the corresponding information to the WebGL digital twin interface. The
virtual robot updated its position within the three-dimensional broiler house using the
transformed coordinates and adjusted its orientation according to the heading information.
Meanwhile, the status panel synchronously displayed the robot device status, current
position, speed, battery level, and task information.
After a user submitted a control request through the WebGL interface, the cloud
platform back end validated the command parameters, evaluated the current task status,
encapsulated the command, and forwarded it to the robot. After completing the target
action or entering the corresponding task state, the robot returned its latest position, task
status, and execution result to the cloud platform, and the digital twin interface was
updated accordingly. For the deployed control functions, the number of command records,
number of successful executions, execution success rate, and mean feedback time were
recorded. The results are presented in Table 8.
https://doi.org/10.3390/agriculture16171845

## Page 27 (chars=3947, imgs=0)

Agriculture 2026, 16, 1845
27 of 31
Table 8. Interactive control-command evaluation results for the inspection robot.
Control Command
Number of
Command Records
Successful
Executions
Success Rate
(%)
Mean Feedback
Time (s)
Inspection task control (start/pause/resume)
864
861
99.65
1.2
Waypoint navigation and motion control
4982
4956
99.48
1.0
Return to charging station
528
525
99.43
1.0
Emergency stop
38
38
100.00
1.0
As shown in Table 8, 864 inspection-task-control command records were evaluated,
of which 861 were successfully executed, corresponding to a success rate of 99.65%. A
total of 4982 waypoint-navigation and motion-control command records were evaluated,
with 4956 successful executions and a success rate of 99.48%. For the return-to-charge
operation, 528 command records were evaluated, of which 525 were successfully executed,
corresponding to a success rate of 99.43%. In addition, 38 emergency-stop command records
were evaluated, and all were successfully executed, yielding a success rate of 100.00%. The
mean feedback time for the different control-command types ranged from 1.0 to 1.2 s. The
substantial difference in command counts reflected their different frequencies of use during
field operation. Waypoint navigation and motion commands were repeatedly issued
as the robot progressed through predefined inspection routes and therefore accounted
for most control records. In contrast, return-to-charge and task-level commands were
generally triggered only at specific stages of an inspection task, while emergency-stop
commands were intentionally invoked much less frequently because they were safety-
related operations rather than routine control actions.
A total of 6412 control-command records were evaluated across all functions, of which
6380 were successfully executed, resulting in an overall execution success rate of 99.50%. A
request was considered successful when the robot completed the target action or entered
the intended task state and returned valid feedback. Requests that did not return valid
feedback or failed to enter the intended task state were classified as unsuccessful.
The control-command evaluation results demonstrate that inspection, manual move-
ment, return-to-charge, and emergency-stop requests initiated through the WebGL digital
twin interface could be validated and forwarded through the cloud platform. Following
execution, the robot task status and execution results were returned to the cloud plat-
form and synchronously updated in the digital twin interface. The platform therefore
enabled bidirectional data interaction among robot status uploading, virtual-scene up-
dating, control-command delivery, and execution-result feedback, thereby supporting
inspection task management and remote operational monitoring of robots in cage-reared
broiler houses.
4. Discussion
4.1. Semantic Cage-Level Localization and Three-Dimensional Mapping of Abnormality Events
Existing poultry abnormality-monitoring and robotic inspection studies have primar-
ily focused on abnormality recognition and autonomous inspection [7,17,19,25,28], whereas
continuous robot coordinates cannot directly correspond to the cage-row, tier, and group
identifiers used in practical production management. In this study, robot SLAM positions,
inspection aisle identifiers, camera identifiers, and cage arrangement parameters were
combined to convert continuous robot positions into semantic cage locations, achieving a
cage-level localization accuracy of 96.0% for dead-bird events. All 81 incorrectly localized
events involved a one-group displacement between adjacent cage groups, with no larger
group-displacement, cage-row, or cage-tier errors observed, indicating that localization
uncertainty was concentrated at cage-group boundaries and was mainly associated with
local SLAM pose fluctuations and the sensitivity of discrete group assignment near spa-
https://doi.org/10.3390/agriculture16171845

## Page 28 (chars=4083, imgs=0)

Agriculture 2026, 16, 1845
28 of 31
tial boundaries. Semantic cage anchors further reduced the dependence of abnormality
mapping on specific model names or mesh coordinates by establishing a stable corre-
spondence between cage identifiers and three-dimensional locations, thereby facilitating
model adjustment and potential adaptation to other regularly arranged cage-reared broiler
houses. Open-mouth breathing, infrared abnormalities, and acoustic abnormalities were
incorporated mainly for multisource data integration, hierarchical spatial visualization, and
event-detail retrieval; therefore, the reported 96.0% localization accuracy applies specifically
to dead-bird cage-level localization and should not be generalized to these supplementary
abnormality types. Because the field evaluation was conducted in a single commercial
broiler house over one production cycle, the transferability of the cage-mapping method to
other housing layouts requires further validation.
4.2. Virtual–Physical Robot Synchronization, WebGL Deployment, and Interactive Control
Digital twin technologies have been increasingly investigated in agriculture and live-
stock farming for environmental representation, equipment-state monitoring, production
management, and decision support [29,35,39]. In the present study, a two-dimensional
affine transformation was used to map robot SLAM coordinates into the Unity environ-
ment, enabling synchronized visualization of robot position, orientation, and operating
status. The synchronization mechanism distinguished movement direction from body
orientation so that reverse motion without a heading change did not result in an incorrect
rotation of the virtual robot. The additional LOOCV analysis provided a retrospective
out-of-sample assessment of the coordinate transformation using the available field mea-
surements; all held-out prediction errors remained below 0.1 m. However, because the
same six spatial locations were reused in the LOOCV procedure, additional independent
validation points are still required to assess mapping performance at intermediate aisle
and turning locations. Model optimization reduced browser-side geometric and resource
demands, and the optimized scene maintained an average frame rate of 67.83 fps with
500 simultaneously displayed abnormality markers under the tested hardware conditions.
Interactive control-command evaluation further yielded an overall execution success rate
of 99.50%, with mean feedback times of 1.0–1.2 s, indicating a stable control and status-
feedback link among the WebGL interface, cloud platform, and physical robot. Compared
with conventional two-dimensional maps and back-end task lists, the integrated interface
provided a more intuitive spatial context by jointly presenting robot status, surrounding
cages, and abnormality locations. Future work should focus on 3–5 additional independent
spatial validation points, semi-automatic calibration, boundary-refinement strategies such
as multi-frame pose smoothing, inspection-aisle centerline projection, and buffer-zone or
hysteresis rules, as well as autonomous local reinspection and quantitative evaluation of
supplementary abnormality types.
5. Conclusions
This study developed and field-tested a digital twin platform integrating an inspection
robot, semantic cage-location mapping, three-dimensional visualization, cloud-based data
services, and remote interactive control for a commercial cage-reared broiler house. Based
on the physical dimensions, cage arrangement, and inspection-aisle layout of the broiler
house, a parameterized digital twin environment comprising 7 cage rows, 4 tiers, and
63 groups was constructed. Semantic anchors were established for all 1764 cage locations,
providing a unified association among cage identifiers, abnormality event records, and
three-dimensional spatial positions.
Model optimization reduced the triangle count, vertex count, and model file size by
48.34%, 30.47%, and 44.81%, respectively, and shortened the initial WebGL loading time
https://doi.org/10.3390/agriculture16171845

## Page 29 (chars=4146, imgs=0)

Agriculture 2026, 16, 1845
29 of 31
from 3.84 to 3.05 s. The optimized scene maintained an average frame rate of 67.83 fps
when 500 abnormality markers were displayed simultaneously, while scene navigation,
marker selection, event-detail retrieval, and robot-status updating remained functional. The
semantic cage-location calculation method combined robot SLAM poses, inspection-aisle
identifiers, camera identifiers, and cage arrangement parameters. Among 2035 dead-bird
events evaluated throughout the production cycle, 1954 were correctly localized in terms
of cage row, tier, and group, resulting in a cage-level localization accuracy of 96.0%. All
81 localization errors involved a one-group displacement between adjacent cage groups,
indicating that cage-group boundaries remained sensitive to SLAM pose fluctuations and
discrete group assignment.
A two-dimensional affine transformation based on six calibration points was used
to establish the coordinate correspondence between the SLAM map and the Unity en-
vironment. This enabled synchronized visualization of the robot’s position, orientation,
velocity, lifting-platform height, battery level, charging status, and task progress. A total
of 6412 robot control-command records were evaluated, of which 6380 were successfully
executed, yielding an overall execution success rate of 99.50%. Mean feedback times for the
different command categories ranged from 1.0 to 1.2 s.
Overall, the developed platform integrated dead-bird cage localization, hierarchical
visualization of multisource abnormalities, virtual–physical robot synchronization, and
remote operational control within a unified digital twin environment. The system provides
practical support for abnormality visualization, cage-level localization, subsequent on-site
verification, inspection-process tracking, and robot operation management in multi-tier
cage-reared broiler houses. However, the current evaluation was conducted in one broiler
house over a single production cycle, and only dead-bird events were quantitatively evalu-
ated for cage-level localization. Future work should therefore focus on cross-environment
deployment, further validation of virtual–physical synchronization, boundary-localization
refinement, semi-automatic coordinate calibration, and long-term quantitative evaluation
of open-mouth breathing, infrared abnormalities, and acoustic abnormalities.
Author Contributions: S.L.: writing—original draft, software, methodology, investigation, validation,
data curation, writing—review and editing. W.Z.: writing—review and editing, data curation,
investigation, supervision. D.H.: investigation, data curation, methodology. H.Z.: investigation, data
curation. J.S. (Jingkun Sun): investigation, data curation, methodology. J.S. (Jiaze Sun): data curation,
methodology. J.S. (Jingkun Sun): investigation, data curation, methodology. J.S. (Jiaze Sun): data
curation, methodology. C.C.: resources, supervision, methodology, investigation, funding acquisition.
All authors have read and agreed to the published version of the manuscript.
Funding: This research was funded by the earmarked fund for the China Agriculture Research System
(CARS-40) and the National Key Research and Development Program of China (2023YFD2000800).
Institutional Review Board Statement: Ethical review and approval were waived for this study
because it involved only non-invasive data collection of broiler chickens without any direct handling,
intervention, or manipulation of the animals.
Data Availability Statement: The data are not publicly available at this time due to ongoing related
research but are available from the corresponding author upon reasonable request.
Acknowledgments: This work was supported by the earmarked fund for CARS-40 and the National
Key R&D Program of China (2023YFD2000800). The authors also acknowledge the support of the Key
Laboratory of Smart Breeding (Co-construction by Ministry and Province), Ministry of Agriculture
and Rural Affairs, Tianjin Agricultural University.
Conflicts of Interest: The authors declare no conflict of interest.
https://doi.org/10.3390/agriculture16171845

## Page 30 (chars=5463, imgs=0)

Agriculture 2026, 16, 1845
30 of 31
References
1.
Berckmans, D. General Introduction to Precision Livestock Farming. Anim. Front. 2017, 7, 6–11. [CrossRef]
2.
Norton, T.; Chen, C.; Larsen, M.L.V.; Berckmans, D. Review: Precision Livestock Farming: Building “Digital Representations” to
Bring the Animals Closer to the Farmer. Animal 2019, 13, 3009–3017. [CrossRef] [PubMed]
3.
Li, N.; Ren, Z.; Li, D.; Zeng, L. Review: Automated Techniques for Monitoring the Behaviour and Welfare of Broilers and Laying
Hens: Towards the Goal of Precision Livestock Farming. Animal 2020, 14, 617–625. [CrossRef] [PubMed]
4.
Rowe, E.; Dawkins, M.S.; Gebhardt-Henrich, S.G. A Systematic Review of Precision Livestock Farming in the Poultry Sector: Is
Technology Focussed on Improving Bird Welfare? Animals 2019, 9, 614. [CrossRef] [PubMed]
5.
Astill, J.; Dara, R.A.; Fraser, E.D.G.; Roberts, B.; Sharif, S. Smart Poultry Management: Smart Sensors, Big Data, and the Internet of
Things. Comput. Electron. Agric. 2020, 170, 105291. [CrossRef]
6.
Ojo, R.O.; Ajayi, A.O.; Owolabi, H.A.; Oyedele, L.O.; Akanbi, L.A. Internet of Things and Machine Learning Techniques in Poultry
Health and Welfare Management: A Systematic Literature Review. Comput. Electron. Agric. 2022, 200, 107266. [CrossRef]
7.
Okinda, C.; Nyalala, I.; Korohou, T.; Okinda, C.; Wang, J.; Achieng, T.; Wamalwa, P.; Mang, T.; Shen, M. A Review on Computer
Vision Systems in Monitoring of Poultry: A Welfare Perspective. Artif. Intell. Agric. 2020, 4, 184–208. [CrossRef]
8.
Fang, C.; Zhuang, X.; Zheng, H.; Yang, J.; Zhang, T. The Posture Detection Method of Caged Chickens Based on Computer Vision.
Animals 2024, 14, 3059. [CrossRef] [PubMed]
9.
Yang, J.; Zhang, T.; Fang, C.; Zheng, H. A Defencing Algorithm Based on Deep Learning Improves the Detection Accuracy of
Caged Chickens. Comput. Electron. Agric. 2023, 204, 107501. [CrossRef]
10.
Guo, Y.; Aggrey, S.E.; Oladeinde, A.; Johnson, J.; Zock, G.; Chai, L. A Machine Vision-Based Method Optimized for Restoring
Broiler Chicken Images Occluded by Feeding and Drinking Equipment. Animals 2021, 11, 123. [CrossRef] [PubMed]
11.
Guo, Y.; Aggrey, S.E.; Wang, P.; Oladeinde, A.; Chai, L. Monitoring Behaviors of Broiler Chickens at Different Ages with Deep
Learning. Animals 2022, 12, 3390. [CrossRef] [PubMed]
12.
Aydin, A. Using 3D Vision Camera System to Automatically Assess the Level of Inactivity in Broiler Chickens. Comput. Electron.
Agric. 2017, 135, 4–10. [CrossRef]
13.
Zhuang, X.; Bi, M.; Guo, J.; Wu, S.; Zhang, T. Development of an Early Warning Algorithm to Detect Sick Broilers. Comput.
Electron. Agric. 2018, 144, 102–113. [CrossRef]
14.
Okinda, C.; Lu, M.; Liu, L.; Nyalala, I.; Muneri, C.; Wang, J.; Zhang, H.; Shen, M. A Machine Vision System for Early Detection
and Prediction of Sick Birds: A Broiler Chicken Model. Biosyst. Eng. 2019, 188, 229–242. [CrossRef]
15.
Zhuang, X.; Zhang, T. Detection of Sick Broilers by Digital Image Processing and Deep Learning. Biosyst. Eng. 2019, 179, 106–116.
[CrossRef]
16.
Ma, Y.; Guo, Y.; Gao, B.; Zheng, P.; Chen, C. Development of a Tool to Detect Open-Mouthed Respiration in Caged Broilers.
Animals 2025, 15, 2732. [CrossRef] [PubMed]
17.
McManus, C.; Tanure, C.B.; Peripolli, V.; Seixas, L.; Fischer, V.; Gabbi, A.M.; Menegassi, S.R.O.; Stumpf, M.T.; Kolling, G.J.; Dias,
E.; et al. Infrared Thermography in Animal Production: An Overview. Comput. Electron. Agric. 2016, 123, 10–16. [CrossRef]
18.
Huang, J.; Wang, W.; Zhang, T. Method for Detecting Avian Influenza Disease of Chickens Based on Sound Analysis. Biosyst. Eng.
2019, 180, 16–24. [CrossRef]
19.
Carpentier, L.; Vranken, E.; Berckmans, D.; Paeshuyse, J.; Norton, T. Development of Sound-Based Poultry Health Monitoring
Tool for Automated Sneeze Detection. Comput. Electron. Agric. 2019, 162, 573–581. [CrossRef]
20.
Cuan, K.; Zhang, T.; Huang, J.; Fang, C.; Guan, Y. Detection of Avian Influenza-Infected Chickens Based on a Chicken Sound
Convolutional Neural Network. Comput. Electron. Agric. 2020, 178, 105688. [CrossRef]
21.
Hao, H.; Fang, P.; Duan, E.; Yang, Z.; Wang, L.; Wang, H. A Dead Broiler Inspection System for Large-Scale Breeding Farms Based
on Deep Learning. Agriculture 2022, 12, 1176. [CrossRef]
22.
Liu, H.-W.; Chen, C.-H.; Tsai, Y.-C.; Hsieh, K.-W.; Lin, H.-T. Identifying Images of Dead Chickens with a Chicken Removal System
Integrated with a Deep Learning Algorithm. Sensors 2021, 21, 3579. [CrossRef] [PubMed]
23.
Luo, S.; Ma, Y.; Jiang, F.; Wang, H.; Tong, Q.; Wang, L. Dead Laying Hens Detection Using TIR-NIR-Depth Images and Deep
Learning on a Commercial Farm. Animals 2023, 13, 1861. [CrossRef] [PubMed]
24.
Yang, J.; Zhang, T.; Fang, C.; Zheng, H.; Ma, C.; Wu, Z. A Detection Method for Dead Caged Hens Based on Improved YOLOv7.
Comput. Electron. Agric. 2024, 226, 109388. [CrossRef]
25.
Ma, W.; Wang, X.; Yang, S.X.; Xue, X.; Li, M.; Wang, R.; Yu, L.; Song, L.; Li, Q. Autonomous Inspection Robot for Dead Laying
Hens in Caged Layer House. Comput. Electron. Agric. 2024, 227, 109595. [CrossRef]
26.
Ham, G.-S.; Oh, K. Dead Broiler Detection and Segmentation Using Transformer-Based Dual Stream Network. Agriculture 2024,
14, 2082. [CrossRef]
27.
Vroegindeweij, B.A.; Blaauw, S.K.; IJsselmuiden, J.M.M.; Van Henten, E.J. Evaluation of the Performance of PoultryBot, an
Autonomous Mobile Robotic Platform for Poultry Houses. Biosyst. Eng. 2018, 174, 295–315. [CrossRef]
https://doi.org/10.3390/agriculture16171845

## Page 31 (chars=2499, imgs=0)

Agriculture 2026, 16, 1845
31 of 31
28.
Özentürk, U.; Chen, Z.; Jamone, L.; Versace, E. Robotics for Poultry Farming: Challenges and Opportunities. Comput. Electron.
Agric. 2024, 226, 109411. [CrossRef]
29.
Verdouw, C.; Tekinerdogan, B.; Beulens, A.; Wolfert, S. Digital Twins in Smart Farming. Agric. Syst. 2021, 189, 103046. [CrossRef]
30.
Pylianidis, C.; Osinga, S.; Athanasiadis, I.N. Introducing Digital Twins to Agriculture. Comput. Electron. Agric. 2021, 184, 105942.
[CrossRef]
31.
Neethirajan, S.; Kemp, B. Digital Twins in Livestock Farming. Animals 2021, 11, 1008. [CrossRef] [PubMed]
32.
Purcell, W.; Neubauer, T. Digital Twins in Agriculture: A State-of-the-Art Review. Smart Agric. Technol. 2023, 3, 100094. [CrossRef]
33.
Tagarakis, A.C.; Benos, L.; Kyriakarakos, G.; Pearson, S.; Sorensen, C.G.; Bochtis, D. Digital Twins in Agriculture and Forestry: A
Review. Sensors 2024, 24, 3117. [CrossRef] [PubMed]
34.
Arulmozhi, E.; Deb, N.C.; Tamrakar, N.; Kang, D.Y.; Kang, M.Y.; Kook, J.; Basak, J.K.; Kim, H.T. From Reality to Virtuality:
Revolutionizing Livestock Farming Through Digital Twins. Agriculture 2024, 14, 2231. [CrossRef]
35.
Symeonaki, E.; Maraveas, C.; Arvanitis, K.G. Recent Advances in Digital Twins for Agriculture 5.0: Applications and Open Issues
in Livestock Production Systems. Appl. Sci. 2024, 14, 686. [CrossRef]
36.
Zhang, Y.; Zhang, Y.; Gao, M.; Dai, B.; Kou, S.; Wang, X.; Fu, X.; Shen, W. Digital Twin Perception and Modeling Method for
Feeding Behavior of Dairy Cows. Comput. Electron. Agric. 2023, 214, 108181. [CrossRef]
37.
Alves, R.G.; Maia, R.F.; Lima, F. Development of a Digital Twin for Smart Farming: Irrigation Management System for Water
Saving. J. Clean. Prod. 2023, 388, 135920. [CrossRef]
38.
Zhu, H.; Zhao, G.; Wang, H.; Luo, S.; Sun, J.; Hao, D.; Chen, C. Design and Application of a Cloud Platform for Broiler Inspection
Robots. Agriculture 2026, 16, 1201. [CrossRef]
39.
Neethirajan, S. Digital Twins for Cows and Chickens: From Hype Cycles to Hard Evidence in Precision Livestock Farming.
Agriculture 2026, 16, 166. [CrossRef]
Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.
https://doi.org/10.3390/agriculture16171845
