# Seeing Text in the Dark: Algorithm and Benchmark

Chengpei Xu<sup>*1</sup>, Hao Fu<sup>2</sup>, Long Ma<sup>†2</sup>, Wenjing Jia<sup>3</sup>, Chengqi Zhang<sup>3</sup>, Feng Xia<sup>4</sup>, Xiaoyu Ai<sup>1</sup>, Binghao Li<sup>1</sup>, and Wenjie Zhang 

<sup>1</sup>University of New South Wales 

<sup>2</sup>Dalian University of Technology <sup>3</sup>University of Technology Sydney <sup>4</sup>RMIT University 

## Abstract

Localizing text in low-light environments is challenging due to visual degradations. Although a straightforward solution involves a two-stage pipeline with low-light image enhancement (LLE) as the initial step followed by detector, LLE is primarily designed for human vision instead of machine and can accumulate errors. In this work, we propose an efficient and effective single-stage approach for localizing text in dark that circumvents the need for LLE. We introduce a constrained learning module as an auxiliary mechanism during the training stage of the text detector. This module is designed to guide the text detector in preserving textual spatial features amidst feature map resizing, thus minimizing the loss of spatial information in texts under low-light visual degradations. Specifically, we incorporate spatial reconstruction and spatial semantic constraints within this module to ensure the text detector acquires essential positional and contextual range knowledge. Our approach enhances the original text detector’s ability to identify text’s local topological features using a dynamic snake feature pyramid network and adopts a bottom-up contour shaping strategy with a novel rectangular accumulation technique for accurate delineation of streamlined text features. In addition, we present a comprehensive lowlight dataset for arbitrary-shaped text, encompassing diverse scenes and languages. Notably, our method achieves state-of-the-art results on this low-light dataset and exhibits comparable performance on standard normal light datasets. The code and dataset will be released. 

## 1 Introduction

Scene text detection plays a crucial role in multimedia understanding, laying the groundwork for tasks such as text recognition, information extraction, and scene understanding. Generally, approaches to arbitrary scene text detection can be divided into two main categories: Typically, topdown methods [27, 43, 13, 25, 22, 23, 35, 41, 30] primarily focus on either directly or incrementally delineating the entire contour and area of the text. In contrast, typical bottomup [21, 16, 31, 10, 33, 32] approaches first identify individual text components and then assemble them together. Both top-down and bottom-up methods, each with their unique strengths, have undergone rapid development and achieved remarkable results in the field of arbitrary shape text detection. Although significant progress has been made in text detection, detecting arbitrary shape text in low light conditions remains a significant challenge. Insufficient lighting leads to visual degradations such as blurred details, reduced brightness and contrast, and distorted color representation, making it difficult for both humans and text detectors to locate text. Figure 1 illustrates some examples of the text in low-light scenes. 

A straightforward two-stage approach to solving lowlight text detection involves enhancing low-light text images using existing LLE methods such as [17, 42, 40], followed by applying text detectors to the enhanced images. However, mainstream LLE methods, tailored for better visual effect for human vision, often overlook the need of downstream detection tasks. This can lead to the degradation of text’s inherent features through brightness or color adjustments, often resulting in detection failures, as shown in Figure 1. Furthermore, the lack of dedicated datasets for arbitrary shape text detection in low-light conditions has significantly hampered the validation of methods in realworld scenarios. This has led to an over-reliance on synthetic data, exacerbating a notable domain gap in low-light arbitrary shape text detection research. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/8c0028a459c0233bfbfd51539b9066ea708f10b24312de9ab401cfa584f62a17.jpg)



Figure 1: (a) The visual statistics and examples of low-light text images of different scenes, languages, and lighting conditions in the newly curated LATeD dataset. All images are enhanced for clearer vision. (b) Single-stage text detectors originally designed for normal lighting conditions struggle with low-light images. Even with low-light enhancement and fine-tuned with [37] following a two-stage step, the results remain unsatisfactory. This is because the enhancer, aimed at overall improvement of visibility, may inadvertently compromise text features.


Consequently, we step outside the “enhance-first, detectlater” framework and propose a one-stage solution without any pre-processing enhancement. To address the issue of spatial information loss in normal-light text detectors caused by degradation factors such as low illumination and low contrast, we design a spatial-constrained learning module during the training process. 

This module, guided by two constraints, i.e., Spatial Reconstruction Constraint and Spatial Semantic Constraint, effectively preserves and pinpoints both the positional and contextual range information of text, which are often at risk of being lost during the spatial resizing of feature maps. 

We further enhance the text detector by focusing on the intrinsic characteristics of text and adapting text shaping methods to low-light scenarios. Unlike general objects, text possesses a unique topological distribution and streamline structure. We incorporate Dynamic Snake Convolution [19] (DSC), noted for its prowess in preserving tubular topological features, alongside conventional convolutions to capture the intrinsic characteristics of text in parallel. Additionally, we design a self-attention gate to control the proportions of convolutions and construct a novel feature pyramid network [9] (FPN) structure, thereby significantly enriching the representation of local topological features of text through improved fusion steps. 

Regarding text’s streamline characteristics, the top-down text detection approaches are less suitable under low-light conditions due to limited receptive field and difficulties in global information acquisition. Therefore, we adopt a more flexible bottom-up modeling strategy, employing an innovative rectangular accumulation approach for text contour modeling. This strategy enables the creation of a streamlined and flexible representation of text contours, alleviating the issue of limited receptive fields. 

The major contributions of this paper are fourfold: 

1) For the first time, we propose a one-stage pipeline for low-light arbitrary-shaped text detection that effectively utilizes spatial constraint to adeptly guide the training process. 

2) We devise a novel method concentrating on the extraction of topological distribution features and the modeling of streamline characteristics to shape low-light text contours effectively. 

3) We curate the first low-light arbitrary-shape text dataset (LATeD) featuring 13,923 multilingual and arbitrary shape texts across diverse low-light scenes, as shown in Figure 1, effectively bridging the existing domain gap. 

4) Our method, employing a constrained learning strategy and capturing intrinsic text features, attains state-of-theart results on low-light text detection dataset without image enhancement modules and also excels over SOTAs on similarly well-lit datasets. 

## 2 Related Works

Arbitrary Shape Text Detection: Generally, arbitraryshape scene text detection approaches can be divided into two main categories: the top-down approaches [27, 43, 

13, 25, 22, 23, 35] and bottom-up approaches [21, 16, 31, 10, 33]. Top-down approaches tend to estimate the overall contour and area of the text directly or progressively. Some methods, such as those in [8, 35, 27, 26, 25], view text detection as a segmentation task, using the contours of text masks to delineate final text boundaries. Concurrently, various top-down techniques have transitioned from segmentation mask prediction to contour control point generation, employing curve functions or boundary transformation [13, 39, 23, 43]. For instance, TextRay [23] exploits Chebyshev curves, while ABCNet [13] and FCENet [43] utilize Bezier and Fourier curves, respectively. The inher- ´ ent challenge for top-down methods lies in the necessity to assimilate extensive global information and enhance the receptive field to adequately capture the final contours and areas of text. 

Bottom-up methods usually begin by pinpointing potential text components and subsequently combine them, offering a counter to the challenges posed by top-down methods, notably regarding receptive field constraints and long-range dependency capture. Bottom-up methods such as [21, 15] employ heuristic rules to merge text blocks. Approaches like [38, 16, 29] make use of convolutional graph neural networks [6] to integrate text blocks. Bottom-up methods, to some extent, adopt a divide-and-conquer approach, which facilitates more flexible shape representation and higher tolerance for receptive field limitations. However, they often require complex post-processing and NMS (Non-Maximum Suppression) to generate suitable text components, leading to a rise in computational costs. 

Although arbitrary-shaped text detection is a prominent research topic, current studies rarely address scenarios with insufficient illumination. Our arbitrary shape text detector adopts a bottom-up design, modeling the topological and streamline features of text under low-light conditions while optimizing the selection of text components and computational complexity. 

Low-Light Image Enhancement: Low-light image enhancement aims to uncover image details hidden in dark areas, thereby improving image quality. Methods like KinD [40] and ZeroDCE [3] refine existing models with new training losses and heuristic quadratic curves, while approaches such as [12, 17] leverage Retinex-inspired frameworks and self-calibrated illumination for robust enhancement in low-light conditions. It is worth noting that existing low-light enhancement methods primarily aim to improve images’ visual quality for human vision and often neglect the need of machine vision of the downstream tasks. Therefore, enhancing human visual experience does not necessarily translate to improved performance of the downstream machine vision tasks. In this paper, we move beyond the conventional framework of LLE-then-detection, and develop a low-light text detector based on constrained learning and effectively capturing the intrinsic characteristics of text. 

Text Detection Datasets in Low-Light Scenes: When it comes to text detection datasets in low-light scenes, the existing datasets for arbitrary shape text detection [34, 14, 1] have been primarily collected under normal lighting conditions. Some methods such as [5, 11] have artificially introduced noise, reduce brightness, and adjust contrast to synthetically generate low-light text images based on these mainstream datasets. However, the synthetically generated low-light text images cannot fully emulate real low-light text images in terms of pixel appearances, illumination, color, and noise intensity. Additionally, synthetic methods often target on the entire image, and cannot effectively replicate the inherent visual and semantic features of lowlight text. 

Xue et al. [33] introduced a rudimentary dataset for lowlight text detection. However, this dataset is characterized by limited diversity, including significant scene and text repetition, an average of fewer than two texts per image, and low-light text images produced by synthetically dimming daylight photos. It utilizes rectangular labeling instead of the more accurate polygonal approach and contains no curved text samples. To address these limitations and facilitate research in low-light text detection, we devise a new multilingual dataset for arbitrary shape text detection across diverse adverse scenes. Detailed description and comparison of the new dataset can be found in Section 4. 

## 3 The Proposed Method

## 3.1 Base Text Detector

The general training process of a base normal-light text detector can be formulated as: 

$$
\min _ {\boldsymbol {\theta}} \mathcal {L} _ {\mathrm{t}} (\Psi (\mathbf {u}; \boldsymbol {\theta})),\tag{1}
$$

where ${ \mathcal { L } } _ { \mathrm { t } }$ represents the training loss used to constrain the detected output derived from the observation u using the text detector Ψ with a learnable parameter θ. In environments degraded by low light, the inherent process of reducing spatial dimensions in deeper feature maps of text detectors exacerbates the risk of losing or inaccurately capturing vital text spatial information, such as positional and contextual range details, thereby increasing the rate of false detections. 

## 3.2 Spatial-Constrained Modeling

To address the challenges posed by low-light conditions as previously described, we design a new learning constraint that integrates spatial information into the learning process of text detector $\Psi .$ . This formulation can be expressed as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/738b4138c10d14e80439601fd17f4b8cdcca9a665b2bac0dc34b90d3495d0d2f.jpg)



Figure 2: The overall structure of the proposed method, where $^ { 6 6 } 1 / 1 , 2 5 6 ^ { , 9 }$ ...indicate the resize ratio and the channel number. The SCM is only employed during the training stage for assisting spatial information awareness of low-light text.


$$
\begin{array}{c} \min _ {\boldsymbol {\theta}} \mathcal {L} _ {\mathrm{t}} \big (\Psi (\mathbf {u}; \boldsymbol {\theta} (\boldsymbol {\vartheta} ^ {*})) \big), \\ s. t., \boldsymbol {\vartheta} ^ {*} = \arg \min _ {\boldsymbol {\vartheta}} \mathcal {L} _ {\mathrm{s}} \big (\Phi (\mathbf {u}; \boldsymbol {\vartheta} (\boldsymbol {\theta})) \big). \end{array}\tag{2}
$$

Here, the loss $\mathcal { L } _ { \mathrm { s } }$ represents the newly introduced spatial constraint, designed to extract crucial position and semantic range details of text in low-light conditions from a spatial auxiliary learning module Φ (the Spatial-Constrained Learning Module (SCM) in Figure 2) with learnable parameters ϑ to aid in the training process. The objective of the spatial constrained modeling is to identify an optimal $\vartheta ^ { \ast }$ that simultaneously minimizes the loss of SCM and the base text detector. To find the optimal $\vartheta ^ { \ast }$ in low-light condition, we design the spatial constraint from a dual-level perspective. 

## 3.2.1 Spatial Reconstruction Constraint.

The first level, termed spatial reconstruction loss $\mathcal { L } _ { \mathrm { s r } } ,$ is designed to enable the network to acquire a wealth of valuable information concerning the reconstruction of textual positions, thereby ensuring the preservation of textual spatial information throughout the process of reducing the dimensions of feature maps. In the training phase, we begin by creating a text position mask utilizing label information from the ground truth. Next, the output features $( \mathcal { C } _ { 0 }$ in Figure 2) are upsampled and integrated with the positional embedding via an element-wise operation. The resulting merged features are then directed to a specially designed decoder for alignment with the ground truth. 

## 3.2.2 Spatial Semantic Constraint.

The second level, known as spatial semantic loss ${ \mathcal { L } } _ { \mathrm { s s } } ,$ aims to align the main network and the auxiliary learning module on the contextual semantic features of text after spatial reconstruction. While the spatial reconstruction loss aids in reclaiming some lost spatial details, the semantic feature map produced by the auxiliary learning module offers contextual range information vital for text detection tasks. The $\mathcal { L } _ { \mathrm { s s } }$ can help to bridge the contextual semantic gap between the auxiliary learning branch and original text detection branch, and emphasizes the text region with greater focus in both low-light and normal-light conditions. 

## 3.3 Bottom-up Text Topological Modeling

Upon introducing spatial constraints, we enhance the base text detector from the perspective of reinforcing the expression and modeling of text’s topological structure, aiming to better adapt to low-light environments. 

## 3.3.1 Dynamic Snake FPN (DSF)

Text typically exhibits a slender distribution along its central line, with the strokes of the characters branching out in various curved directions. Therefore, we have redesigned the feature pyramid structure, leveraging the advantages of both DSC and conventional convolution by parallelizing these two convolution operations. This integration meticulously aims to capture the intricate local topological features inherent in textual elements, leveraging both the DSC branch and the regular convolution branch to enhance the detection of text’s topological, visual, and semantic features 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/090aab36249d2ad72fef23acebfa6e2552946f5df3628149a285f22760d91664.jpg)



Input


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/b2d91656f2f65750c1f963de11d95fe42143e6ca0ff13f143b9d9e4c9d7e99a0.jpg)



w/o SCM+DSF


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/39964cb918fd8ada2289052518c0b25ec213cf7686419bdf933f0530874b6119.jpg)



SCM+DSF


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/e2e7a716ff862ec0344f156da93d3839ff6571dfc2145a0aee76c175d9c1f047.jpg)



TSR



Figure 3: Our approach, bolstered by SCM, DSF, and TSR, effectively captures both the topological structure and the center location of text components. This is demonstrated in the text center heatmaps (3rd column) and the streamlined text shaping (4th column). For clarity, images are enhanced. Here, $\mathrm { ^ { 6 6 } w / o ^ { 3 9 } }$ denotes “without”.


The DSF is mainly composed of several perceptual modulation blocks stacked. As shown in Figure 2, the perceptual modulation block consists of three branches, including a DSC branch, a regular convolution branch, and a gated self-attention prediction branch, and the formulation can be written as: 

$$
\mathcal {X} _ {i} = \operatorname{concat} \left(\mathcal {C} _ {i}, \mathcal {F} _ {i - 1}\right),\tag{3}
$$

$$
\mathcal {V} = \operatorname{concat} \left(\operatorname{Conv} \left(\mathcal {X} _ {i}\right), D S C \left(\mathcal {X} _ {i}\right)\right),\tag{4}
$$

$$
\mathcal {F} _ {i} = \text {softmax} (\frac {\sigma (\mathcal {W} _ {Q} \cdot \mathcal {V} + b _ {Q}) (\sigma (\mathcal {W} _ {K} \cdot \mathcal {V} + b _ {K})) ^ {T}}{\sqrt {d _ {k}}}) \cdot \mathcal {V}.\tag{5}
$$

Here, X<sub>i</sub> represents the concatenation of feature map Ci with $\mathcal { F } i - 1$ , upon which we further concatenate the outputs from both DSC and conventional convolution applied to $\mathcal { X } _ { i }$ , denoted as V. The $W _ { Q }$ and $W _ { K }$ are weight matrices for queries and keys, respectively, $b _ { Q }$ and $b _ { K }$ are bias terms, σ represents the nonlinear activation function, $d _ { k }$ are the dimensionality of the keys. 

## 3.3.2 Text Shaping with Rotated Rectangular Accumulation (TSR)

Following the text topology capture with DSF, we further employ a bottom-up shaping approach to enhance the expression of text’s streamlined topology. Unlike some topdown methods that rely on text feature map for text shaping, the bottom-up modeling tolerates errors and needs less intact text feature maps. 

For delineating text contours across various lighting conditions, our process commences with the generation of two typical text feature maps for scene text detection from the final layer of DSF: a text map and a text center region map. Concurrently, this layer also generates the geometric attributes of the rotated rectangle that comprises the text component, represented by $( x , y , h , w , \theta )$ . Here, $( x , y )$ designate the center of the rectangle, while $( h , w , \theta )$ determine the rectangle’s height, width, and angular orientation. 

Taking inspiration from the integration concept, which converts the calculation of an area into the summation of many small blocks, we accumulate the rotated rectangular text components to better fit the natural shapes of text and thereafter generate text contours. Here, we fixed the width of these rotated rectangles. While the text center region mask accurately locates the centers of the rotated rectangles, it can potentially produce an overwhelming number of candidate rectangles. Recent bottom-up approaches [38, 31, 33, 16] commonly use NMS to filter through these candidates. However, NMS does not consider the streamline distribution characteristics of text, which can result in a failure to ensure a topological distribution consistent with the central region of the text. Furthermore, NMS depends on certain degrees of randomness and parameter settings, necessitating multiple calculations of overlaps among rotated rectangles and thereby increasing the computational burden. 

To address these challenges, our work abandons the NMS approach and instead adopts Farthest Point Sampling [2] to filter potential text components. Farthest Point Sampling often used in point cloud processing to reduce the size point cloud while attempting to preserve the geometrical and topological structure of the data. Here we use Farthest Point Sampling to effectively reduces the number of potential text segment centers while preserving the linear streamline characteristics of the text center regions. Here we consider $P$ as the final set of potential text components center. During each selection of potential center selection, we ensure the chosen point $p$ in P satisfies the condition of being the farthest from any point in the text center region T. This is achieved by computing the minimum Euclidean 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/a21cb95a6183044bcc6117670a5154a684dba247a4c1fb32b276fb124840ec0a.jpg)



Figure 4: Text shaping with rotated rectangular accumulation on normal light text images. As shown in the 2nd row, our method remains unaffected by obstructions, showing reliable performance in modeling text streamline features.


distance d to $T \colon$ : 

$$
p = \operatorname * {a r g m a x} _ {p \in P} \min _ {t \in T} \| p - t \| _ {2}.\tag{6}
$$

The dotted lines (rotated rectangle center) in the 4th column of Figure 3 and 1st column in Figure 4 display the effective filtering of representative text center points using the Farthest Point Sampling technique. The cumulative shape of rotated rectangular text components accurately reflect the text’s actual contours (more visual examples are in Supplementary). To accumulate these text components, our SCM approach and the DSF have also preserved the spatial position and topological structure of the text center regions. Therefore, based on this foundation, we simply need to carry out a series of straightforward morphological closing operations on each text center to fill the minor gaps between text components and within their interiors. As shown in the 3rd column of Figure 3, even when some text center regions generated with SCM and DSF support are not particularly complete due to severe visual degradation, the TSR still have a chance to accumulate the text regions. 

## 3.4 Loss Function

The overall loss function under spatial constrains is formulated as: 

$$
\mathcal {L} _ {\mathrm{t}} = \mathcal {L} _ {\text { s   e   g }} + \mathcal {L} _ {H} + \mathcal {L} _ {\theta} + \mathcal {L} _ {\mathrm{ss}} + \mathcal {L} _ {\mathrm{sr}}.\tag{7}
$$

Here, $\mathcal { L } _ { \mathrm { s e g } }$ is the sum of cross-entropy losses for the text map and text center region. Here, $\mathcal { L } _ { H }$ and $\mathcal { L } _ { \theta }$ represent the smoothed L1 losses for the height and rotation angle of the text component, respectively. The $\mathcal { L } _ { \mathrm { s s } }$ is L2 loss and the $\mathcal { L } _ { \mathrm { s r } }$ is the L1 loss for the text spatial constraint. 

## 4 LATeD: Low-Light Arbitrary-Shape Text Detection Dataset

The existing low-light text detection dataset [33] presents several issues: on average, it contains fewer than two texts per image, exhibits significant scene and text repetition, uses rectangular labeling instead of the more precise polygon labeling, and is limited in size with no curved text samples. Additionally, ambiguity regarding the composition of the test and training sets, along with the absence of specified evaluation criteria, hinders the effective use and reproducibility of this dataset. These challenges significantly impede the development of low-light arbitrary scene text detection. 

Therefore, we created LATeD, a new dataset manually curated during nighttime from multilingual communities in Sydney and Melbourne, and tailored specially for arbitraryshape scene text detection in low-light environments. 

Drawing inspiration from the flagship dataset CTW1500 [14], which was created for text detection under normal lighting conditions, LATeD surpasses CTW1500 in both the total text count and the number of curved texts (see Table 1). The LATeD dataset consists of 1,500 images (1,000 for training and 500 for testing), featuring 13,923 arbitrarily shaped texts, with 4,554 being curved texts. Each text is accompanied by precise line-level polygon annotations, providing resource for low-light text detection research. 

Furthermore, LATeD is a multilingual dataset, primarily composed of English and Chinese texts while also containing languages such as Japanese, Korean, Vietnamese, and Arabic. The dataset covers a broad spectrum of low-light scenes, ranging from indoor to outdoor settings and encompassing various mediums such as standard printed posters to packages, clothes, billboards, road signs, graffiti, and more. 

Detailed statistics regarding light condition, scene and languages of the two datasets can be found in Table 1 with the visual statistics and examples of images in the LATeD dataset shown in Figure 1. Notably, every image in LATeD was 100% manually curated during nighttime from multilingual communities in Sydney and Melbourne. To assess text detection accuracy, we simply follow the evaluation protocol from CTW1500. 

## 5 Experiments

To demonstrate the superior performance of the proposed text detector, we conducted series of experiments comparing our approach with SOTA methods. For low-light text detection, we utilized the newly created low-light text dataset LATeD. For normal-light text detection, we utilized CTW1500, Total-text [1] and MSRA-TD500 [34]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/f8f867c507fff0b6c760085011e59cede005c4ed5b84c6e5bf297daada5614be.jpg)



(f) URetinex Enhanced


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/2cb6804158bb5c431a3b466c79592861357f551a4e3fdfafcdce07535fea260d.jpg)



(g) Fully-trained [37]



(h) Manually Enhanced


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/5d0e611c6581542a53bbe4ba87c7b2e64bc5282cd3b788478df7523ada649816.jpg)



(i) Ground Truth


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/ce4408f2aac8810966ae4db4d58f8bf6b310a800208f76a802d6f69f09734dc5.jpg)



(j) Ours



Figure 5: Visual comparison of the text detection results on images from the LATeD dataset. (a)-(f) Texts detected with BPN++ fine-tuned using images enhanced with six different LLE techniques. (g) Texts detected with fully trained BPN++ using low-light images. (h) Manually enhanced images. (i) Input low-light images with ground-truth bounding boxes. (j) Text detected with our method.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Illumination</td><td rowspan="2">Annotation Type</td><td colspan="2">Size</td><td rowspan="2">Text Length</td><td colspan="3">Text Types</td><td rowspan="2">Repeat Scene/Text</td></tr><tr><td>Train</td><td>Test</td><td>Multi.</td><td>Curve</td><td>Total</td></tr><tr><td>Dataset in [33]</td><td>Low Light</td><td>Rectangle</td><td>300</td><td>200</td><td>Short</td><td>-</td><td>-</td><td>766</td><td>Very High</td></tr><tr><td>CTW1500</td><td>Normal Light</td><td>Polygon</td><td>1,000</td><td>500</td><td>Long</td><td>7,221</td><td>3,530</td><td>10,751</td><td>Low</td></tr><tr><td>LATeD</td><td>Low Light</td><td>Polygon</td><td>1,000</td><td>500</td><td>Long</td><td>9,369</td><td>4,554</td><td>13,923</td><td>Low</td></tr></table>


Table 1: Comparison between the existing datasets and the newly constructed LATeD dataset. Multi. denotes Multi-oriented.


## 5.1 Implementation Details

The backbone of our network is ResNet50. We first pre-trained it on the SynthText dataset [4] for 2 epochs using input images of size 640 × 640 pixels. Subsequently, we fine-tuned it for an additional 100 epochs on the MLT dataset [18]. We employed the Adam optimizer with a learning rate initialized at 0.0001, which was reduced by a factor of 0.1 every 100 epochs. Our data augmentation strategy included random cropping, resizing, color adjustments, noise injection, flipping, and rotation. The batch size was set to 10 and the training epoch was set to 250. The training was conducted on a single NVIDIA RTX A6000 GPU, supported by a 3.60GHz Intel Xeon Gold 5122 CPU. During testing, input images were resized to 640 × 640 for LATeD and CTW1500, and 640×1200 for Total-Text and 640×960 for MSRA-TD500. 

## 5.2 Comparison on Low-Light Text Detection

We benchmarked eleven state-of-the-art (SOTA) methods for arbitrary shape scene text detection and six SOTA low-light enhancement techniques across four settings: text detection on low-light images (“Detection”), text detection on enhanced images (“LLE”), text detector fine-tuned with enhanced images (“FINE-TUNE”), and fully trained text detector (“FULLY-TRAINED”). 

Table 2 compares the detailed detection results of our approach with the SOTA approaches under all four settings. Figure 5 visualize the text detection results of two exemplar images from the LATeD dataset using different approaches. 

Setting 1: Text Detection on Low-light Images (Detection): we first tested eleven text detectors originally designed for normal light conditions (pretrained on the CTW1500) on low-light images. As shown in Table 2, unsurprisingly, all of the examined text detectors performed poorly under low-light conditions with none exceeding 50% F1 scores and BPN++ performing the best. This shows the limited generalizability of existing text detectors in dealing with low-light conditions. 

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="3">LATeD</td></tr><tr><td>P(%)</td><td>R(%)</td><td>F1(%)</td></tr><tr><td colspan="6">Scene text detection (Detection)</td></tr><tr><td rowspan="11">Detection</td><td>PSENet [25]</td><td>CVPR&#x27;19</td><td>53.7</td><td>7.5</td><td>13.1</td></tr><tr><td>PAN [26]</td><td>ICCV&#x27;19</td><td>56.4</td><td>8.1</td><td>14.2</td></tr><tr><td>DB [7]</td><td>AAAI&#x27;20</td><td>4.6</td><td>9.8</td><td>13.2</td></tr><tr><td>DRRG [38]</td><td>CVPR&#x27;20</td><td>73.2</td><td>12.2</td><td>20.9</td></tr><tr><td>ContourNet [38]</td><td>CVPR&#x27;20</td><td>61.2</td><td>19.5</td><td>24.1</td></tr><tr><td>TextFuseNet [35]</td><td>IJCAI&#x27;20</td><td>21.8</td><td>91.8</td><td>37.4</td></tr><tr><td>FCEnet [43]</td><td>CVPR&#x27;21</td><td>15.5</td><td>72.4</td><td>32.1</td></tr><tr><td>TextBPN [39]</td><td>ICCV&#x27;21</td><td>21.9</td><td>65.9</td><td>42.7</td></tr><tr><td>DB++ [8]</td><td>TPAMI&#x27;22</td><td>61.2</td><td>19.6</td><td>29.6</td></tr><tr><td>DPText [36]</td><td>AAAI&#x27;23</td><td>86.7</td><td>30.6</td><td>45.3</td></tr><tr><td>BPN++ [37]</td><td>TMM&#x27;23</td><td>73.6</td><td>35.3</td><td>47.7</td></tr><tr><td colspan="6">Text detection on enhanced images (LLE)</td></tr><tr><td rowspan="6">LLE</td><td>KinD [40]</td><td>MM&#x27;19</td><td>88.4</td><td>35.9</td><td>51.1</td></tr><tr><td>ZeroDCE [3]</td><td>CVPR&#x27;20</td><td>90.8</td><td>37.7</td><td>53.3</td></tr><tr><td>RUAS [12]</td><td>CVPR&#x27;21</td><td>90.4</td><td>36.5</td><td>52.0</td></tr><tr><td>SCI [17]</td><td>CVPR&#x27;22</td><td>91.1</td><td>34.0</td><td>49.5</td></tr><tr><td>DDC [42]</td><td>CVPR&#x27;22</td><td>88.6</td><td>39.1</td><td>54.2</td></tr><tr><td>URetinex [28]</td><td>CVPR&#x27;22</td><td>89.4</td><td>34.3</td><td>49.6</td></tr><tr><td colspan="6">Fine-tuned on enhanced images (FINE-TUNE)</td></tr><tr><td rowspan="6">Fine-tune</td><td>KinD [40]</td><td>MM&#x27;19</td><td>76.0</td><td>43.6</td><td>55.4</td></tr><tr><td>ZeroDCE [3]</td><td>CVPR&#x27;20</td><td>78.1</td><td>46.6</td><td>59.1</td></tr><tr><td>RUAS [12]</td><td>CVPR&#x27;21</td><td>79.1</td><td>48.1</td><td>59.8</td></tr><tr><td>SCI [17]</td><td>CVPR&#x27;22</td><td>78.0</td><td>47.7</td><td>59.2</td></tr><tr><td>DDC [42]</td><td>CVPR&#x27;22</td><td>75.2</td><td>47.5</td><td>58.2</td></tr><tr><td>URetinex [28]</td><td>CVPR&#x27;22</td><td>72.5</td><td>40.8</td><td>52.4</td></tr><tr><td colspan="6">Fully trained text detectors (FULLY-TRAINED)</td></tr><tr><td rowspan="2">Fully Trained</td><td>BPN++ [37]</td><td>TMM&#x27;23</td><td>74.5</td><td>49.5</td><td>59.5</td></tr><tr><td>Ours</td><td>-</td><td>82.6</td><td>57.0</td><td>67.1</td></tr></table>


Table 2: Quantitative comparison of text detection results obtained on the low-light text detection dataset LATeD under the four settings.


Setting 2: Text Detection on Enhanced Images (LLE): We then applied six different LLE methods to enhance the images in the LATeD dataset, and used BPN++ as the baseline text detector due to its superior F1-score, indicating better robustness and generalizability. As shown in Figure 5 and Table 2 (the “LLE” section), although most enhancement techniques managed to improve the image visibility for human eyes, the improvement is insufficient for the downstream detectors to perceive the text clearly. These methods fall short when compared to ours, as they introduced introduce too many visual distortions and still exhibit gaps with normal light images. 

Setting 3: Fine-tuned Text Detectors with Enhanced Images (FINE-TUNE): To further validate the effectiveness of the “enhance-first, detect-later” approaches, we enhanced the training images of the LATeD dataset using various LLE methods [40, 3, 12, 17, 42, 28] and then fine-tuned the BPN++ on the enhanced training images. 

As shown in Figure 5 (a)-(f), some LLE methods such as DDC, KinD, and URetinex introduced visual distortions, impairing images’ textual information and leading to direct text detection failures. Other methods such as RUAS and SCI enhanced image brightness, but misled the downstream text detector at text boundary areas. These results, however, still cannot match the performance of our method, which effectively detects text without requiring any LLE module. The primary reason is that those enhancement techniques are not designed with downstream tasks in mind, leading to a semantic gap between enhanced low-light images and those taken in normal lighting conditions. 

Setting 4: Fully Trained Text Detector (FULLY-TRAINED): Given that LLE techniques are not tailored for downstream text detection, we trained BPN++ and our method directly on the low-light images of the LATeD dataset. As shown in Table 2, the F1-scores obtained with the LLE methods are close to that of the BPN++, suggesting LLE is tuned for enhanced human visual perception rather than maximizing downstream performance. Moreover, general text detectors such as BPN++ lack targeted guidance on text spatial information during training for low-light settings, hence falling short in effectively expressing the local topology and streamline features of text. This results in a performance that was not on par with our purposefully designed method. The detection results shown in Figure 5 also demonstrate the robustness of our method for detecting curved texts in low-light environments. 

## 5.3 Comparison on Well-lit Text Detection

In this section, we present experimental results for text detection under normal lighting using the CTW1500, Total-Text [1], and MSRA-TD500 [34] datasets, following their official evaluation protocols. Table 3 presents the details results, where “Extend” indicates the additional datasets used for pre-training in comparative methods. In our implementation, we used ResNet50 as the primary backbone for all networks, except for those mentioned in [7] and [8], which used ResNet50 with deformable convolution. 

According to Table 3, our method achieved an F1-score of 86.2% on CTW1500, surpassing current state-of-theart methods such as [8, 37]. Additionally, for arbitrarily shaped short texts at the word level, our approach established a new state-of-the-art on the Total-Text dataset with an F1-score of 88.5% and the highest recall of 86.6%. For long multi-oriented texts under normal lighting, our method achieved a performance comparable to the state-of-the-art on the MSRA-TD500 dataset. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">Extend</td><td colspan="3">CTW1500</td><td colspan="3">Total-Text</td><td colspan="3">MSRA-TD500</td></tr><tr><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td></tr><tr><td>PSENet [25]</td><td>CVPR&#x27;19</td><td>MLT</td><td>86.9</td><td>80.2</td><td>83.4</td><td>84.0</td><td>78.0</td><td>80.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DB [7]</td><td>AAAI&#x27;20</td><td>SynthText</td><td>86.9</td><td>80.2</td><td>83.4</td><td>87.1</td><td>82.5</td><td>84.7</td><td>91.5</td><td>79.2</td><td>84.9</td></tr><tr><td>ContourNet [27]</td><td>CVPR&#x27;20</td><td>-</td><td>83.7</td><td>84.1</td><td>83.9</td><td>86.9</td><td>83.9</td><td>85.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TextFuseNet[35]</td><td>IJCAI&#x27;20</td><td>SynthText</td><td>85.0</td><td>85.8</td><td>85.4</td><td>87.5</td><td>83.2</td><td>85.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DRRG [38]</td><td>CVPR&#x27;20</td><td>MLT</td><td>85.9</td><td>83.0</td><td>84.4</td><td>86.5</td><td>84.9</td><td>85.7</td><td>88.1</td><td>82.3</td><td>85.1</td></tr><tr><td>FCENet [43]</td><td>CVPR&#x27;21</td><td>-</td><td>85.7</td><td>80.7</td><td>83.1</td><td>87.4</td><td>79.8</td><td>83.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TextBPN [39]</td><td>ICCV&#x27;21</td><td>MLT</td><td>86.5</td><td>83.6</td><td>85.5</td><td>90.7</td><td>85.2</td><td>87.9</td><td>86.6</td><td>84.5</td><td>85.6</td></tr><tr><td>FSGNet [22]</td><td>CVPR&#x27;22</td><td>MLT</td><td>88.1</td><td>82.4</td><td>85.2</td><td>90.7</td><td>85.7</td><td>88.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DB++ [8]</td><td>TPAMI&#x27;22</td><td>SynthText</td><td>87.9</td><td>82.8</td><td>85.3</td><td>88.9</td><td>83.2</td><td>86.0</td><td>91.5</td><td>83.3</td><td>87.2</td></tr><tr><td>SIR [20]</td><td>MM&#x27;23</td><td>Syn</td><td>87.4</td><td>83.7</td><td>85.5</td><td>90.9</td><td>85.6</td><td>88.2</td><td>93.6</td><td>86.0</td><td>89.6</td></tr><tr><td>MTM [24]</td><td>MM&#x27;23</td><td>Syn</td><td>85.8</td><td>83.4</td><td>84.6</td><td>89.6</td><td>82.1</td><td>85.7</td><td>90.3</td><td>81.4</td><td>85.6</td></tr><tr><td>BPN++ [37]</td><td>TMM&#x27;23</td><td>MLT</td><td>87.3</td><td>83.8</td><td>85.5</td><td>91.8</td><td>85.3</td><td>88.5</td><td>89.2</td><td>85.4</td><td>87.3</td></tr><tr><td>Ours</td><td>-</td><td>MLT</td><td>88.7</td><td>83.9</td><td>86.2</td><td>90.4</td><td>86.6</td><td>88.5</td><td>91.0</td><td>83.5</td><td>87.1</td></tr></table>


Table 3: Text detection results on CTW1500, TOTAL-TEXT and MSRA-TD500.


<table><tr><td rowspan="2">SCM</td><td rowspan="2">DSF</td><td rowspan="2">TSR</td><td colspan="4">LATeD</td><td colspan="4">CTW1500</td></tr><tr><td>FPS</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>FPS</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td></tr><tr><td>×</td><td>×</td><td>×</td><td>2.1</td><td>78.8</td><td>47.9</td><td>59.6</td><td>2.2</td><td>84.1</td><td>80.6</td><td>82.3</td></tr><tr><td>×</td><td>×</td><td>√</td><td>10.9</td><td>80.8</td><td>48.2</td><td>60.4</td><td>11.1</td><td>84.2</td><td>81.0</td><td>82.6</td></tr><tr><td>√</td><td>×</td><td>√</td><td>10.9</td><td>81.5</td><td>52.7</td><td>64.0</td><td>11.1</td><td>85.1</td><td>83.4</td><td>84.2</td></tr><tr><td>×</td><td>√</td><td>√</td><td>10.2</td><td>79.8</td><td>51.1</td><td>62.3</td><td>10.4</td><td>84.9</td><td>82.9</td><td>83.8</td></tr><tr><td>√</td><td>√</td><td>√</td><td>10.2</td><td>82.8</td><td>55.4</td><td>66.4</td><td>10.4</td><td>86.2</td><td>83.5</td><td>84.8</td></tr></table>


Table 4: Ablation study on the effectiveness of the proposed SCM, DSF and TSR on the LATeD and CTW1500 dataset.


The effectiveness of our method is attributed to the combined use of SCM, DSF, and TSR, crucial for capturing text’s topological and streamline features in normal lighting. This demonstrates our method’s potential to deliver high performance and efficiency in both normal light and low light conditions. Examples of visual detection results can be found in Figure 6 and supplementary. 

## 5.4 Ablation Study

We further conducted a series of ablation studies on LATeD and CTW1500 to validate the effectiveness of the proposed modules, TSR, DSF and SCM, comparing them with a baseline with a standard FPN and 0.5 threshold NMS for rotated rectangle text components but excluding the SCM. 

As shown in Table 4, adding the text shaping module TSR into the baseline model yielded a 0.8% and 0.3% increase in F1-score on LATeD and CTW1500, respectively, and greatly enhanced inference speed. This improvement can be attributed to the farthest points sampling, which helps maintain the text component’s topological distribution. Our method retained a set number of text components, avoiding the extensive overlap calculations for numerous candidates required by NMS, thus significantly reducing computation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-22/9da04a4d-4e04-473b-a83f-727acbec68a9/9dd222bb1bd41d1ca9539a08ca4bcaad639aad9d3a78fba347506d800defb285.jpg)



Figure 6: Visualization of text detection results on well-lit datasets (see Supplementary for more visual examples).


Furthermore, after implementing the SCM and TSR, the model’s F1-score improved by 4.4% and 1.9%, respectively, without any reduction in inference speed. SCM, during training, provides cues about the text’s spatial information, aiding the network in perceiving text locations under low contrast and degraded low light conditions. Similarly, under normal lighting, SCM continues to provide spatial information for the text detection task, facilitating precise detection of the center location of text components. 

After replacing the FPN structure with our designed DSF and incorporating our text shaping method, the F1-score improved by 2.7% and 1.5% on the two datasets, respectively. These improvements are due to the DSF’s adaptive weighting of regular convolution and DSC operations, focusing on capturing the topological features of text, thereby generating more reliable text center region and geometric features of text components. As demonstrated in the third column of Figure 3, the absence of SCM and DSF led to ineffective detection of the topological structure of text center in lowlight conditions, resulting in low-confidence outcomes and inaccurate disconnections of text center. 

Utilizing SCM, DSF, and TSR, our model achieved F1- scores of 66.4% and 84.8% on the two datasets, respectively. This represents an increase of 6.8% and 2.5% in F1-score over the baseline. The efficacy of our method in both low light and normal light conditions stemmed from its precision in providing spatial cues for text location and its capacity to preserve text’s topological streamline and features. When allied with a bottom-up design approach, these attributes were instrumental in sustaining long-range dependencies, crucial for precise text detection. The fourth column’s results highlighted in Figure 3 and second row’s results in Figure 4 show the TSR method’s effectiveness in sampling text components and maintaining text streamline features. 

## 6 Conclusion

In this work, we crafted a constrained learning module that capitalizes on spacial information of text, thereby enhancing the efficacy of text detectors in low-light environments. Our approach, integrating Dynamic Snake FPN with a rectangular, bottom-up text contour shaping method, marks a significant advancement in accurately representing text’s topological distribution and streamline features. This innovative methodology has enabled us to achieve state-ofthe-art results in both low-light and normal-light text detection datasets. In addition, we have created an extensive and diverse dataset for arbitrary-shaped text specific to low-light conditions, encompassing a broad spectrum of scenes and languages, thereby significantly enhancing the resources available in this research area. 

## References



[1] C. K. Ch’ng and C. S. Chan. Total-text: A comprehensive dataset for scene text detection and recognition. In Proc. IEEE Int. Conf. on Document Anal. and Recognit., volume 1, pages 935–942. IEEE, 2017. 





[2] Y. Eldar, M. Lindenbaum, M. Porat, and Y. Y. Zeevi. The farthest point strategy for progressive image sam-





pling. IEEE Trans. Image Process., 6(9):1305–1315, 1997. 





[3] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 1780– 1789, 2020. 





[4] A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data for text localisation in natural images. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 2315–2324, 2016. 





[5] P.-H. Hsu, C.-T. Lin, C. C. Ng, J. L. Kew, M. Y. Tan, S.-H. Lai, C. S. Chan, and C. Zach. Extremely lowlight image enhancement with scene text restoration. In Int. Conf. Pattern Recog., pages 317–323. IEEE, 2022. 





[6] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. 





[7] M. Liao, Z. Wan, C. Yao, K. Chen, and X. Bai. Realtime scene text detection with differentiable binarization. In Proc. AAAI Conf. Artif. Intell., volume 34, pages 11474–11481, 2020. 





[8] M. Liao, Z. Zou, Z. Wan, C. Yao, and X. Bai. Realtime scene text detection with differentiable binarization and adaptive scale fusion. IEEE Trans. Pattern Anal. Mach. Intell., 45(1):919–931, 2022. 





[9] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan,´ and S. Belongie. Feature pyramid networks for object detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 2117–2125, 2017. 





[10] H. Liu, A. Guo, D. Jiang, Y. Hu, and B. Ren. Puzzlenet: scene text detection by segment context graph learning. arXiv preprint arXiv:2002.11371, 2020. 





[11] H. Liu, M. Yuan, T. Wang, P. Ren, and D.-M. Yan. List: low illumination scene text detector with automatic feature enhancement. The Vis. Comput., 38(9- 10):3231–3242, 2022. 





[12] R. Liu, L. Ma, J. Zhang, X. Fan, and Z. Luo. Retinexinspired unrolling with cooperative prior architecture search for low-light image enhancement. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 10561–10570, 2021. 





[13] Y. Liu, H. Chen, C. Shen, T. He, L. Jin, and L. Wang. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 9809–9818, 2020. 





[14] Y. Liu, L. Jin, S. Zhang, C. Luo, and S. Zhang. Curved scene text detection via transverse and longitudinal sequence connection. Pattern Recognit., 90:337–345, 2019. 





[15] S. Long, J. Ruan, W. Zhang, X. He, W. Wu, and C. Yao. Textsnake: A flexible representation for detecting text of arbitrary shapes. In Proc. Eur. Conf. Comput. Vision, pages 20–36, 2018. 





[16] C. Ma, L. Sun, Z. Zhong, and Q. Huo. Relatext: exploiting visual relationships for arbitrary-shaped scene text detection with graph convolutional networks. Pattern Recognit., 111:107684, 2021. 





[17] L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo. Toward fast, flexible, and robust low-light image enhancement. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 5637–5646, 2022. 





[18] N. Nayef, F. Yin, I. Bizid, H. Choi, Y. Feng, D. Karatzas, Z. Luo, U. Pal, C. Rigaud, J. Chazalon, et al. Icdar2017 robust reading challenge on multilingual scene text detection and script identificationrrc-mlt. In Proc. IEEE Int. Conf. on Document Anal. and Recognit., volume 1, pages 1454–1459. IEEE, 2017. 





[19] Y. Qi, Y. He, X. Qi, Y. Zhang, and G. Yang. Dynamic snake convolution based on topological geometric constraints for tubular structure segmentation. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 6070–6079, 2023. 





[20] X. Qin, P. Lyu, C. Zhang, Y. Zhou, K. Yao, P. Zhang, H. Lin, and W. Wang. Towards robust real-time scene text detection: From semantic to instance representation learning. In Proc. ACM Int. Conf. Multimedia, pages 2025–2034, 2023. 





[21] B. Shi, X. Bai, and S. Belongie. Detecting oriented text in natural images by linking segments. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 2550–2558, 2017. 





[22] J. Tang, W. Zhang, H. Liu, M. Yang, B. Jiang, G. Hu, and X. Bai. Few could be better than all: Feature sampling and grouping for scene text detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 4563–4572, 2022. 





[23] F. Wang, Y. Chen, F. Wu, and X. Li. Textray: Contourbased geometric modeling for arbitrary-shaped scene text detection. In Proc. ACM Int. Conf. Multimedia, pages 111–119, 2020. 





[24] K. Wang, H. Xie, Y. Wang, D. Zhang, Y. Qu, Z. Gao, and Y. Zhang. Masked text modeling: A selfsupervised pre-training method for scene text detection. In Proc. ACM Int. Conf. Multimedia, pages 2006–2015, 2023. 





[25] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 9336–9345, 2019. 





[26] W. Wang, E. Xie, X. Song, Y. Zang, W. Wang, T. Lu, G. Yu, and C. Shen. Efficient and accurate arbitraryshaped text detection with pixel aggregation network. In Proc. IEEE Int. Conf. Comput. Vision, pages 8440– 8449, 2019. 





[27] Y. Wang, H. Xie, Z.-J. Zha, M. Xing, Z. Fu, and Y. Zhang. Contournet: Taking a further step toward accurate arbitrary-shaped scene text detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 11753–11762, 2020. 





[28] W. Wu, J. Weng, P. Zhang, X. Wang, W. Yang, and J. Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 5901–5910, 2022. 





[29] C. Xu, W. Jia, T. Cui, R. Wang, Y.-f. Zhang, and X. He. Arbitrary-shape scene text detection via visual-relational rectification and contour approximation. IEEE Trans. Multimedia, 2022. 





[30] C. Xu, W. Jia, R. Wang, X. He, B. Zhao, and Y. Zhang. Semantic navigation of powerpoint-based lecture video for autonote generation. IEEE Transactions on Learning Technologies, 16(1):1–17, 2022. 





[31] C. Xu, W. Jia, R. Wang, X. Luo, and X. He. Morphtext: Deep morphology regularized accurate arbitrary-shape scene text detection. IEEE Trans. Multimedia, 2022. 





[32] C. Xu, R. Wang, S. Lin, X. Luo, B. Zhao, L. Shao, and M. Hu. Lecture2note: Automatic generation of lecture notes from slide-based educational videos. In 2019 IEEE International Conference on Multimedia and Expo (ICME), pages 898–903. IEEE, 2019. 





[33] M. Xue, P. Shivakumara, C. Zhang, Y. Xiao, T. Lu, U. Pal, D. Lopresti, and Z. Yang. Arbitrarily-oriented text detection in low light natural scene images. IEEE Trans. Multimedia, 23:2706–2720, 2020. 





[34] C. Yao, X. Bai, W. Liu, Y. Ma, and Z. Tu. Detecting texts of arbitrary orientations in natural images. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 1083–1090. IEEE, 2012. 





[35] J. Ye, Z. Chen, J. Liu, and B. Du. Textfusenet: Scene text detection with richer fused features. Proc. Int. Joint Conf. Artif. Intell., 2020. 





[36] M. Ye, J. Zhang, S. Zhao, J. Liu, B. Du, and D. Tao. Dptext-detr: Towards better scene text detection with dynamic points in transformer. In Proc. AAAI Conf. Artif. Intell., volume 37, pages 3241–3249, 2023. 





[37] S.-X. Zhang, C. Yang, X. Zhu, and X.-C. Yin. Arbitrary shape text detection via boundary transformer. IEEE Trans. Multimedia, 2023. 





[38] S.-X. Zhang, X. Zhu, J.-B. Hou, C. Liu, C. Yang, H. Wang, and X.-C. Yin. Deep relational reasoning graph network for arbitrary shape text detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 9699–9708, 2020. 





[39] S.-X. Zhang, X. Zhu, C. Yang, H. Wang, and X.-C. Yin. Adaptive boundary proposal network for arbitrary shape text detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 1305–1314, 2021. 





[40] Y. Zhang, J. Zhang, and X. Guo. Kindling the darkness: A practical low-light image enhancer. In Proc. ACM Int. Conf. Multimedia, pages 1632–1640, 2019. 





[41] Y.-f. Zhang, J. Zheng, L. Li, N. Liu, W. Jia, X. Fan, C. Xu, and X. He. Rethinking feature aggregation for deep rgb-d salient object detection. Neurocomputing, 423:463–473, 2021. 





[42] Z. Zhang, H. Zheng, R. Hong, M. Xu, S. Yan, and M. Wang. Deep color consistent network for low-light image enhancement. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 1899–1908, 2022. 





[43] Y. Zhu, J. Chen, L. Liang, Z. Kuang, L. Jin, and W. Zhang. Fourier contour embedding for arbitraryshaped text detection. In Proc. IEEE Conf. Comput. Vision and Pattern Recognit., pages 3123–3131, 2021. 

