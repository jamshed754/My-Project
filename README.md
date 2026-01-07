1.	Research Title: An Ensemble CNN-Based Framework with Explainable AI for Gastrointestinal Cancer Classification
2.	‎Description – An overview of the code/dataset.
   
2.1	Dataset Overview Description: 
This project uses the publicly available Kvasir-V2 dataset for gastrointestinal disease classification. The dataset contains three distinct classes: Esophagitis, Polyps, and Ulcerative Colitis. Each class has 1000 labeled images, making the dataset balanced for training and evaluation purposes. While the dataset size is limited for deep learning models, data augmentation techniques are applied. To enhance the training process. The Kvasir-V2 dataset provides reliable, well-standardized, and high-quality images, making it suitable for reproducible research and accurate model evaluation.
2.2 Code Overview Description: 
The code is implemented in Python using the TensorFlow framework to support deep learning model development for gastrointestinal disease classification. The repository is organized into modular scripts for reproducibility and ease of experimentation. Data preprocessing scripts handle image resizing, normalization, and basic enhancements using libraries such as NumPy, OpenCV, and Matplotlib. Advanced preprocessing techniques, including CLAHE, sharpening, and Gaussian filters, are also applied to improve the quality of input images for the models. Separate scripts are used for training multiple pre-trained CNN models, including InceptionV3, MobileNetV2, and ResNet-18. These models are trained individually, and an ensemble approach is employed to combine their predictions for final classification. Finally, explainability techniques are applied using SHAP and LIME to interpret model predictions, with dedicated scripts to generate visualizations for explainable AI analysis. This modular structure allows researchers to reproduce the experiments, modify the models, and evaluate the performance effectively.
4.	Dataset Information:
The experiments in this study utilize the publicly available Kvasir‑V2 dataset for gastrointestinal disease classification. This dataset contains three distinct pathological classes: Esophagitis, Polyps, and Ulcerative Colitis, with 1000 labeled images per class, making it balanced for initial training and evaluation. The dataset can be accessed at the following link: (https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset).To overcome the limited number of original images, extensive data augmentation techniques were applied. Each class was augmented to increase the number of images from 1000 to 4500, resulting in a total of 13,500 images across the three classes. Augmentation techniques included rotation, flipping, scaling, and translation, which enhanced the diversity of the dataset and improved the robustness of the deep learning models. All images were preprocessed by resizing to 224×224 pixels and normalization. Additional image enhancement methods, including Contrast Limited Adaptive Histogram Equalization (CLAHE), sharpening, and Gaussian filters, were applied to improve the quality of input images for the models. The dataset was split into 70% for training, 15% for validation, and 15% for testing to ensure proper model evaluation and generalization. This organized dataset supports the training, validation, and testing of multiple CNN models and ensemble approaches, providing a reliable and reproducible foundation for gastrointestinal disease classification research.
5.   Code Information:
4.1  Preprocessing:
 The dataset is initially preprocessed using Python and OpenCV to prepare it for deep learning model training. The original Kvasir-V2 images were organized in a folder structure and loaded from 'preprocessed basic'. Each image was resized to 224×224×3 pixels and normalized to standardize inputs for the CNN models. Advanced image enhancement techniques were applied to improve image quality and highlight important features. These include Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast, sharpening filters to highlight edges, and Gaussian filters for smoothing and noise reduction. Each enhancement was saved separately, maintaining the class-wise folder structure in the 'preprocessed dataset'. This preprocessing pipeline ensures that the dataset is ready for augmentation and subsequent model training. providing high-quality inputs for accurate and reproducible deep learning experiments.
4.2  Data Augmentation
Data augmentation was applied to address the limited number of training samples and to enhance the generalization capability of the deep learning models. After initial preprocessing, augmentation techniques were implemented using Python-based image processing pipelines to artificially increase dataset diversity while preserving the pathological characteristics of gastrointestinal images. Common geometric transformations, including rotation, horizontal and vertical flipping, scaling, translation, and zooming, were applied class-wise to the dataset. Through this augmentation process, the number of images in each class was increased from 1,000 to 4,500, resulting in a total of 13,500 images across the three pathological classes: esophagitis, polyps, and ulcerative colitis. The augmented dataset maintained the original class balance and the predefined data split of 70% training, 15% validation, and 15% testing. This augmentation strategy reduces overfitting, improves robustness, and enables the CNN models to learn invariant and discriminative features more effectively, supporting reproducible and reliable experimental results.
4.3 Model Training
Three convolutional neural network architectures were independently trained to perform gastrointestinal disease classification, ensuring reproducibility and robust evaluation.
4.3.1 InceptionV3 Model
The InceptionV3 model was employed as a pre-trained CNN using transfer learning to extract high-level features from endoscopic images. The network was initialized with ImageNet weights and configured without the top classification layers. All base layers were frozen to retain pre-trained representations, while a custom classification head with global average pooling, fully connected layers with ReLU activation, and dropout (0.3) was added. The final layer used Softmax activation to classify images into three pathological classes. Training was conducted using the Adam optimizer (learning rate = 1×10⁻⁴), categorical cross-entropy loss, a batch size of 32, and 15 epochs. The model was saved in Keras format for reproducibility.
4.3.2 MobileNetV2 Model
MobileNetV2 was utilized as a lightweight and efficient CNN, initialized with ImageNet weights and trained using transfer learning. All base layers were frozen, and a custom classification head was appended with global average pooling, fully connected layers (ReLU), and dropout (0.4 and 0.3) to reduce overfitting. Input images were augmented with rotation, zooming, translation, and horizontal flipping. The model was optimized using Adam, categorical cross-entropy loss, batch size of 32, and trained for 10 epochs. Early stopping and model checkpointing were applied, and the best-performing model was saved in Keras format.
4.3.3 ResNet-18 Model
A custom ResNet-18 architecture was developed using residual blocks with identity shortcuts, enabling deeper feature extraction and mitigating vanishing gradients. The network included multiple stages with increasing filter depths (64, 128, 256, 512), followed by global average pooling and a fully connected Softmax classification layer. Dropout (0.3) was applied before the final layer, and extensive data augmentation (rotation, translation, zooming, shearing, horizontal flipping) was used during training. The model was optimized with Adam (learning rate = 1×10⁻⁴) using categorical cross-entropy loss, batch size 32, and trained for 20 epochs. Early stopping, learning rate reduction, and model checkpointing ensured optimal performance, and the best model was saved in Keras format.
4.3.4 An ensemble learning approach was employed to combine the predictions of three independently trained convolutional neural networks: InceptionV3, ResNet-18, and MobileNetV2. The trained models were loaded in Keras format, ensuring reproducibility and consistent evaluation across experiments. Test images were prepared using the same preprocessing and normalization (resizing to 224×224 pixels and rescaling to [0,1]) as during model training. No additional augmentation was applied to the test data to maintain an unbiased evaluation.  Softmax outputs from each model were aggregated using simple averaging across the three networks to generate ensemble predictions. The final class label for each input image was determined by selecting the class with the highest averaged probability. This approach leverages complementary strengths of each model while reducing individual model biases. The ensemble predictions were evaluated using standard classification metrics, including accuracy, precision, recall, F1-score, and confusion matrix. Visualization of the confusion matrix provided further insight into the ensemble's classification performance across three gastrointestinal disease classes. This modular ensemble pipeline allows researchers to reproduce experiments, evaluate alternative combinations of models, and extend the ensemble with additional networks in future studies.
4.3.4 Explainable AI – SHAP & LIME
To interpret the predictions of the trained ensemble model for gastrointestinal disease classification, two complementary explainable AI (XAI) techniques were applied: SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
4.3.5 SHAP Analysis 
SHAP was used to assess the contribution of each image region to the predicted probabilities of the ensemble model. For each class (esophagitis, polyps, and ulcerative colitis), representative test images were selected. Images were resized to 224×224 pixels and normalized to match the training preprocessing.  A blurring-based masker isolated the contribution of pixel regions, generating SHAP values for each class. SHAP visualizations (heatmaps) highlight areas of the image that strongly influence model decisions, allowing global interpretability of the ensemble predictions.
4.3.5 LIME Analysis
LIME provided local interpretability for individual image predictions. Super pixel segmentation (Quick shift) was applied to partition images into coherent regions. Perturbation analysis measured the effect of each superpixel on the predicted class probabilities.  LIME output maps highlighted the most influential regions for each predicted label. Visualizations were generated for two representative images per class to demonstrate model reasoning.
4.3.6 Implementation
Dedicated Python scripts were used to load trained models, preprocess images, compute SHAP and LIME values, and generate visual explanations. This workflow ensures transparency, reproducibility, and trust in the ensemble model's predictions, enabling researchers to analyze and validate results effectively.
6. Usage Instructions- How to use or load the dataset and code
This section describes the step-by-step procedure to use the dataset and execute the provided code for gastrointestinal disease classification.
Step 1: Dataset Upload and Preprocessing
First, the raw image dataset was uploaded to the Kaggle environment in a structured folder format. The preprocessing code was executed on this folder to standardize all images to a fixed input size of 224 × 224 × 3. Pixel normalization was applied by rescaling image intensities to the range [0,1][0,1][0,1] to ensure stable model training.
Step 2: Image Enhancement
The preprocessed images were then passed through enhancement scripts to improve visual quality. Contrast Limited Adaptive Histogram Equalization (CLAHE), image sharpening, and Gaussian filtering techniques were applied. The enhanced images were automatically saved in a separate enhancement folder within the Kaggle working directory.
Step 3: Dataset Splitting
After enhancement, the dataset was split into training (70%), validation (15%), and testing (15%) subsets. Separate directories were created for each split, and these folders were stored for subsequent experiments.
Step 4: Data Augmentation
The training dataset was uploaded again to Kaggle, where data augmentation techniques such as rotation, flipping, zooming, and translation were applied. Augmentation was performed class-wise to balance the dataset, increasing the number of images to 4500 per class for three pathological classes (polyps, esophagitis, and ulcerative colitis). The augmented dataset was saved as a new directory.
Step 5: Model Training
The augmented training dataset path was provided sequentially to three deep learning models: InceptionV3, MobileNetV2, and ResNet-18. Each model was trained independently using transfer learning, and the trained model files were saved separately in Keras format.
Step 6: Ensemble Prediction
The saved model files were then loaded into the ensemble script. A softmax averaging strategy was used to combine predictions from all three models, producing the final ensemble predictions. The ensemble performance was evaluated using accuracy, classification reports, and confusion matrices.
Step 7: Explainable AI Analysis
Finally, explainability techniques were applied to the trained models. SHAP and LIME scripts were executed using the saved model files and selected test images. These scripts generated visual explanations highlighting the important regions influencing the model predictions.
Step 8: Testing and Evaluation
Each trained model and the ensemble model were evaluated on unseen test images. Test accuracy and class-wise performance metrics were recorded and reported in the results section.
7. Requirements: Any Dependencies (e.g., Python libraries).
The implementation of the proposed ensemble CNN-based framework was carried out using the Python programming language. All experiments were conducted in a Kaggle Notebook environment to ensure reproducibility and ease of execution. The following software libraries and dependencies are required to run the provided code successfully.
•	Python (version 3.8 or higher)
•	TensorFlow / Keras (TensorFlow 2.x) – used for building, training, saving, and loading deep learning models, including InceptionV3, MobileNetV2, and ResNet-18
•	NumPy – for numerical computations and array manipulations
•	OpenCV (cv2) – for image loading, resizing, preprocessing, CLAHE, sharpening, and Gaussian filtering
•	Matplotlib – for plotting training/validation accuracy, loss curves, and confusion matrices
•	scikit-learn – for performance evaluation metrics such as accuracy score, classification report, and confusion matrix
•	SHAP – for generating pixel-level explainability visualizations using a blurring-based image masker
•	LIME (lime-image) – for local interpretable explanations of model predictions
•	scikit-image – used internally by LIME for image segmentation and boundary visualization
•	Pillow (PIL) – for image loading and format handling during explainability analysis
•	OS (built-in Python module) – for directory handling and file system operations
All dependencies are commonly available in standard Kaggle environments. If executed locally, the required libraries can be installed using standard package managers such as pip.
8. Methodology (if applicable): Steps taken for data processing or modeling
This study follows a systematic and reproducible pipeline for gastrointestinal disease classification using an ensemble of deep learning models integrated with explainable AI techniques. The complete methodology consists of the following sequential steps:
Step 1: Dataset Acquisition
The publicly available Kvasir-V2 dataset was used for this study, consisting of gastrointestinal images from three pathological classes: Esophagitis, Polyps, and Ulcerative Colitis. Initially, each class contained 1000 images. The dataset was organized into class-wise folders and uploaded to the Kaggle environment for experimentation.
Step 2: Basic Image Preprocessing
All images were resized to a standard input dimension of 224 × 224 × 3 to ensure compatibility with pre-trained CNN architectures. Pixel intensity normalization was applied by scaling values to the range [0,1][0,1][0,1]. This step ensured uniformity across the dataset and improved training stability.
Step 3: Image Enhancement
To improve visual quality and emphasize disease-relevant features, advanced image enhancement techniques were applied:
•	Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast improvement
•	Sharpening filters to enhance edge details
•	Gaussian filtering to reduce noise
Enhanced images were saved into a separate directory structure while preserving class labels.
Step 4: Dataset Splitting
The enhanced dataset was divided into three subsets:
•	70% Training
•	15% Validation
•	15% Testing
This split ensured unbiased performance evaluation and prevented data leakage during the training and testing phases.
Step 5: Data Augmentation
To address limited data availability and improve model generalization, data augmentation was applied to the training set using geometric transformations such as rotation, zooming, shifting, and horizontal flipping. Through augmentation, each class was expanded to 4500 images, resulting in a total of 13,500 images across all three classes.
Step 6: Individual Model Training
Three deep learning models were trained independently using transfer learning:
•	InceptionV3
•	MobileNetV2
•	ResNet-18
Pre-trained ImageNet weights were used, and the convolutional base of each model was frozen initially. Custom classification layers, including global average pooling, dense layers, and dropout, were added to reduce overfitting. All models were trained using the Adam optimizer with categorical cross-entropy loss. Trained models were saved separately for later use.
Step 7: Ensemble Learning
An ensemble strategy based on softmax probability averaging was employed to combine predictions from the three trained models. This approach leverages complementary feature representations from different architectures, leading to improved classification robustness and accuracy.
Step 7: Ensemble Learning
An ensemble strategy based on softmax probability averaging was employed to combine predictions from the three trained models. This approach leverages complementary feature representations from different architectures, leading to improved classification robustness and accuracy.
Step 9: Explainable AI (XAI)
To enhance model transparency and interpretability, SHAP and LIME were applied:
•	SHAP was used to generate pixel-level explanations using a blurring-based image masker.
•	LIME was employed to highlight locally important regions influencing model predictions.
These techniques provided visual explanations that support clinical interpretability of the classification results.

Step 10: Reproducibility
All scripts were modularized, and trained model files were saved in. keras format. The complete workflow—from preprocessing to explainability—can be reproduced using the provided code and dataset paths.
8. Citations (if applicable): If this dataset was used in research, provide references.
If you use this code or dataset in your research, please cite the following resources:
1.	Kvasir-V2 Dataset
Pogorelov, K., Randel, K. R., Griwodz, C., et al.
Kvasir: A Multi-Class Image Dataset for Computer-Aided Gastrointestinal Disease Detection.
Proceedings of the 8th ACM Multimedia Systems Conference (MMSys), 2017.
Dataset available at:
https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset
2.	TensorFlow / Keras
Abadi, M., et al.
TensorFlow: Large-scale machine learning on heterogeneous systems.
Software available from tensorflow.org
3.	SHAP
Lundberg, S. M., & Lee, S.-I.
A Unified Approach to Interpreting Model Predictions.
Advances in Neural Information Processing Systems (NeurIPS), 2017.
4.	LIME
Ribeiro, M. T., Singh, S., & Guestrin, C.
“Why Should I Trust You?” Explaining the Predictions of Any Classifier.
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
9. License & Contribution Guidelines:
License
This project is provided for academic and research purposes only. The code is shared to support the reproducibility of the experimental results reported in the associated research article. Users are free to use, modify, and extend the code for non-commercial research and educational purposes, provided that proper credit is given to the original authors.
The dataset used in this project (Kvasir-V2) is subject to its original license and usage terms as specified by the dataset providers. Users are advised to review and comply with the dataset’s licensing conditions before use.
10. Contribution Guidelines
Contributions to this project are welcome for research and educational improvements. If you wish to contribute:
•	Fork the repository and create a new branch for your changes.
•	Clearly describe the purpose of the modification or enhancement.
•	Ensure that any added code follows the existing project structure and coding style.
•	Provide appropriate documentation and comments where necessary.
•	Submit a pull request with a clear explanation of the changes.
By contributing to this repository, you agree that your contributions may be used for research and educational purposes with proper attribution.









 


