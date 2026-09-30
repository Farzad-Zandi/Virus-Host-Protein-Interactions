# Virus-Host-Protein-Interactions
# Optimized Deep Knowledge Distillation for Large-Scale Virus–Host Protein Interactions Prediction
![Graphical Abstract](https://github.com/Farzad-Zandi/Virus-Host-Protein-Interactions/blob/main/Graphical%20Abstract.png)

![Languages](https://img.shields.io/badge/Languages-Python%20%7C%20R-brightgreen.svg)  ![Libraries](https://img.shields.io/badge/Libraries-PyTorch%20%7C%20protr-purple.svg)  ![Repository](https://img.shields.io/badge/Repository-UniProt-orange.svg)  ![Deployment](https://img.shields.io/badge/Deployment-Github-yellow.svg)  ![Debugging](https://img.shields.io/badge/Debugging-LocalHost-blue.svg)

## Abstract
<p align="justify">
Background: In a rapidly evolving viral landscape, understanding virus–host protein interactions (VHPIs) is essential to elucidate viral infection mechanisms and support the development of effective therapeutic strategies. Accurate computational Prediction of VHPIs remains challenging because of the complexity and high dimensionality of protein sequence and functional representations.
   
Methods: We propose a Knowledge Distillation framework for pairwise VHPI prediction that integrates three teacher networks with a hybrid student architecture. Protein characteristics were represented using Conjoint Triad  (CT), Dipeptide Composition (DC), Dipeptide Deviation from Expected Mean (DDE), Pseudo-Amino Acid Composition (PseAAC), amino acid encoding, and Gene Ontology (GO) features. We used tuned Monte Carlo ISUD for feature selection and compared it with five alternative approaches: PCA, MCFS, NMF, FA, and Optuna. We used an autoencoder to compress the one-hot encoded GO features. We evaluated on 25 randomly balanced datasets and one additional balanced dataset generated using NearMiss. Five-fold cross-validation was performed at the interaction-pair level using an 80:20 training-to-test ratio in each fold. The final configuration consisted of CT features, NearMiss-based data balancing, tuned MC-ISUD feature selection, and the Knowledge Distillation teacher–student framework.

Results: The proposed framework achieved 92.91% accuracy and 96.16% AUC-ROC on the evaluated virus–host protein interaction datasets. Comparative experiments demonstrated that the proposed approach outperformed the existing methods considered in this study. These results are specific to the evaluated datasets and pairwise cross-validation design and should not be interpreted as evidence of generalization to completely unseen viral or host proteins.

Conclusions: The findings indicate that combining CT-based representation, NearMiss-based data balancing, tuned ISUD feature selection, and Knowledge Distillation provides a computational framework for pairwise VHPI prediction. The predictions should be regarded as computational hypotheses requiring independent experimental validation.
</p>

## Keyword
Virus-Host Protein Interaction, Deep Learning, Feature Extraction, Feature Selection, Data Balancing.
## Authors
Farzad Zandi, Parvaneh Mansouri.
## DOI and Links
- DOI: [https://](https://)
- Article: [https://](https://)
## Description
The raw datasets are available in [https://github.com/Farzad-Zandi/Virus-Host-Protein-Interactions/releases/tag/Raw_Datasets](https://github.com/Farzad-Zandi/Virus-Host-Protein-Interactions/releases/tag/Raw_Datasets)

All codes are available in the Code folder.

The results for all datasets evaluated using the three feature extraction methods are provided in the Results folder in the file “Results.pdf.”

## Usage
To run the model, follow the steps below:

1. Download the raw files.
2. Run Feature Extraction.R to extract the features. The script includes four feature extraction methods: CT, DC, DDE, and PseAAC. after extraction save the four     extracted files.
3. To extract the GO annotations, run GO_Extraction.R and save the output file.
4. Run GO_to_ONE_HOT.py to generate one-hot encoded data from GO annotation.
5. Run Auto_Encoder.py to make the encoded GO representation.
6. Run Balancing_technique.py to balance the datasets.
7. Run Feature_Selection_Methods.py for feature selection.
8. Run the Model.py.
## Citiation
```bibtex
@article {
}
```
## Contact
For further inquiries, please contact us:
- Farzad Zandi.
- Website: [Zandigroup](https://Zandigroup.ir)
- LinkedIn: [Farzad Zandi](https://www.linkedin.com/in/farzad-zandi-86a37326a/)
- Email: [info@zandigroup.ir](info@zandigroup.ir)
- Email: [zandi8farzad@gmail.com](zandi8farzad@gmail.com)
- Email: [zandi_farzad@yahoo.com](zandi_farzad@yahoo.com)

