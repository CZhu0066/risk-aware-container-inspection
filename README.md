# risk-aware-container-inspection


This repository provides the core implementation of the container inspection decision framework proposed in:

**Risk-Aware Container Inspection via Uncertainty-Weighted Preference Learning and Counterfactual Regularization**

The repository contains three main parts:

1. Dataset and example data
2. State indicator extraction
3. Inspection decision framework

---

## 1. Dataset

### 1.1 Public Dataset

The public container defect images used in this study were collected and reorganized from multiple publicly available datasets on **Roboflow Universe**.

The selected images cover six defect categories:

- Fracture
- Dent
- Rust
- Deformation
- Scratch
- Hole

Due to the large size of the complete dataset, only representative examples are currently included in this repository. Additional dataset information and source links will be progressively updated.

> **Dataset Notice:**  
> The original images obtained from Roboflow Universe remain subject to the licenses and terms specified by their respective dataset owners. This repository does not claim ownership of these third-party images. Users should refer to the original dataset pages and comply with the corresponding licenses before downloading, redistributing, or using the data.

Example data are provided in:

```text
Examples of small datasets.zip

