# DLMI-Brain-cancer-seg
## Dataset 
- Dataset:  from BraTS 2016 and 2017 ([download here](http://medicaldecathlon.com/index.html.))
- Patient: 750 multi-parametric magnetic resonance imaging (MRI) scans from patients diagnosed with either glioblastoma or lower-grade glioma were included
- Multi-modality: The multi-para-metric MRI sequences of each patient included native (T1) and post-Gadolinium contrast T1-weighted (T1-Gd), native T2-weighted (T2), and T2 Fluid- Attenuated Inversion Recovery (T2-FLAIR) volumes.

## Task 
Brain gliomas automated segmentation  

## To do list 
- Implement a simple segmentation model (U-NET)
- Experiment 1: Add fusion layer 
    - in the input 
    - in the middle of the encoder phase
    - at the last layer to ensemble 
- Experiment 2: Separate branch for different modality 
- Report + code (March 18th)
- Presentation (March 20th)

