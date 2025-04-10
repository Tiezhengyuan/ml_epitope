
# Epitope Prediction

## introduction
Identification of epitopes or binding segments of a protein with antibodies is critical in the development of antibody drugs or vaccin design.

### Epitopes could be categorized into linear, conformational, or discontinuous types. Various features can be used for epitope prediction:
- compositions of amino acids
- physical-chemical amino acids
- frequency of amino acids
- AA sequence
- 3D structure data from antibody-antigen complex

### Dataset used for prediction:
- 3D structure data from antibody-antigen complex
- epitope and antigen sequences from IEDB
- protein annotations from Uniprot

### ML/DL/LLMs for epitope prediction:
- Parameter estimateion: calculate statistics and do significance testing
- Random Forest: evaluate feature importance
- ANN: employ physical-chemical properties and frequency of amino acids of epitope sequences to predict linear epitope
- RNN: Employ amino acid sequence only to predict epitopes.
- Supervised fine-tune (SFT) Llama-3-8b-Instruct: input text is defined as 
    "<sequence>" or
    "<AA in sequence>" or
    "<AA in sequence> | <physical chemicals>  <frequenct>"

## study
### study I
How have composition of amino acids impact on epitope prediction ?
 - linear epiptopes from GenPept, IEDB, UniProt, and ChEBI
 - features: physical chemical properties including hydrophobicity, polarity, polarizability, Van der Waals volume, and frequency of 20 amino acids
 - models: artificial neural network, two fully connected layers 
 - results: accuracy on test data is >99 %.
