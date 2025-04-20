## BioThreat Detection System

### Key Features
- Early warning system for biological threats
- Integrates epidemiological, genomic, and mobility data
- Threshold-optimized neural network model

### Methodology
1. **Data Processing**: Temporal alignment of multi-source data
2. **Class Balancing**: Adaptive weighting of minority class
3. **Model Architecture**: 
   - Input Layer: 401 features
   - Hidden Layers: 64 → 32 neurons with dropout
   - Output: Sigmoid-activated threat probability

### Results
![Evaluation Report](results/evaluation_report.png)