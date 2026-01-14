# Wavelength Selection Pipeline - Documentation Index

**Complete documentation package for hyperspectral wavelength selection using autoencoder-based perturbation analysis**

---

## 📚 Documentation Overview

This documentation package provides comprehensive coverage of the wavelength selection pipeline, from high-level concepts to detailed mathematical formulas. Choose the document that best fits your current needs:

---

## 🎯 Quick Navigation

### For First-Time Users:
**Start here:** [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) - Sections 1-2
- Read Executive Summary (Section 1)
- Review Pipeline Architecture (Section 2)

### For Implementation:
**Use:** [CONFIGURATION_QUICK_REFERENCE.md](CONFIGURATION_QUICK_REFERENCE.md)
- Quick configuration templates
- Pre-configured profiles
- Decision trees

### For Understanding the Math:
**Reference:** [MATHEMATICAL_FORMULAS_REFERENCE.md](MATHEMATICAL_FORMULAS_REFERENCE.md)
- All mathematical equations
- Detailed derivations
- Formula examples

### For Deep Dive:
**Read:** [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) - All sections
- Complete technical details
- Step-by-step explanations
- Theoretical foundations

---

## 📖 Document Descriptions

### 1. PIPELINE_DOCUMENTATION.md
**Main comprehensive technical documentation**

**Total Length:** ~150 pages equivalent
**Best For:** Understanding the complete system

**Contents:**
- Section 1: Executive Summary
  - Purpose and methodology
  - Key innovations
  - Typical results

- Section 2: Pipeline Architecture
  - High-level overview with diagrams
  - 7-phase pipeline flow
  - System architecture

- Section 3: Mathematical Foundations
  - Autoencoder architecture
  - Dimension selection methods (variance, activation, PCA)
  - Perturbation methods (percentile, std, range)
  - Influence measurement

- Section 4: Configuration Parameters - Complete Reference
  - All 11 parameters explained
  - Options, interpretations, recommendations
  - Experimental results for each option
  - When to use each setting

- Section 5: Perturbation & Influence Calculation
  - Complete algorithm with pseudocode
  - Step-by-step example execution
  - Why the method works

- Section 6: Diversity Constraint Methods
  - Maximum Marginal Relevance (MMR)
  - Minimum Distance method
  - Algorithm details and examples

- Section 7: Normalization Methods
  - Variance normalization
  - Max per excitation
  - When to use each

- Section 8: Complete Pipeline Flow
  - End-to-end execution trace
  - Data flow at each step
  - Input/output specifications

- Section 9: Clustering & Validation
  - KNN-based clustering
  - Evaluation metrics (Purity, ARI, NMI, Silhouette)

- Section 10: Performance Metrics
  - Data reduction
  - Quality improvement
  - Speed improvement

- Appendices:
  - Configuration examples
  - Troubleshooting guide

**When to Read:**
- First time learning the system
- Need detailed understanding
- Writing papers or documentation
- Troubleshooting issues

---

### 2. CONFIGURATION_QUICK_REFERENCE.md
**Practical guide for configuration setup**

**Total Length:** ~20 pages equivalent
**Best For:** Quick implementation and parameter tuning

**Contents:**
- Quick Configuration Template
  - Copy-paste ready configuration

- Parameter Reference Table
  - All parameters in one table
  - Best values highlighted
  - Impact ratings

- Quick Decision Trees
  - Choose dimension selection method
  - Choose number of bands
  - Choose perturbation settings
  - Choose diversity settings

- Pre-configured Profiles
  1. Best Overall (recommended)
  2. Maximum Efficiency
  3. Conservative
  4. Influence-Focused
  5. Diversity-Focused
  6. Experimental PCA

- Parameter Tuning Guide
  - If purity is too low
  - If execution is too slow
  - If wavelengths are too clustered
  - If wavelengths miss important features

- Common Mistakes to Avoid
  - What NOT to do

- Validation Checklist
  - How to verify results

- Advanced Tuning
  - Fine-tuning lambda_diversity
  - Fine-tuning n_important_dimensions
  - Fine-tuning perturbation_magnitudes

- Example Usage
  - Complete code example

- Results Interpretation
  - What constitutes good results

**When to Read:**
- Setting up a new experiment
- Tuning parameters
- Need quick answers
- Copying configuration templates

---

### 3. MATHEMATICAL_FORMULAS_REFERENCE.md
**Complete mathematical reference**

**Total Length:** ~30 pages equivalent
**Best For:** Understanding equations and mathematical details

**Contents:**
- 1. Autoencoder Architecture
  - Input/output dimensions
  - Encoder/decoder formulas
  - Loss function

- 2. Dimension Selection Methods
  - Variance formula (expanded)
  - Activation formula
  - PCA formula (SVD)

- 3. Perturbation Methods
  - Percentile method formula
  - Standard deviation formula
  - Absolute range formula
  - Examples with numbers

- 4. Influence Measurement
  - Core influence formula
  - Accumulated influence
  - Gradient approximation

- 5. Normalization Methods
  - Variance normalization
  - Max per excitation
  - Raw (none)

- 6. Maximum Marginal Relevance
  - MMR score formula
  - Relevance calculation
  - Cosine similarity
  - Iterative selection algorithm
  - Lambda interpretation

- 7. Distance-Based Selection
  - Minimum distance constraint
  - Selection algorithm

- 8. Clustering Metrics
  - Purity (detailed)
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)
  - Silhouette score

- 9. Performance Metrics
  - Data reduction
  - Compression ratio
  - Speed improvement
  - Quality improvement
  - Efficiency score

- 10. Statistical Tests
  - Significance testing
  - Confidence intervals

- Summary of Key Equations
  - Top 5 most important formulas

**When to Read:**
- Need specific formula
- Implementing algorithms
- Writing mathematical sections of papers
- Verifying calculations
- Understanding theoretical foundations

---

## 🔍 Finding Specific Information

### Configuration Parameters

| What You Need | Where to Find It |
|---------------|------------------|
| Parameter list & best values | CONFIGURATION_QUICK_REFERENCE.md - Parameter Reference Table |
| Detailed parameter explanation | PIPELINE_DOCUMENTATION.md - Section 4 |
| Mathematical formula for method | MATHEMATICAL_FORMULAS_REFERENCE.md - Sections 2-3 |

### Algorithms & Methods

| Algorithm | Where to Find It |
|-----------|------------------|
| Dimension selection (variance/activation/PCA) | PIPELINE_DOCUMENTATION.md - Section 3.2 |
| Perturbation methods | PIPELINE_DOCUMENTATION.md - Section 3.3 |
| Influence calculation | PIPELINE_DOCUMENTATION.md - Section 3.4 |
| MMR diversity | PIPELINE_DOCUMENTATION.md - Section 6.1 |
| Complete pipeline flow | PIPELINE_DOCUMENTATION.md - Section 8 |

### Mathematical Formulas

| Formula | Where to Find It |
|---------|------------------|
| Variance calculation | MATHEMATICAL_FORMULAS_REFERENCE.md - Section 2.1 |
| Standard deviation perturbation | MATHEMATICAL_FORMULAS_REFERENCE.md - Section 3.2 |
| Influence measurement | MATHEMATICAL_FORMULAS_REFERENCE.md - Section 4 |
| MMR score | MATHEMATICAL_FORMULAS_REFERENCE.md - Section 6 |
| Purity metric | MATHEMATICAL_FORMULAS_REFERENCE.md - Section 8.1 |

### Practical Implementation

| Task | Where to Find It |
|------|------------------|
| Quick setup template | CONFIGURATION_QUICK_REFERENCE.md - Quick Configuration Template |
| Pre-made profiles | CONFIGURATION_QUICK_REFERENCE.md - Pre-configured Profiles |
| Decision making | CONFIGURATION_QUICK_REFERENCE.md - Quick Decision Trees |
| Troubleshooting | PIPELINE_DOCUMENTATION.md - Appendix B |
| Code example | CONFIGURATION_QUICK_REFERENCE.md - Example Usage |

---

## 📊 Key Results & Benchmarks

### Best Configuration
**From experimental validation:**

```python
{
    'name': 'mmr_lambda050_variance',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [15, 30, 45],
    'n_important_dimensions': 7,
    'n_bands_to_select': 10,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.5
}
```

**Performance:**
- Purity: **0.8682** (baseline: 0.8543)
- Data Reduction: **91.0%** (111 → 10 bands)
- Speed Improvement: **2.24×** faster
- Quality Improvement: **+1.6%** over full data

**Reference:** See PIPELINE_DOCUMENTATION.md - Section 1.4 and Appendix A.1

---

## 🎓 Learning Path

### For Beginners:
1. Read PIPELINE_DOCUMENTATION.md - Section 1 (Executive Summary)
2. Review CONFIGURATION_QUICK_REFERENCE.md - Quick Configuration Template
3. Try Pre-configured Profile 1 (Best Overall)
4. Review results and interpret

### For Intermediate Users:
1. Read PIPELINE_DOCUMENTATION.md - Sections 1-4
2. Understand configuration parameters in detail
3. Experiment with different profiles
4. Use CONFIGURATION_QUICK_REFERENCE.md for tuning

### For Advanced Users:
1. Read complete PIPELINE_DOCUMENTATION.md
2. Study MATHEMATICAL_FORMULAS_REFERENCE.md
3. Implement custom modifications
4. Optimize for specific use cases

### For Paper Writing:
1. Read PIPELINE_DOCUMENTATION.md - Sections 3 & 5 (math details)
2. Reference MATHEMATICAL_FORMULAS_REFERENCE.md for equations
3. Use performance metrics from Section 10
4. Cite experimental results from Appendix A

---

## 🛠️ Common Use Cases

### Use Case 1: Quick Start
**Goal:** Get wavelength selection working ASAP

**Steps:**
1. Open CONFIGURATION_QUICK_REFERENCE.md
2. Copy "Best Overall" profile
3. Adjust paths in example code
4. Run and validate

**Expected Time:** 15 minutes

---

### Use Case 2: Parameter Optimization
**Goal:** Find best parameters for your specific data

**Steps:**
1. Start with "Best Overall" profile
2. Use Decision Trees in CONFIGURATION_QUICK_REFERENCE.md
3. Follow Parameter Tuning Guide
4. Iterate based on results

**Expected Time:** 2-3 hours

---

### Use Case 3: Understanding the Method
**Goal:** Deeply understand how the pipeline works

**Steps:**
1. Read PIPELINE_DOCUMENTATION.md - Sections 1-3
2. Study example in Section 5.2
3. Review mathematical foundations in MATHEMATICAL_FORMULAS_REFERENCE.md
4. Trace through complete flow in Section 8

**Expected Time:** 4-6 hours

---

### Use Case 4: Writing Research Paper
**Goal:** Document method for publication

**Steps:**
1. Read PIPELINE_DOCUMENTATION.md - Sections 3-5 for Methods section
2. Use MATHEMATICAL_FORMULAS_REFERENCE.md for equations
3. Reference experimental results from Appendix A
4. Include performance metrics from Section 10

**Expected Time:** Full day

---

## 📝 Quick Reference

### Top 5 Parameters to Know

1. **dimension_selection_method: 'variance'**
   - Most important parameter
   - Use 'variance' for best results

2. **n_bands_to_select: 7-11**
   - Controls data reduction vs quality
   - 10 is optimal for most cases

3. **use_diversity_constraint: True**
   - Always enable for best results
   - Prevents redundant wavelengths

4. **diversity_method: 'mmr'**
   - Best diversity method
   - Use with lambda_diversity = 0.5

5. **normalization_method: 'max_per_excitation'**
   - Best normalization for multi-excitation data
   - Ensures fair comparison

**Full details:** CONFIGURATION_QUICK_REFERENCE.md - Parameter Reference Table

---

### Key Equations (Most Used)

1. **Variance Selection**
   ```
   I_var(d) = (1/N) Σᵢ (zᵢ[d] - μ_d)²
   ```

2. **Standard Deviation Perturbation**
   ```
   δ = s × (m/100) × σ_d
   ```

3. **MMR Score**
   ```
   MMR = λ·Relevance - (1-λ)·MaxSimilarity
   ```

4. **Purity**
   ```
   Purity = (1/N) Σᵢ max_j |Cᵢ ∩ Lⱼ|
   ```

**Full formulas:** MATHEMATICAL_FORMULAS_REFERENCE.md

---

## 🔗 Cross-References

### Pipeline Overview
- **Architecture:** PIPELINE_DOCUMENTATION.md - Section 2
- **Mathematical basis:** PIPELINE_DOCUMENTATION.md - Section 3
- **Complete flow:** PIPELINE_DOCUMENTATION.md - Section 8

### Configuration
- **Parameter reference:** PIPELINE_DOCUMENTATION.md - Section 4
- **Quick setup:** CONFIGURATION_QUICK_REFERENCE.md
- **Tuning guide:** CONFIGURATION_QUICK_REFERENCE.md - Parameter Tuning Guide

### Mathematics
- **All formulas:** MATHEMATICAL_FORMULAS_REFERENCE.md
- **Method explanations:** PIPELINE_DOCUMENTATION.md - Section 3
- **Algorithm details:** PIPELINE_DOCUMENTATION.md - Sections 5-6

### Performance
- **Benchmarks:** PIPELINE_DOCUMENTATION.md - Section 10
- **Experimental results:** PIPELINE_DOCUMENTATION.md - Appendix A
- **Metrics definitions:** MATHEMATICAL_FORMULAS_REFERENCE.md - Section 8

---

## 🚀 Getting Started Checklist

- [ ] Read Executive Summary (PIPELINE_DOCUMENTATION.md - Section 1)
- [ ] Review pipeline architecture diagram (PIPELINE_DOCUMENTATION.md - Section 2)
- [ ] Copy configuration template (CONFIGURATION_QUICK_REFERENCE.md)
- [ ] Adjust paths for your data
- [ ] Run first experiment
- [ ] Validate results using checklist (CONFIGURATION_QUICK_REFERENCE.md)
- [ ] Tune parameters if needed
- [ ] Document final configuration

---

## 📞 Troubleshooting Quick Links

| Problem | Solution Location |
|---------|------------------|
| Low purity (<0.85) | PIPELINE_DOCUMENTATION.md - Appendix B.1 |
| Slow execution | PIPELINE_DOCUMENTATION.md - Appendix B.2 |
| Poor wavelength diversity | PIPELINE_DOCUMENTATION.md - Appendix B.3 |
| Parameter tuning needed | CONFIGURATION_QUICK_REFERENCE.md - Parameter Tuning Guide |
| Understanding errors | PIPELINE_DOCUMENTATION.md - Appendix B |

---

## 📈 Performance Expectations

Based on experimental validation:

### Typical Results:
- **Data Reduction:** 90-93%
- **Clustering Purity:** 0.86-0.87
- **Speed Improvement:** 2-3×
- **Quality Change:** -1% to +2% (maintains or improves)

### Best Case:
- **Data Reduction:** 91%
- **Clustering Purity:** 0.868
- **Speed Improvement:** 2.24×
- **Quality Improvement:** +1.6%

### Efficiency Case:
- **Data Reduction:** 93.7%
- **Clustering Purity:** 0.860
- **Speed Improvement:** 3.1×
- **Quality Change:** +0.7%

**Full details:** PIPELINE_DOCUMENTATION.md - Section 10

---

## 🎯 Key Takeaways

1. **Variance dimension selection performs best** (0.868 purity vs 0.784 for PCA)
2. **MMR diversity is essential** for avoiding redundant wavelengths
3. **7-11 wavelengths is optimal** for balance of quality and efficiency
4. **Standard deviation perturbation** works best with magnitudes [15,30,45]
5. **Max per excitation normalization** ensures fair comparison across excitations

**Source:** Experimental results across 10 configurations

---

## 📚 Citation Information

If using this pipeline for research, please reference:

**Pipeline Documentation:**
- Complete methodology: PIPELINE_DOCUMENTATION.md
- Mathematical formulations: MATHEMATICAL_FORMULAS_REFERENCE.md
- Configuration details: CONFIGURATION_QUICK_REFERENCE.md

**Key Methods:**
- Autoencoder-based wavelength selection
- Perturbation-based influence measurement
- Maximum Marginal Relevance (MMR) for diversity
- KNN-based validation

---

## 🔄 Document Versions

**Current Version:** 1.0
**Last Updated:** October 2025

**Included Documents:**
1. PIPELINE_DOCUMENTATION.md (v1.0)
2. CONFIGURATION_QUICK_REFERENCE.md (v1.0)
3. MATHEMATICAL_FORMULAS_REFERENCE.md (v1.0)
4. DOCUMENTATION_INDEX.md (v1.0) - this file

---

## 📞 Support & Questions

For questions about:

**Implementation:** → CONFIGURATION_QUICK_REFERENCE.md
**Theory:** → PIPELINE_DOCUMENTATION.md
**Math:** → MATHEMATICAL_FORMULAS_REFERENCE.md
**Navigation:** → DOCUMENTATION_INDEX.md (this file)

**Still need help?**
- Check Troubleshooting (PIPELINE_DOCUMENTATION.md - Appendix B)
- Review Common Mistakes (CONFIGURATION_QUICK_REFERENCE.md)
- Trace complete flow (PIPELINE_DOCUMENTATION.md - Section 8)

---

**End of Index**

**Next Steps:** Choose a document from the navigation guide above and start reading!
