# Mathematical Formulas Reference

**Complete mathematical reference for wavelength selection pipeline**

---

## 1. Autoencoder Architecture

### Input/Output Dimensions

```
Input hyperspectral data:
  X^(λₑₓ) ∈ ℝ^(H × W × Nₑₘ)

  Where:
    H, W = spatial dimensions
    Nₑₘ = number of emission bands for excitation λₑₓ
```

### Encoder

```
Encoder: E : ℝ^(H × W × Nₑₘ) → ℝ^(C × L × Hₗ × Wₗ)

z = E(X)

  Where:
    C = number of latent channels
    L = latent depth
    Hₗ, Wₗ = latent spatial dimensions
```

### Decoder

```
Decoder: D : ℝ^(C × L × Hₗ × Wₗ) → ℝ^(H × W × Nₑₘ)

X̂ = D(z)
```

### Loss Function

```
ℒ(θ) = ||X - X̂||²₂ + α·KL(q(z)||p(z)) + β·||z||₁

  Where:
    ||X - X̂||²₂ = reconstruction loss (MSE)
    KL(q(z)||p(z)) = KL divergence regularization
    ||z||₁ = L1 sparsity constraint
    α, β = hyperparameters
```

---

## 2. Dimension Selection Methods

### Method 1: Variance

**Formula:**

```
For latent representation Z = [z₁, z₂, ..., zₙ] where zᵢ ∈ ℝ^D

Flatten: z_flat[i] ∈ ℝ^D  (D = C × L × Hₗ × Wₗ)

For dimension d ∈ {1, ..., D}:

  μ_d = (1/N) Σᵢ₌₁ᴺ zᵢ[d]

  I_var(d) = (1/N) Σᵢ₌₁ᴺ (zᵢ[d] - μ_d)²

Rank dimensions by I_var(d) in descending order
Select top K dimensions
```

**Expanded form:**

```
Variance = E[(X - E[X])²]
         = E[X²] - (E[X])²

For dimension d:
  I_var(d) = (1/N) Σᵢ₌₁ᴺ zᵢ[d]² - ((1/N) Σᵢ₌₁ᴺ zᵢ[d])²
```

### Method 2: Activation

**Formula:**

```
I_act(d) = (1/N) Σᵢ₌₁ᴺ |zᵢ[d]|

Rank dimensions by I_act(d) in descending order
Select top K dimensions
```

**Interpretation:**
Mean absolute value of latent activations

### Method 3: PCA

**Formula:**

```
1. Standardize latent representations:
   Z_scaled[i,d] = (Z[i,d] - μ_d) / σ_d

2. Singular Value Decomposition:
   Z_scaled = U Σ Vᵀ

   Where:
     U ∈ ℝ^(N × K) = left singular vectors (samples)
     Σ ∈ ℝ^(K × K) = singular values (diagonal)
     V ∈ ℝ^(D × K) = right singular vectors (features)
     K = min(N, D) or specified n_components

3. Principal components:
   PC_j = V[:,j] for j = 1, ..., K

4. Dimension importance:
   I_pca(d) = Σⱼ₌₁ᴷ |PC_j[d]|

Rank dimensions by I_pca(d) in descending order
Select top K dimensions
```

**Variance explained:**

```
Total variance explained by PC_j:
  var_explained_j = λⱼ / (Σₖ λₖ)

  Where λⱼ = Σ[j,j]² (squared singular value)
```

---

## 3. Perturbation Methods

### Method 1: Percentile

**Formula:**

```
Given:
  - Dimension d with distribution Z[d] = {z₁[d], z₂[d], ..., zₙ[d]}
  - Magnitude m ∈ [0, 100]
  - Sign s ∈ {-1, +1}

1. Target percentile:
   p_target = 50 + s × (m / 2)

2. Empirical quantile:
   Q_p(Z[d]) = value such that p% of observations ≤ Q_p

3. Perturbation amount:
   δ_percentile(d, m, s) = Q_p_target(Z[d]) - z_d

4. Perturbed value:
   z'_d = z_d + δ_percentile(d, m, s)
```

**Example:**

```
m = 20, s = +1:
  p_target = 50 + 1×(20/2) = 60th percentile
  δ = Q₀.₆₀(Z[d]) - z_d

m = 30, s = -1:
  p_target = 50 - 1×(30/2) = 35th percentile
  δ = Q₀.₃₅(Z[d]) - z_d
```

### Method 2: Standard Deviation

**Formula:**

```
Given:
  - Dimension d with mean μ_d and std σ_d
  - Magnitude m ∈ [0, 100]
  - Sign s ∈ {-1, +1}

1. Standard deviation:
   σ_d = √((1/N) Σᵢ₌₁ᴺ (zᵢ[d] - μ_d)²)

2. Perturbation amount:
   δ_std(d, m, s) = s × (m / 100) × σ_d

3. Perturbed value:
   z'_d = z_d + δ_std(d, m, s)
```

**Example:**

```
σ_d = 2.5, m = 30, s = +1:
  δ = +1 × (30/100) × 2.5 = +0.75
  z'_d = z_d + 0.75

σ_d = 2.5, m = 45, s = -1:
  δ = -1 × (45/100) × 2.5 = -1.125
  z'_d = z_d - 1.125
```

**Interpretation:**

```
m/100 represents fraction of standard deviation:
  m = 15 → 0.15σ perturbation
  m = 30 → 0.30σ perturbation
  m = 45 → 0.45σ perturbation
```

### Method 3: Absolute Range

**Formula:**

```
Given:
  - Dimension d with range R_d = max(Z[d]) - min(Z[d])
  - Magnitude m ∈ [0, 100]
  - Sign s ∈ {-1, +1}

1. Range:
   R_d = max_{i} zᵢ[d] - min_{i} zᵢ[d]

2. Perturbation amount:
   δ_range(d, m, s) = s × (m / 100) × R_d

3. Perturbed value:
   z'_d = z_d + δ_range(d, m, s)
```

**Example:**

```
R_d = 5.0, m = 40, s = +1:
  δ = +1 × (40/100) × 5.0 = +2.0
  z'_d = z_d + 2.0

R_d = 5.0, m = 60, s = -1:
  δ = -1 × (60/100) × 5.0 = -3.0
  z'_d = z_d - 3.0
```

**Interpretation:**

```
m/100 represents fraction of full range:
  m = 20 → move 20% of range
  m = 40 → move 40% of range
  m = 60 → move 60% of range
```

---

## 4. Influence Measurement

### Core Influence Formula

**Formula:**

```
For dimension d, perturbation δ:

1. Perturb latent:
   z' = z + δ · e_d

   Where e_d = unit vector in direction d

2. Decode:
   X̂ = D(z)      (baseline reconstruction)
   X̂' = D(z')    (perturbed reconstruction)

3. Difference:
   Δ(ex, em) = X̂'(ex,em) - X̂(ex,em)

4. Influence on band (ex, em):
   I(ex, em | d, δ) = (1/N) Σᵢ₌₁ᴺ (1/(H·W)) Σₓ,ᵧ |Δᵢ(ex,em)[x,y]|
```

**Expanded:**

```
I(ex, em | d, δ) = (1/(N·H·W)) Σᵢ₌₁ᴺ Σₓ₌₁ᴴ Σᵧ₌₁ᵂ |X̂'ᵢ(ex,em)[x,y] - X̂ᵢ(ex,em)[x,y]|

Where:
  N = number of patches
  H, W = spatial dimensions
  (ex, em) = excitation-emission pair
```

### Accumulated Influence

**Formula:**

```
Total influence across all perturbations:

I_total(ex, em) = Σ_(d∈D) Σ_(m∈M) Σ_(s∈S) w(m,s) · I(ex,em | d, δ(d,m,s)) · I_dim(d)

Where:
  D = set of K important dimensions
  M = set of perturbation magnitudes
  S = set of directions {-1, +1}
  w(m,s) = weight (typically 0.5 for bidirectional, 1.0 for unidirectional)
  I_dim(d) = importance score of dimension d
  δ(d,m,s) = perturbation amount
```

**Weight calculation:**

```
For bidirectional perturbation:
  w(m, +1) = 0.5
  w(m, -1) = 0.5
  (Sum to 1.0 per magnitude)

For unidirectional:
  w(m, +1) = 1.0
  or
  w(m, -1) = 1.0
```

### Gradient Approximation

**Interpretation:**

```
Influence approximates gradient magnitude:

I(ex,em | d) ≈ |∂X̂(ex,em) / ∂z_d|

Finite difference approximation:
  ∂X̂/∂z_d ≈ (X̂(z + δe_d) - X̂(z)) / δ

Therefore:
  I(ex,em | d) ≈ |X̂(z + δe_d) - X̂(z)| / δ

Averaging over multiple magnitudes and directions provides robust estimate.
```

---

## 5. Normalization Methods

### Method 1: Variance Normalization

**Formula:**

```
For each band (ex, em):

1. Compute spatial variance:
   Var(ex, em) = (1/(H·W)) Σₓ,ᵧ (X(ex,em)[x,y] - μ_(ex,em))²

   Where:
     μ_(ex,em) = (1/(H·W)) Σₓ,ᵧ X(ex,em)[x,y]

2. Normalize influence:
   I'(ex, em) = I(ex, em) / Var(ex, em)
```

**Interpretation:**

```
Adjusted influence score:
  I'(ex,em) = signal / baseline_noise

Bands with:
  - High I, low Var → High I' (signal-rich)
  - High I, high Var → Low I' (noise-rich)
  - Low I, low Var → Medium I' (weak signal)
```

### Method 2: Max Per Excitation

**Formula:**

```
For each excitation ex:

1. Find maximum influence:
   I_max(ex) = max_{em} I(ex, em)

2. Normalize all emission bands:
   I'(ex, em) = I(ex, em) / I_max(ex)

Result: I'(ex, em) ∈ [0, 1] for all em
```

**Purpose:**

```
Ensures fair comparison across excitations by scaling each to [0,1]:

Before normalization:
  Ex365: I ∈ [0.5, 2.5]
  Ex385: I ∈ [0.1, 0.8]  ← Would be underrepresented

After normalization:
  Ex365: I' ∈ [0.2, 1.0]
  Ex385: I' ∈ [0.125, 1.0]  ← Fair representation
```

### Method 3: None (Raw)

**Formula:**

```
I'(ex, em) = I(ex, em)

No transformation applied.
```

---

## 6. Maximum Marginal Relevance (MMR)

### MMR Score

**Formula:**

```
For candidate wavelength wᵢ and selected set S:

MMR(wᵢ) = λ · Relevance(wᵢ) - (1-λ) · MaxSimilarity(wᵢ, S)

Where:
  λ ∈ [0, 1] = diversity parameter

  Relevance(wᵢ) = I(wᵢ) / max_{w} I(w)

  MaxSimilarity(wᵢ, S) = max_{wⱼ ∈ S} Similarity(wᵢ, wⱼ)
```

### Relevance

**Formula:**

```
Relevance(wᵢ) = I(wᵢ) / I_max

Where:
  I(wᵢ) = influence score of wavelength wᵢ
  I_max = max_{w} I(w) = maximum influence across all wavelengths

Result: Relevance ∈ [0, 1]
```

### Similarity (Cosine)

**Formula:**

```
For wavelengths wᵢ and wⱼ:

1. Extract spectral profiles:
   p(wᵢ) = X(ex,em)[flatten] ∈ ℝ^(H×W)

2. Normalize:
   p̂(wᵢ) = p(wᵢ) / ||p(wᵢ)||₂

   Where: ||p||₂ = √(Σₖ pₖ²)

3. Cosine similarity:
   Similarity(wᵢ, wⱼ) = p̂(wᵢ) · p̂(wⱼ)
                       = (Σₖ p̂ᵢ[k] · p̂ⱼ[k])
```

**Expanded cosine similarity:**

```
cos(θ) = (p(wᵢ) · p(wⱼ)) / (||p(wᵢ)||₂ · ||p(wⱼ)||₂)

       = (Σₖ pᵢ[k]·pⱼ[k]) / (√(Σₖ pᵢ[k]²) · √(Σₖ pⱼ[k]²))

Range: [-1, +1]
  +1 = identical patterns
   0 = orthogonal (uncorrelated)
  -1 = opposite patterns
```

### MMR Selection Algorithm

**Iterative Formula:**

```
Initialize:
  S₀ = {w_top}  where w_top = argmax_{w} I(w)
  C = {all wavelengths} \ S₀

For t = 1 to n_bands - 1:

  w* = argmax_{wᵢ ∈ C} [λ · Relevance(wᵢ) - (1-λ) · max_{wⱼ ∈ Sₜ₋₁} Similarity(wᵢ, wⱼ)]

  Sₜ = Sₜ₋₁ ∪ {w*}
  C = C \ {w*}

Return: Sₙ_bands
```

### Lambda Interpretation

**Effect of λ:**

```
λ = 0:
  MMR(wᵢ) = -MaxSimilarity(wᵢ, S)
  → Pure diversity, minimize similarity

λ = 0.5:
  MMR(wᵢ) = 0.5·Relevance(wᵢ) - 0.5·MaxSimilarity(wᵢ, S)
  → Balanced

λ = 1:
  MMR(wᵢ) = Relevance(wᵢ)
  → Pure relevance, ignore diversity
```

**First derivative:**

```
∂MMR/∂λ = Relevance(wᵢ) + MaxSimilarity(wᵢ, S)

Increasing λ:
  - Increases weight on relevance (influence)
  - Decreases weight on diversity
```

---

## 7. Distance-Based Selection

### Minimum Distance Constraint

**Formula:**

```
For candidate wavelength (ex_i, em_i) and selected set S:

Valid(ex_i, em_i) = ∀(ex_j, em_j) ∈ S:
  (ex_i ≠ ex_j) OR (|em_i - em_j| ≥ d_min)

Where:
  d_min = minimum distance in nm
```

**Selection Algorithm:**

```
S = ∅  (empty set)
C = {all wavelengths sorted by influence}

For each wᵢ = (ex_i, em_i) in C:
  is_valid = True

  For each wⱼ = (ex_j, em_j) in S:
    If ex_i == ex_j AND |em_i - em_j| < d_min:
      is_valid = False
      break

  If is_valid:
    S = S ∪ {wᵢ}

  If |S| == n_bands:
    break

Return S
```

---

## 8. Clustering Metrics

### Purity

**Formula:**

```
Purity = (1/N) Σᵢ₌₁ᴷ max_{j} |Cᵢ ∩ Lⱼ|

Where:
  N = total number of samples
  K = number of clusters
  Cᵢ = set of samples in cluster i
  Lⱼ = set of samples in ground truth class j
  |·| = cardinality (size of set)
```

**Interpretation:**

```
For each cluster, count samples in the majority class.
Purity = fraction of correctly assigned samples (if each cluster maps to majority class).

Range: [0, 1]
  1 = perfect clustering
  1/K = random clustering (K classes)
```

### Adjusted Rand Index (ARI)

**Formula:**

```
ARI = (RI - Expected[RI]) / (max(RI) - Expected[RI])

Where Rand Index:
  RI = (TP + TN) / (TP + TN + FP + FN)

  TP = pairs in same cluster and same class
  TN = pairs in different clusters and different classes
  FP = pairs in same cluster but different classes
  FN = pairs in different clusters but same class
```

**Detailed calculation:**

```
Given contingency table n_ij = |Cᵢ ∩ Lⱼ|:

1. Compute:
   a = Σᵢⱼ C(n_ij, 2)  where C(n,k) = n!/(k!(n-k)!)
   b = Σᵢ C(aᵢ, 2)  where aᵢ = Σⱼ n_ij
   c = Σⱼ C(bⱼ, 2)  where bⱼ = Σᵢ n_ij
   d = C(N, 2)

2. ARI:
   ARI = (a - (b·c)/d) / ((b+c)/2 - (b·c)/d)

Range: [-1, 1]
  1 = perfect agreement
  0 = random assignment
  <0 = worse than random
```

### Normalized Mutual Information (NMI)

**Formula:**

```
NMI(C, L) = MI(C, L) / √(H(C) · H(L))

Where:
  MI(C, L) = Σᵢⱼ P(i,j) log(P(i,j) / (P(i)·P(j)))

  H(C) = -Σᵢ P(i) log P(i)
  H(L) = -Σⱼ P(j) log P(j)

  P(i,j) = |Cᵢ ∩ Lⱼ| / N
  P(i) = |Cᵢ| / N
  P(j) = |Lⱼ| / N
```

**Expanded:**

```
Mutual Information:
  MI = H(C) + H(L) - H(C,L)

Where joint entropy:
  H(C,L) = -Σᵢⱼ P(i,j) log P(i,j)

Range: [0, 1]
  1 = perfect correlation
  0 = independent
```

### Silhouette Score

**Formula:**

```
For sample i in cluster Cₖ:

1. Mean intra-cluster distance:
   a(i) = (1/|Cₖ|-1) Σ_{j ∈ Cₖ, j≠i} d(i, j)

2. Mean nearest-cluster distance:
   b(i) = min_{ℓ≠k} (1/|Cℓ|) Σ_{j ∈ Cℓ} d(i, j)

3. Silhouette coefficient:
   s(i) = (b(i) - a(i)) / max(a(i), b(i))

Overall silhouette:
  S = (1/N) Σᵢ₌₁ᴺ s(i)
```

**Interpretation:**

```
s(i) close to 1:
  → Sample well-clustered (far from other clusters)

s(i) close to 0:
  → Sample on boundary between clusters

s(i) close to -1:
  → Sample likely in wrong cluster

Range: [-1, 1]
```

---

## 9. Performance Metrics

### Data Reduction

**Formula:**

```
Reduction (%) = (1 - n_selected / n_total) × 100

Where:
  n_selected = number of selected wavelengths
  n_total = total number of wavelengths
```

**Example:**

```
n_total = 111 bands
n_selected = 10 bands

Reduction = (1 - 10/111) × 100
          = (1 - 0.090) × 100
          = 91.0%
```

### Compression Ratio

**Formula:**

```
Compression Ratio = n_total / n_selected

Example:
  111 / 10 = 11.1×
```

### Speed Improvement

**Formula:**

```
Speedup = t_baseline / t_optimized

Where:
  t_baseline = clustering time with full data
  t_optimized = clustering time with selected wavelengths
```

**Time Saved:**

```
Time Saved = t_baseline - t_optimized
Time Saved (%) = (Time Saved / t_baseline) × 100
```

### Quality Improvement

**Formula:**

```
Improvement (%) = (M_optimized - M_baseline) / M_baseline × 100

Where M is metric (purity, ARI, NMI, etc.)
```

**Example:**

```
Purity_baseline = 0.8543
Purity_optimized = 0.8682

Improvement = (0.8682 - 0.8543) / 0.8543 × 100
            = 0.0139 / 0.8543 × 100
            = 1.63%
```

### Efficiency Score

**Formula:**

```
Efficiency = Quality / Compression_Factor

Where:
  Quality = purity (or other metric)
  Compression_Factor = n_selected / n_total

Higher is better (high quality with low data usage)
```

**Example:**

```
Baseline:
  Efficiency = 0.8543 / (111/111) = 0.8543

Optimized:
  Efficiency = 0.8682 / (10/111) = 0.8682 / 0.090 = 9.65

Relative improvement:
  9.65 / 0.8543 = 11.3× better efficiency
```

---

## 10. Statistical Tests

### Significance Testing

**Paired t-test for metric comparison:**

```
Null hypothesis: μ_diff = 0

Test statistic:
  t = (x̄_diff - 0) / (s_diff / √n)

Where:
  x̄_diff = mean difference
  s_diff = standard deviation of differences
  n = number of samples

p-value from t-distribution with n-1 degrees of freedom
```

### Confidence Intervals

**95% Confidence Interval for purity:**

```
CI = μ̂ ± 1.96 × (σ̂ / √n)

Where:
  μ̂ = estimated purity
  σ̂ = estimated standard deviation
  n = number of runs/samples
```

---

## Summary of Key Equations

### Most Important Formulas:

1. **Variance selection:**
   ```
   I_var(d) = (1/N) Σᵢ (zᵢ[d] - μ_d)²
   ```

2. **Standard deviation perturbation:**
   ```
   δ_std = s × (m/100) × σ_d
   ```

3. **Influence measurement:**
   ```
   I(ex,em | d,δ) = (1/(N·H·W)) Σᵢₓᵧ |X̂'(ex,em)[x,y] - X̂(ex,em)[x,y]|
   ```

4. **MMR score:**
   ```
   MMR(wᵢ) = λ·Relevance(wᵢ) - (1-λ)·MaxSimilarity(wᵢ, S)
   ```

5. **Purity:**
   ```
   Purity = (1/N) Σᵢ max_j |Cᵢ ∩ Lⱼ|
   ```

---

**End of Mathematical Reference**
