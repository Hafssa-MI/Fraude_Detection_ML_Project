# Détection de Fraude Bancaire — Classification Robuste en Environnement Déséquilibré

> **Projet Fin de module — Master SDIA1**  
> Implémentation complète d'un pipeline de machine learning pour la détection de fraudes sur transactions par carte bancaire, avec gestion du déséquilibre extrême, calibration des probabilités et interprétabilité.

---

## Table des Matières

1. [Problématique](#-problématique)
2. [Dataset](#-dataset)
3. [Structure du Projet](#-structure-du-projet)
4. [Installation](#-installation)
5. [Pipeline Complet](#-pipeline-complet)
   - [Étape 1 — EDA & Préparation](#étape-1--analyse-exploratoire-eda)
   - [Étape 2 — Feature Engineering](#étape-2--feature-engineering)
   - [Étape 3 — Préparation des Données](#étape-3--préparation-des-données)
   - [Étape 4 — Modèle 1 : Régression Logistique Elastic Net](#étape-4--modèle-1--régression-logistique-elastic-net)
   - [Étape 5 — Modèle 2 : Random Forest + Matrice de Proximité](#étape-5--modèle-2--random-forest--analyse-de-proximité)
   - [Étape 6 — Modèle 3 : XGBoost Cost-Sensitive + Optuna](#étape-6--modèle-3--xgboost-cost-sensitive--optuna)
   - [Étape 7 — Évaluation Comparative](#étape-7--évaluation-comparative)
   - [Étape 8 — Calibration des Probabilités](#étape-8--calibration-des-probabilités)
   - [Étape 9 — Interprétabilité SHAP](#étape-9--interprétabilité-shap)
6. [Résultats](#-résultats)
7. [Visualisations Clés](#-visualisations-clés)
8. [Bibliothèques Utilisées](#-bibliothèques-utilisées)
9. [Auteurs](#-auteurs)

---

## Problématique

### Contexte

Dans les systèmes bancaires réels, les fraudes représentent une **infime minorité** des transactions. Ce déséquilibre extrême pose un problème fondamental pour les algorithmes de machine learning classiques :

> Un modèle naïf qui prédit **"jamais fraude"** obtient une accuracy de **99.83%** — mais ne détecte aucune fraude. Cette métrique est donc **trompeuse et inutilisable**.

### Le Défi

| Caractéristique | Valeur |
|---|---|
| Transactions totales | 284 807 |
| Transactions légitimes | 283 253 (99.83%) |
| Transactions frauduleuses | 473 (0.17%) |
| Ratio de déséquilibre | **599 : 1** |
| Accuracy naïve trompeuse | **99.83%** |

### Objectifs du Projet

Ce projet répond à 4 objectifs scientifiques précis :

1. **Feature Engineering avancé** — Créer des variables pertinentes, encoder intelligemment, sélectionner par importance statistique
2. **Optimisation de modèles** — Comparer régression logistique, Random Forest et XGBoost avec justification des hyperparamètres
3. **Calibration de probabilités** — S'assurer que le score de sortie correspond à une réalité statistique (p=0.8 doit signifier 80% de chance de fraude)
4. **Interprétabilité** — Expliquer le "pourquoi" derrière chaque prédiction via SHAP

### Pourquoi ces métriques et pas l'Accuracy ?

| Métrique | Justification |
|---|---|
| **F1-Macro** | Moyenne non-pondérée du F1 par classe — traite les deux classes également malgré le déséquilibre |
| **MCC (Matthews Correlation Coefficient)** | Utilise les 4 cellules de la matrice de confusion (TP, TN, FP, FN) — robuste même avec déséquilibre extrême. Varie de -1 à +1 |
| **AUPRC (Area Under PR Curve)** | Plus informatif que AUC-ROC sur données déséquilibrées. La baseline aléatoire = 0.0017 (ratio fraudes) |
| ~~Accuracy~~ | **Exclue** — un classifieur naïf obtient 99.83% sans détecter aucune fraude |

---

## Dataset

**Source :** [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Description :**
- 284 807 transactions européennes sur 2 jours (septembre 2013)
- Features **V1 à V28** : composantes PCA anonymisées (données bancaires confidentielles)
- Feature **Time** : secondes écoulées depuis la première transaction
- Feature **Amount** : montant de la transaction
- Target **Class** : 0 = légitime, 1 = fraude

> **Note :** Les features V1–V28 sont orthogonales par construction (PCA), ce qui garantit l'absence de multicolinéarité entre elles — confirmé par la matrice de corrélation et le VIF.

---

## Structure du Projet

```
projet_fraude_detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv              ← Dataset original (jamais modifié)
│   └── processed/
│       ├── train_original.csv          ← Train set brut
│       ├── train_smote.csv             ← Train rééchantillonné SMOTE
│       ├── train_adasyn.csv            ← Train rééchantillonné ADASYN
│       └── train_nearmiss.csv          ← Train sous-échantillonné NearMiss
│
├── notebooks/
│   └── projet_fraude_detection.ipynb   ← Notebook principal unique
│
├── outputs/
│   ├── figures/                        ← Toutes les visualisations (.png)
│   ├── models/                         ← Modèles sauvegardés (.pkl / .json)
│   │   ├── logistic_elasticnet.pkl
│   │   ├── random_forest.pkl
│   │   └── xgboost_tuned.json
│   └── results/
│       └── comparison_table.csv        ← Métriques comparatives
│
├── report/
│   └── rapport_final.pdf
│
├── venv/                               ← Environnement virtuel Python
├── requirements.txt
└── README.md
```

---

## Installation

### Prérequis
- Python 3.10+
- VS Code avec extension Jupyter

### Étapes

```bash
# 1. Cloner / ouvrir le projet
cd projet_fraude_detection

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn \
            imbalanced-learn xgboost lightgbm optuna shap lime plotly \
            ipykernel notebook jupyterlab openpyxl

# 5. Enregistrer le kernel pour VS Code
python -m ipykernel install --user --name=venv_fraude --display-name "Python (fraude_detection)"

# 6. Sauvegarder les dépendances
pip freeze > requirements.txt
```

### Placer le dataset
Télécharger `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et le placer dans `data/raw/creditcard.csv`.

---

## Pipeline Complet

### Étape 1 — Analyse Exploratoire (EDA)

#### 1.1 Distribution des classes

Le dataset présente un déséquilibre **extrême** :

<!-- INSÉRER ICI : outputs/figures/2_1_class_distribution.png -->

> 📊 *Insérer ici la figure `2_1_class_distribution.png`*

```
Distribution :
  Non-Fraude (0) : 283 253 transactions (99.827%)
  Fraude    (1) :     473 transactions  (0.173%)
  Ratio          : 599:1
```

#### 1.2 Analyse de Amount et Time

Le montant (`Amount`) présente une distribution **fortement asymétrique** (skewed) avec des valeurs allant de 0€ à 25 691€.

<!-- INSÉRER ICI : outputs/figures/2_2_amount_time_distribution.png -->
> 📊 *Insérer ici la figure `2_2_amount_time_distribution.png`*

**Résultat du test Mann-Whitney U :** différence statistiquement significative entre les montants des fraudes et des transactions légitimes (p < 0.05).

#### 1.3 Distributions V1–V28 (Kolmogorov-Smirnov)

Test KS appliqué à chaque feature pour mesurer la différence de distribution entre fraudes et transactions légitimes.

<!-- INSÉRER ICI : outputs/figures/2_3_V1_V28_distributions.png -->
> 📊 *Insérer ici la figure `2_3_V1_V28_distributions.png`*

**Features les plus discriminantes (KS Stat > 0.3) :**
```
['V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V3', 'V16', 'V7',
 'V2', 'V9', 'V21', 'V18', 'V6', 'V27', 'V1', 'V5', 'V8', 'V20', 'V28', 'V19']
```
→ **21 features sur 28** sont fortement discriminantes.

#### 1.4 Matrice de Corrélation

<!-- INSÉRER ICI : outputs/figures/2_4_correlation_matrix.png -->
> 📊 *Insérer ici la figure `2_4_correlation_matrix.png`*

**Observations :**
- Les composantes V1–V28 sont **orthogonales par construction** (PCA) → aucune corrélation entre elles → matrice quasi-blanche
- Seule corrélation notable : **V2 ↔ Amount = -0.533**
- Top 5 features corrélées avec `Class` : V11, V4, V2, V8, V21

#### 1.5 VIF (Variance Inflation Factor)

<!-- INSÉRER ICI : outputs/figures/2_5_vif_scores.png -->
> 📊 *Insérer ici la figure `2_5_vif_scores.png`*

**Résultat :** Seule la feature `Amount` présente un VIF > 10 → multicolinéarité due à sa corrélation avec V2 (-0.533).  
**Décision :** Transformer `Amount` pour réduire son VIF.

---

### Étape 2 — Feature Engineering

#### 2.1 Transformation de Amount

Comparaison de trois transformations pour réduire l'asymétrie :

<!-- INSÉRER ICI : outputs/figures/3_1_amount_transformations.png -->
> 📊 *Insérer ici la figure `3_1_amount_transformations.png`*

| Transformation | Skewness | Kurtosis |
|---|---|---|
| Original | ~5.5 | ~50+ |
| log1p | ~0.8 | ~2.1 |
| **Yeo-Johnson** ✔ | **~0.1** | **~0.3** |

**Décision : Yeo-Johnson sélectionné** (skewness la plus proche de 0).

#### 2.2 Encodage Cyclique de Time

`Time` représente des secondes écoulées sur 48h. On le convertit en **heure de la journée** puis en encodage sin/cos pour capturer la périodicité sans discontinuité artificielle entre 23h et 0h.

<!-- INSÉRER ICI : outputs/figures/3_2_time_cyclique.png -->
> 📊 *Insérer ici la figure `3_2_time_cyclique.png`*

```
Time_sin = sin(2π × heure / 24)
Time_cos = cos(2π × heure / 24)
```

**Pourquoi sin + cos ?** `sin` seul est ambigu (0h = 12h). La paire `(sin, cos)` donne une représentation unique de chaque heure sur le cercle unitaire.

#### 2.3 Nouvelles Features Créées

| Feature | Formule | Justification |
|---|---|---|
| `Amount_transformed` | Yeo-Johnson(Amount) | Réduire skew, VIF > 10 |
| `Time_sin` | sin(2π × heure / 24) | Périodicité journalière |
| `Time_cos` | cos(2π × heure / 24) | Périodicité (sin ambigu seul) |
| `V2_x_Amount` | V2 × Amount_transformed | Corrélation V2-Amount = -0.533 |
| `V14_x_V12` | V14 × V12 | Top 2 features KS |
| `V11_x_V4` | V11 × V4 | Top corrélation avec Class |
| `Fraud_signal_norm` | \|\|(V14,V10,V12,V4,V11,V17)\|\|₂ | Intensité globale du signal fraude |
| `Is_night` | 1 si 22h ≤ heure ≤ 6h | Feature métier : fraudes nocturnes |

#### 2.4 Sélection des Features (Dual Ranking MI + RF)

<!-- INSÉRER ICI : outputs/figures/3_4_feature_importance.png -->
> 📊 *Insérer ici la figure `3_4_feature_importance.png`*

**Méthode :** Mutual Information + Random Forest rapide → ranking combiné  
**Résultat :** 32 features sélectionnées sur 36 (top 85%)  
**Features éliminées :** `Time_cos`, `V22`, `V24`, `V25` (faible pouvoir discriminant confirmé par KS)

---

### Étape 3 — Préparation des Données

#### 3.1 Split Stratifié

```
Train set : 198 607 observations (80%)
Val set   :  28 372 observations (10%) — calibration uniquement
Test set  :  56 746 observations (20%) — évaluation finale
```

> ⚠️ **Règle absolue :** Le test set n'est jamais touché pendant l'entraînement. Le `StandardScaler` est **fitté uniquement sur le train** puis appliqué sur val et test. Violer cette règle = **data leakage**.

#### 3.2 Stratégies de Rééchantillonnage

Quatre variantes du train set préparées pour comparaison :

<!-- INSÉRER ICI : outputs/figures/4_3_resampling_comparison.png -->
> 📊 *Insérer ici la figure `4_3_resampling_comparison.png`*

| Variante | Méthode | Mécanisme | Fraudes train |
|---|---|---|---|
| Original | — | Déséquilibré (référence) | ~380 |
| SMOTE | Over-sampling | Interpolation entre k=5 voisins minoritaires | ~198 000 |
| ADASYN | Over-sampling adaptatif | Plus de points dans les zones de faible densité | ~198 000 |
| NearMiss-3 | Under-sampling | Conserve les majoritaires les plus éloignés | ~380 |

> ⚠️ **Règle critique :** Le rééchantillonnage est appliqué **uniquement sur le train set**. Appliquer SMOTE sur le test = évaluer sur des données synthétiques → métriques artificiellement gonflées.

---

### Étape 4 — Modèle 1 : Régression Logistique Elastic Net

**Rôle :** Baseline linéaire interprétable. Permet de mesurer le gain des modèles complexes.

**Formule de la pénalité Elastic Net :**
```
λ [ l1_ratio × ||β||₁ + (1 - l1_ratio)/2 × ||β||₂² ]
```
- **L1 (Lasso)** → met des coefficients à zéro (sélection implicite de features)
- **L2 (Ridge)** → stabilise les coefficients corrélés

#### Hyperparamètres justifiés

| Paramètre | Valeur | Justification |
|---|---|---|
| `solver` | saga | Seul solver supportant Elastic Net sur grands datasets |
| `C` | [0.01, 0.1, 1, 10] | Inverse de la régularisation — grille sur 4 ordres de magnitude |
| `l1_ratio` | [0.1, 0.5, 0.9] | 0=Ridge pur, 1=Lasso pur, 0.5=équilibre |
| `max_iter` | 1000 | Convergence garantie sur 200k observations |
| `class_weight` | balanced | Compense le ratio 599:1 automatiquement |

#### Résultats

<!-- INSÉRER ICI : outputs/figures/5_2_gridsearch_classweight.png -->
> 📊 *Insérer ici la figure `5_2_gridsearch_classweight.png`*

<!-- INSÉRER ICI : outputs/figures/5_4_lr_coefficients.png -->
> 📊 *Insérer ici la figure `5_4_lr_coefficients.png`*

**Meilleure variante :** SMOTE (AUPRC CV = 0.991 vs 0.745 pour class_weight)  
**Coefficients mis à zéro par L1 :** 0/32 → l1_ratio faible, L2 domine

<!-- INSÉRER ICI : outputs/figures/5_5_lr_evaluation.png -->
> 📊 *Insérer ici la figure `5_5_lr_evaluation.png`*

```
LR Elastic Net — Test Set :
  F1-Macro : 0.5378
  MCC      : 0.2004
  AUPRC    : 0.7254
  Seuil optimal : à compléter depuis vos résultats
```

---

### Étape 5 — Modèle 2 : Random Forest + Analyse de Proximité

**Différence fondamentale avec LR :** capture les non-linéarités via 200 arbres de décision entraînés en parallèle (bagging).

#### Hyperparamètres justifiés

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 200 | Compromis stabilité/coût (standard académique Breiman 2001) |
| `max_depth` | None | Arbres non élagués — le bagging compense l'overfitting individuel |
| `max_features` | sqrt | √32 ≈ 5-6 features par split → décorrèle les arbres |
| `class_weight` | balanced_subsample | Recalculé à chaque bootstrap — plus robuste que 'balanced' |
| `oob_score` | True | Estimation gratuite de généralisation sans validation set |

<!-- INSÉRER ICI : outputs/figures/6_2_rf_feature_importance.png -->
> 📊 *Insérer ici la figure `6_2_rf_feature_importance.png`*

#### Matrice de Proximité — Concept Clé

La proximité entre deux observations i et j est définie comme la **proportion d'arbres dans lesquels i et j terminent dans la même feuille terminale** :

```
Proximité(i, j) = (1/N_arbres) × Σ 1[feuille_k(i) == feuille_k(j)]
```

**Propriétés :** symétrique, ∈ [0,1], non-linéaire (deux points éloignés en euclidien peuvent avoir une proximité élevée s'ils partagent les mêmes règles de décision).

<!-- INSÉRER ICI : outputs/figures/6_3_proximity_matrix.png -->
> 📊 *Insérer ici la figure `6_3_proximity_matrix.png`*

```
Matrice calculée sur : 5000 observations (sous-échantillon stratifié)
Justification : matrice 5000×5000 = 200 MB (faisable)
               matrice 198607×198607 = 316 GB (impossible)
Moyenne des proximités : 0.3617
Diagonale (auto-proximité) : 1.0000 ✔
```

#### Détection des Outliers de Prédiction

**Formule du score d'outlier :**
```
outlier_score(i) = 1 / Σ_{j ∈ même_classe_prédite} Proximité(i,j)²
```

Un score élevé → l'observation est **isolée** dans l'espace de décision des arbres → le modèle hésite ou échoue.

<!-- INSÉRER ICI : outputs/figures/6_5_proximity_outliers.png -->
> 📊 *Insérer ici la figure `6_5_proximity_outliers.png`*

```
Outliers détectés (P95) : 81 observations (1.6%)
  TN outliers : 73  → transactions légitimes atypiques
  TP outliers :  8  → fraudes détectées mais isolées dans l'espace
  FN outliers :  0  → les fraudes ratées ne sont pas hésitantes
  FP outliers :  0  → les fausses alarmes ne sont pas hésitantes
```

**Interprétation :** Les 28 fraudes ratées (FN) ne sont pas des cas où le modèle hésite — elles sont structurellement différentes des fraudes du train set. Le modèle est confiant (mais dans la mauvaise direction).

#### Résultats

<!-- INSÉRER ICI : outputs/figures/6_6_rf_evaluation.png -->
> 📊 *Insérer ici la figure `6_6_rf_evaluation.png`*

```
Random Forest — Test Set :
  F1-Macro : 0.9084   (+0.3706 vs LR) ✔
  MCC      : 0.8273   (+0.6269 vs LR) ✔
  AUPRC    : 0.8144   (+0.0890 vs LR) ✔
  Fraudes détectées : 67/95
```

---

### Étape 6 — Modèle 3 : XGBoost Cost-Sensitive + Optuna

**Différence avec Random Forest :** les arbres sont construits **séquentiellement**, chaque arbre corrigeant les erreurs du précédent (boosting vs bagging).

#### Deux Stratégies de Gestion du Déséquilibre

**Stratégie A — `scale_pos_weight` :**
```
scale_pos_weight = sum(négatifs) / sum(positifs) = 599
```
Multiplie le gradient de la classe minoritaire par 599 à chaque étape. Équivalent à un dataset parfaitement équilibré sans modifier les données.

**Stratégie B — Fonction de Perte Asymétrique (Custom Loss) :**

Formule :
```
L(y, p) = -[α × y × log(p) + β × (1-y) × log(1-p)]
```
avec α = 10 (pénalité FN) et β = 1 (pénalité FP)

Gradient et Hessien (requis par XGBoost) :
```
∂L/∂p = [-α×y/p + β×(1-y)/(1-p)] × p×(1-p)
∂²L/∂p² = [α×y/p² + β×(1-y)/(1-p)²] × (p×(1-p))²
```

<!-- INSÉRER ICI : outputs/figures/7_2_custom_loss.png -->
> 📊 *Insérer ici la figure `7_2_custom_loss.png`*

#### Optimisation Bayésienne — Optuna (TPE)

**Pourquoi Optuna plutôt que GridSearch ?**

| | GridSearch | Optuna TPE |
|---|---|---|
| Stratégie | Exhaustif — O(N^k) | Probabiliste — exploite l'historique |
| Trials | Tous les combins | 50 trials ciblés |
| Intelligence | Aucune | Modèle probabiliste l(x)/g(x) |
| Durée | >30 min | ~14 min |

**Espace de recherche justifié :**

| Hyperparamètre | Espace | Justification |
|---|---|---|
| `max_depth` | [3, 10] | 3=underfitting, 10=overfitting fort |
| `learning_rate` | [0.01, 0.3] log | Compromis vitesse/généralisation |
| `n_estimators` | [100, 1000] | Couplé à early stopping |
| `subsample` | [0.6, 1.0] | Stochasticité anti-overfitting |
| `colsample_bytree` | [0.6, 1.0] | Décorrèle les arbres (analogue max_features RF) |
| `reg_lambda` L2 | [0, 5] | Pénalise les feuilles à poids extrêmes |
| `reg_alpha` L1 | [0, 3] | Sparsité des feuilles |
| `min_child_weight` | [1, 10] | Contrôle la profondeur sur classe minoritaire |
| `gamma` | [0, 2] | Gain minimum pour effectuer un split |

<!-- INSÉRER ICI : outputs/figures/7_4_optuna_convergence.png -->
> 📊 *Insérer ici la figure `7_4_optuna_convergence.png`*

<!-- INSÉRER ICI : outputs/figures/7_4_param_importance.png -->
> 📊 *Insérer ici la figure `7_4_param_importance.png`*

#### Meilleurs Hyperparamètres Retenus (Optuna)

| Hyperparamètre | Valeur optimale |
|---|---|
| max_depth | 10 |
| learning_rate | 0.2399 |
| n_estimators | 664 |
| subsample | 0.7972 |
| colsample_bytree | 0.7521 |
| reg_lambda | 1.1828 |
| reg_alpha | 1.3730 |
| min_child_weight | 8 |
| gamma | 1.5881 |

#### Comparaison des Stratégies

| Stratégie | Best AUPRC (CV) | Durée |
|---|---|---|
| **A — scale_pos_weight** ✔ | **0.8376** | 13.5 min |
| B — Custom Loss | 0.8360 | 39.7 min |

→ **scale_pos_weight retenu** : meilleure performance ET 3× plus rapide.

#### Résultats

<!-- INSÉRER ICI : outputs/figures/7_5_xgb_evaluation.png -->
> 📊 *Insérer ici la figure `7_5_xgb_evaluation.png`*

```
XGBoost — Test Set :
  F1-Macro : 0.9301   ← best
  MCC      : 0.8650   ← best
  AUPRC    : 0.8176   ← best
  Fraudes détectées : 74/95
```

---

### Étape 7 — Évaluation Comparative

#### Tableau Comparatif Complet

| Modèle | F1-Macro | MCC | AUPRC | TP | FN | FP | Recall Fraude |
|---|---|---|---|---|---|---|---|
| LR Elastic Net | 0.5378 | 0.2004 | 0.7254 | — | — | — | — |
| Random Forest | 0.9084 | 0.8273 | 0.8144 | 67 | 28 | 2 | 0.705 |
| **XGBoost** ✔ | **0.9301** | **0.8650** | **0.8176** | **74** | **21** | **3** | **0.779** |

<!-- INSÉRER ICI : outputs/figures/8_2_pr_curves_comparison.png -->
> 📊 *Insérer ici la figure `8_2_pr_curves_comparison.png`*

<!-- INSÉRER ICI : outputs/figures/8_3_confusion_matrices.png -->
> 📊 *Insérer ici la figure `8_3_confusion_matrices.png`*

**Progression claire :**
```
F1-Macro : LR(0.5378) → RF(0.9084) → XGB(0.9301)
MCC      : LR(0.2004) → RF(0.8273) → XGB(0.8650)
AUPRC    : LR(0.7254) → RF(0.8144) → XGB(0.8176)
```

---

### Étape 8 — Calibration des Probabilités

**Problème :** Un modèle peut avoir d'excellentes métriques de classement mais des **probabilités mal calibrées** — p=0.8 ne signifie pas nécessairement 80% de chance de fraude.

**Outil de diagnostic :** Reliability Diagram (Diagramme de Fiabilité)
- Axe X : probabilité prédite (binned)
- Axe Y : fréquence réelle des positifs dans chaque bin
- Courbe parfaite = diagonale

<!-- INSÉRER ICI : outputs/figures/9_X_calibration_before.png -->
> 📊 *Insérer ici la figure de calibration avant correction*

**Deux méthodes de calibration :**

| Méthode | Type | Quand l'utiliser |
|---|---|---|
| **Platt Scaling** | Paramétrique (régression logistique sur les scores) | Courbe sigmoïde, peu de données |
| **Isotonic Regression** | Non-paramétrique | Plus flexible, beaucoup de données |

<!-- INSÉRER ICI : outputs/figures/9_X_calibration_after.png -->
> 📊 *Insérer ici la figure de calibration après correction*

```
Modèle retenu : XGBoost (scale_pos_weight)
Calibration   : Brut (meilleure calibration native)
Seuil optimal : 0.8540
```

---

### Étape 9 — Interprétabilité SHAP

**SHAP (Shapley Additive Explanations)** est basé sur la théorie des jeux coopératifs. La valeur Shapley d'une feature = sa **contribution marginale moyenne** à la prédiction, calculée sur toutes les coalitions possibles de features.

**Propriétés garanties :** efficacité, symétrie, linéarité, nulle si feature inutile.

#### Summary Plot — Importance Globale

<!-- INSÉRER ICI : outputs/figures/10_X_shap_summary_rf.png ou xgb -->
> 📊 *Insérer ici le Summary Plot SHAP*

**Top 5 features en consensus RF + XGBoost :**
```
['V14', 'V12', 'V4', 'Fraud_signal_norm', 'V8']
```

#### Force Plot — Explication Individuelle

<!-- INSÉRER ICI : outputs/figures/10_X_shap_force_plot.png -->
> 📊 *Insérer ici un Force Plot SHAP pour une fraude confirmée*

**Lecture :** Chaque barre représente la contribution d'une feature à l'écart entre la prédiction et la valeur moyenne. Rouge = pousse vers "fraude", bleu = pousse vers "légitime".

#### Dependence Plot — Effets Non-Linéaires

<!-- INSÉRER ICI : outputs/figures/10_X_shap_dependence.png -->
> 📊 *Insérer ici un Dependence Plot pour V14*

**Interprétation :** Une valeur faible de V14 → forte contribution SHAP positive → augmente le score de fraude. Effet non-linéaire invisible pour la régression logistique.

---

## 📊 Résultats

### Récapitulatif Exécutif

```
═══════════════════════════════════════════════════════════════
  RÉCAPITULATIF — PROJET CLASSIFICATION ROBUSTE
═══════════════════════════════════════════════════════════════

  PROBLÈME
  ────────
  283 253 transactions légitimes vs 473 fraudes
  Ratio de déséquilibre : 599:1
  Accuracy naïve trompeuse : 99.83%

  PIPELINE IMPLÉMENTÉ
  ───────────────────
  ✔ Feature Engineering  : 32 features (8 nouvelles, sélection MI+RF)
  ✔ Déséquilibre         : SMOTE / ADASYN / NearMiss / class_weight
  ✔ Modèles              : LR Elastic Net | RF 200 arbres | XGBoost Optuna
  ✔ Optimisation         : GridSearch (LR) | Bayesian TPE (XGBoost)
  ✔ Calibration          : Platt Scaling & Isotonic Regression
  ✔ Interprétabilité     : SHAP TreeExplainer
  ✔ Métriques            : AUPRC | MCC | F1-Macro (accuracy exclue)

  RÉSULTATS CLÉS
  ──────────────
  Baseline (LR Elastic Net) → AUPRC = 0.7254 | MCC = 0.2004
  Meilleur modèle retenu    → AUPRC = 0.8176 | MCC = 0.8650
  Gain vs baseline          → ΔAUPRC = +12.7% | ΔMCC = +331.9%

  MODÈLE RETENU : XGBoost (scale_pos_weight=599, Optuna 50 trials)
  Seuil optimal : 0.854
  SHAP Top 5    : V14, V12, V4, Fraud_signal_norm, V8
═══════════════════════════════════════════════════════════════
```

### Analyse des Erreurs Critiques

Dans la détection de fraude, les **Faux Négatifs** (fraudes non détectées) sont l'erreur la plus coûteuse :

| Modèle | FN (fraudes ratées) | FP (fausses alarmes) | Fraudes détectées |
|---|---|---|---|
| LR Elastic Net | — | — | — |
| Random Forest | 28 | 2 | 67/95 (70.5%) |
| **XGBoost** ✔ | **21** | **3** | **74/95 (77.9%)** |

---

## 📸 Visualisations Clés

> **Instructions :** Copier les fichiers PNG depuis `outputs/figures/` et les insérer aux emplacements marqués dans ce README.

| Figure | Chemin | Description |
|---|---|---|
| Distribution des classes | `outputs/figures/2_1_class_distribution.png` | Bar chart + pie chart du déséquilibre 599:1 |
| Amount & Time | `outputs/figures/2_2_amount_time_distribution.png` | Distributions, boxplots, statistiques comparatives |
| V1–V28 distributions | `outputs/figures/2_3_V1_V28_distributions.png` | Histogrammes KS test par feature |
| Matrice de corrélation | `outputs/figures/2_4_correlation_matrix.png` | Heatmap orthogonalité PCA |
| VIF scores | `outputs/figures/2_5_vif_scores.png` | Barplot VIF — Amount > 10 |
| Transformations Amount | `outputs/figures/3_1_amount_transformations.png` | Comparaison log1p vs Yeo-Johnson |
| Encodage cyclique Time | `outputs/figures/3_2_time_cyclique.png` | Cercle unitaire sin/cos |
| Feature importance | `outputs/figures/3_4_feature_importance.png` | Dual ranking MI + RF |
| Rééchantillonnage | `outputs/figures/4_3_resampling_comparison.png` | Original vs SMOTE vs ADASYN vs NearMiss |
| GridSearch LR | `outputs/figures/5_2_gridsearch_classweight.png` | Heatmap AUPRC (C, l1_ratio) |
| Coefficients LR | `outputs/figures/5_4_lr_coefficients.png` | Top 20 coefficients Elastic Net |
| Évaluation LR | `outputs/figures/5_5_lr_evaluation.png` | Matrice confusion + Courbe PR |
| RF Feature Importance | `outputs/figures/6_2_rf_feature_importance.png` | Gini importance + courbe cumulative |
| Matrice de proximité | `outputs/figures/6_3_proximity_matrix.png` | Heatmap + distribution des scores |
| Outliers prédiction | `outputs/figures/6_5_proximity_outliers.png` | Analyse des cas difficiles |
| Évaluation RF | `outputs/figures/6_6_rf_evaluation.png` | Matrice confusion + Courbe PR |
| Custom Loss | `outputs/figures/7_2_custom_loss.png` | Fonction de perte asymétrique |
| Convergence Optuna | `outputs/figures/7_4_optuna_convergence.png` | Optimization history |
| Importance params | `outputs/figures/7_4_param_importance.png` | FAnova hyperparamètres |
| Évaluation XGBoost | `outputs/figures/7_5_xgb_evaluation.png` | Matrice confusion + Courbe PR |
| Courbes PR comparées | `outputs/figures/8_2_pr_curves_comparison.png` | 3 modèles sur même graphe |
| Confusion matrices | `outputs/figures/8_3_confusion_matrices.png` | Comparaison côte à côte |

---

## 📚 Bibliothèques Utilisées

| Bibliothèque | Version | Usage |
|---|---|---|
| `numpy` | 2.4.6 | Calculs numériques |
| `pandas` | 3.0.3 | Manipulation des données |
| `scikit-learn` | 1.8.0 | Modèles, métriques, preprocessing |
| `xgboost` | 3.2.0 | Gradient boosting |
| `optuna` | 4.8.0 | Optimisation bayésienne TPE |
| `shap` | 0.51.0 | Interprétabilité SHAP |
| `imbalanced-learn` | — | SMOTE, ADASYN, NearMiss |
| `matplotlib` | — | Visualisations |
| `seaborn` | — | Visualisations statistiques |
| `scipy` | — | Tests statistiques (KS, Mann-Whitney) |
| `statsmodels` | — | VIF (Variance Inflation Factor) |

---

## 👥 Auteurs

> *Insérer ici les noms des membres du groupe*

**Encadrant :** *Insérer le nom du professeur*  
**Formation :** Master — Module Intelligence Artificielle  
**Année :** 2024–2025

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique.  
Dataset source : [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)