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

<img width="1657" height="663" alt="image" src="https://github.com/user-attachments/assets/951112e2-7da7-4814-b930-2df836dcdc45" />

```
Distribution :
  Non-Fraude (0) : 283 253 transactions (99.827%)
  Fraude    (1) :     473 transactions  (0.173%)
  Ratio          : 599:1
```

#### 1.2 Analyse de Amount et Time

Le montant (`Amount`) présente une distribution **fortement asymétrique** (skewed) avec des valeurs allant de 0€ à 25 691€.

<img width="2314" height="1274" alt="image" src="https://github.com/user-attachments/assets/74d3d820-75ec-46aa-b0eb-6477b780ca9c" />

**Résultat du test Mann-Whitney U :** différence statistiquement significative entre les montants des fraudes et des transactions légitimes (p < 0.05).

#### 1.3 Distributions V1–V28 (Kolmogorov-Smirnov)

Test KS appliqué à chaque feature pour mesurer la différence de distribution entre fraudes et transactions légitimes.

<img width="2574" height="3567" alt="image" src="https://github.com/user-attachments/assets/b53096ab-1932-4ee7-add2-c511a69d3b4c" />


**Features les plus discriminantes (KS Stat > 0.3) :**
```
['V14', 'V10', 'V12', 'V4', 'V11', 'V17', 'V3', 'V16', 'V7',
 'V2', 'V9', 'V21', 'V18', 'V6', 'V27', 'V1', 'V5', 'V8', 'V20', 'V28', 'V19']
```
→ **21 features sur 28** sont fortement discriminantes.

#### 1.4 Matrice de Corrélation

<img width="2625" height="2314" alt="image" src="https://github.com/user-attachments/assets/91e073c7-eb75-436e-aa24-77908587c491" />


**Observations :**
- Les composantes V1–V28 sont **orthogonales par construction** (PCA) → aucune corrélation entre elles → matrice quasi-blanche
- Seule corrélation notable : **V2 ↔ Amount = -0.533**
- Top 5 features corrélées avec `Class` : V11, V4, V2, V8, V21

#### 1.5 VIF (Variance Inflation Factor)

<img width="1794" height="884" alt="image" src="https://github.com/user-attachments/assets/6893d0c1-b55c-4feb-82e1-104f409b42cb" />


**Résultat :** Seule la feature `Amount` présente un VIF > 10 → multicolinéarité due à sa corrélation avec V2 (-0.533).  
**Décision :** Transformer `Amount` pour réduire son VIF.

---

### Étape 2 — Feature Engineering

#### 2.1 Transformation de Amount

Comparaison de trois transformations pour réduire l'asymétrie :

<img width="2314" height="1274" alt="image" src="https://github.com/user-attachments/assets/34305ed3-8ec0-4dc5-b904-251bb543bbd7" />


| Transformation | Skewness | Kurtosis |
|---|---|---|
| Original | ~5.5 | ~50+ |
| log1p | ~0.8 | ~2.1 |
| **Yeo-Johnson** ✔ | **~0.1** | **~0.3** |

**Décision : Yeo-Johnson sélectionné** (skewness la plus proche de 0).

#### 2.2 Encodage Cyclique de Time

`Time` représente des secondes écoulées sur 48h. On le convertit en **heure de la journée** puis en encodage sin/cos pour capturer la périodicité sans discontinuité artificielle entre 23h et 0h.

<img width="2314" height="1274" alt="image" src="https://github.com/user-attachments/assets/a1771010-8560-4759-a185-81bfafee704a" />


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

<img width="2314" height="1274" alt="image" src="https://github.com/user-attachments/assets/ac42f13b-e369-4542-ab78-5db530c07ed1" />


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

<img width="2314" height="637" alt="image" src="https://github.com/user-attachments/assets/8d608a5e-cfc6-4646-addf-09e494217447" />


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

<img width="999" height="624" alt="image" src="https://github.com/user-attachments/assets/c04355ec-9395-4b1c-970c-b573ca85c1ed" />


<img width="1534" height="1014" alt="image" src="https://github.com/user-attachments/assets/0614aee8-cf0a-469e-b9ab-1238a3085161" />


**Meilleure variante :** SMOTE (AUPRC CV = 0.991 vs 0.745 pour class_weight)  
**Coefficients mis à zéro par L1 :** 0/32 → l1_ratio faible, L2 domine


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

<img width="2314" height="1019" alt="image" src="https://github.com/user-attachments/assets/a6d53808-82d0-44d9-81e3-b0f3fcb61fd6" />


#### Matrice de Proximité — Concept Clé

La proximité entre deux observations i et j est définie comme la **proportion d'arbres dans lesquels i et j terminent dans la même feuille terminale** :

```
Proximité(i, j) = (1/N_arbres) × Σ 1[feuille_k(i) == feuille_k(j)]
```

**Propriétés :** symétrique, ∈ [0,1], non-linéaire (deux points éloignés en euclidien peuvent avoir une proximité élevée s'ils partagent les mêmes règles de décision).

<img width="2054" height="764" alt="image" src="https://github.com/user-attachments/assets/2d704dbb-4b26-4cd3-96cb-ac9c05cfcffa" />


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

<img width="2054" height="1529" alt="image" src="https://github.com/user-attachments/assets/8b04dc1c-6804-40e6-8b54-ab2ad86b5074" />


```
Outliers détectés (P95) : 81 observations (1.6%)
  TN outliers : 73  → transactions légitimes atypiques
  TP outliers :  8  → fraudes détectées mais isolées dans l'espace
  FN outliers :  0  → les fraudes ratées ne sont pas hésitantes
  FP outliers :  0  → les fausses alarmes ne sont pas hésitantes
```

**Interprétation :** Les 28 fraudes ratées (FN) ne sont pas des cas où le modèle hésite — elles sont structurellement différentes des fraudes du train set. Le modèle est confiant (mais dans la mauvaise direction).

#### Résultats

<img width="1680" height="637" alt="image" src="https://github.com/user-attachments/assets/8c8e1b7e-fb08-49e5-aa60-a91243e6ed53" />


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

<img width="1794" height="637" alt="image" src="https://github.com/user-attachments/assets/1347ae8e-8291-4bfd-bfa6-b6582b6e65fc" />


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

<img width="2311" height="1529" alt="image" src="https://github.com/user-attachments/assets/2d68324d-1885-45d7-bf00-243677db6409" />

<img width="2052" height="764" alt="image" src="https://github.com/user-attachments/assets/e18e6c3d-4bdb-4e0f-b430-07945df5333f" />


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

<img width="1937" height="764" alt="image" src="https://github.com/user-attachments/assets/37b77196-b5dc-4d9b-9545-a08a65056481" />


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

<img width="2314" height="892" alt="image" src="https://github.com/user-attachments/assets/3432683a-897f-4de7-b1ae-602aa812de9c" />


<img width="2278" height="1401" alt="image" src="https://github.com/user-attachments/assets/b5a86413-7662-440b-a2bc-51977cd6a5bb" />


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

<img width="2054" height="2101" alt="image" src="https://github.com/user-attachments/assets/54e16568-8ebe-4425-994b-109f285dfb85" />


**Deux méthodes de calibration :**

| Méthode | Type | Quand l'utiliser |
|---|---|---|
| **Platt Scaling** | Paramétrique (régression logistique sur les scores) | Courbe sigmoïde, peu de données |
| **Isotonic Regression** | Non-paramétrique | Plus flexible, beaucoup de données |

<img width="2574" height="892" alt="image" src="https://github.com/user-attachments/assets/db39db5a-6175-41e3-9477-4db55e12d341" />


```
Modèle retenu : XGBoost (scale_pos_weight)
Calibration   : Brut (meilleure calibration native)
Seuil optimal : 0.8540
```

---

### Étape 9 — Interprétabilité SHAP

**SHAP (Shapley Additive Explanations)** est basé sur la théorie des jeux coopératifs. La valeur Shapley d'une feature = sa **contribution marginale moyenne** à la prédiction, calculée sur toutes les coalitions possibles de features.

**Propriétés garanties :** efficacité, symétrie, linéarité, nulle si feature inutile.

---

#### 9.1 Summary Plot — Importance Globale (Random Forest)

<img width="1433" height="1274" alt="image" src="https://github.com/user-attachments/assets/edcf6719-9d99-41e9-a5b2-82482186921d" />


**Lecture :**
- Axe X : valeur SHAP (contribution à la prédiction de fraude)
- Couleur : valeur de la feature (Rouge = élevée, Bleu = faible)
- `Fraud_signal_norm` en tête → notre feature engineerée capte le signal global
- `V14` faible (bleu) → pousse fortement vers fraude (SHAP négatif = contribution à la classe positive dans ce contexte)

---

#### 9.2 Waterfall Plot — Explication Individuelle (Random Forest)

<img width="1037" height="1144" alt="image" src="https://github.com/user-attachments/assets/30f4141f-a6f4-41ef-804f-333d56b521b6" />


**Lecture :** Fraude confirmée avec score = 0.965
- Point de départ : E[f(X)] = 0.5 (probabilité moyenne du modèle)
- `Fraud_signal_norm` (+0.11) → contribution la plus forte vers fraude
- `V14_x_V12` (+0.08) → notre feature d'interaction confirme le signal
- `V8` (-0.02) → légèrement contre la prédiction fraude sur ce cas précis
- Score final : f(x) = 0.965 → le modèle est très confiant

---

#### 9.3 Summary Plot — Importance Globale (XGBoost)

<img width="1437" height="1274" alt="image" src="https://github.com/user-attachments/assets/e41ad7a6-b6ee-4159-9964-48b4e7dfd1b6" />


**Observations vs Random Forest :**
- `V14` reste la feature dominante dans les deux modèles ✔
- XGBoost accorde plus de poids à `V8`, `V21`, `V20` que RF
- `Fraud_signal_norm` reste dans le top 15 → feature engineerée validée

---

#### 9.4 Bar Plot — Importance Globale (XGBoost)

<img width="1534" height="1014" alt="image" src="https://github.com/user-attachments/assets/8697c3a3-8d14-4320-9c39-378472b4a927" />


**Top 5 features XGBoost (|SHAP| moyen) :**

| Rang | Feature | |SHAP| moyen |
|---|---|---|
| 1 | V14 | 0.00137 |
| 2 | V8 | 0.00117 |
| 3 | V21 | 0.00090 |
| 4 | V20 | 0.00079 |
| 5 | V13 | 0.00075 |

---

#### 9.5 Waterfall Plot — Explication Individuelle (XGBoost)

<img width="2024" height="1575" alt="image" src="https://github.com/user-attachments/assets/9f58c73f-18a4-48df-822c-c5e91649c326" />


**Lecture :** Même fraude confirmée (score = 1.0000)
- Point de départ : E[f(X)] = 0.0101 (probabilité moyenne XGBoost, très faible car données déséquilibrées)
- `V14` (+0.20) → contribution individuelle la plus forte vers fraude
- `Fraud_signal_norm` (+0.16) → notre feature engineerée confirme le signal
- `V4` (+0.15), `V12` (+0.14) → features PCA discriminantes
- `V23` (-0.05) → légèrement contre la prédiction fraude
- Score final : f(x) = 1.0 → certitude maximale du modèle

---

#### 9.6 Dependence Plots — Effets Non-Linéaires (XGBoost)

<img width="1014" height="1144" alt="image" src="https://github.com/user-attachments/assets/db12e252-4292-4bf5-a68d-435dd72fc9f9" />


**Lecture par feature :**

| Feature | Observation | Interprétation |
|---|---|---|
| **V14** | Valeurs < -5 → SHAP positif élevé | Une valeur très négative de V14 = fort signal fraude |
| **V8** | Valeurs < -5 → SHAP négatif | Grandes valeurs négatives de V8 réduisent le score |
| **V21** | Concentré autour de 0 | Effet principalement pour les valeurs extrêmes |
| **V20** | Valeurs autour de 0 → SHAP négatif | Signal fraude pour V20 ≈ 0 |

> Ces effets **non-linéaires** sont invisibles pour la régression logistique → justifie l'utilisation de XGBoost.

---

#### 9.7 Comparaison RF vs XGBoost — Cohérence SHAP

<img width="2054" height="1014" alt="image" src="https://github.com/user-attachments/assets/d08bae97-6f86-40d0-8301-1b64d87f6b81" />

<img width="1217" height="1014" alt="image" src="https://github.com/user-attachments/assets/dbf49fee-9312-4a0f-a092-5d8f679f60a3" />


**Corrélation de Spearman entre rankings SHAP : ρ = 0.357 (p = 0.045)**

**Consensus entre les deux modèles :**

| Feature | RF | XGBoost | Consensus |
|---|---|---|---|
| V14 | #2 | #1 | ✔ Fort |
| V12 | #3 | #6 | ✔ Modéré |
| V4 | #4 | #8 | ✔ Modéré |
| Fraud_signal_norm | #1 | #13 | ⚠ Divergent |
| V8 | #15 | #2 | ⚠ Divergent |

**Interprétation de la divergence :**
- `Fraud_signal_norm` : RF lui donne le rang #1 (norme L2 des top features = signal global) mais XGBoost le décompose en ses composantes individuelles (V14, V8, V12 séparément) → même information, représentation différente
- `V8` : XGBoost détecte un signal fort que RF diffuse sur plusieurs features corrélées
- La corrélation ρ = 0.357 significative (p < 0.05) confirme un **consensus partiel** → les deux modèles s'accordent sur les features les plus importantes

**Top 5 features en consensus RF + XGBoost :**
```
['V14', 'V12', 'V4', 'Fraud_signal_norm', 'V8']
```
## Résultats

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


## Bibliothèques Utilisées

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

## Auteurs

> HAFSSA MIFTAH IDRISSI - HANANE AIT LHAJ

**Encadrant :** Mme Asmae Ouhmida  
**Formation :** Master — SDIA 1 
**Année :** 2025–2026

---

## Licence

Ce projet est réalisé dans un cadre académique.  
Dataset source : [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
