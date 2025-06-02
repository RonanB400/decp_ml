# 🧾 Analyse des marchés publics et détection d'anomalies

**Projet de Data Science – Bootcamp Le Wagon, Batch #1992** 

---

## 🎯 Objectifs

- **Renforcer la transparence** dans la commande publique
- **Aider les collectivités** à mieux estimer les montants de leurs futurs marchés  
- **Identifier les comportements atypiques** (voire suspects) dans les données ouvertes des marchés publics (données DECP)
- **En bonus** : faciliter l'exploration des données via un chatbot basé sur la technologie RAG

---

## 🔧 Modules du projet

### 🔹 Module 1 – Estimation des montants de marché (fourchettes)
**🔍 Objectif** : Prédire une tranche réaliste de montant pour un nouveau marché public

**🎯 Pourquoi** : Aider les collectivités à anticiper leurs dépenses et à prévenir les surcoûts

**🔍 Défi** : Les codes CPV sont parfois trop larges → une prédiction exacte est illusoire

**✅ Solution** :
- Transformation du problème en classification (fourchettes de montant)
- **Modèles testés** : XGBoost, Random Forest, SVM
- **Feature engineering** sur les colonnes CPV, procédure, localisation, acheteur, texte résumé...

### 🔹 Module 2 – Recherche de marchés similaires (clustering & matching)
**🔍 Objectif** : Trouver les 10 marchés passés les plus proches d'un marché donné

**🎯 Pourquoi** : Aider une collectivité à se comparer à des marchés déjà réalisés

**✅ Approches** :
- Vectorisation des marchés (TF-IDF sur la description, features numériques)
- Clustering (KMeans, DBSCAN) + recherche de proximité (NearestNeighbors)
- Moteur de similarité utilisable dans l'interface Streamlit

### 🔹 Module 3 – Détection d'anomalies dans les marchés publics
**🔍 Objectif** : Repérer les marchés anormaux dans la base (montants suspects, comportements hors normes)

**🎯 Pourquoi** : Lutter contre les dérives (clientélisme, favoritisme, entente)

**✅ Méthodes** :
- **Isolation Forest** : analyse non supervisée pour repérer les points rares
- **DBSCAN** : identifie les marchés hors des clusters denses → signal d'alerte
- **Visualisation** des marchés atypiques avec réduction de dimension (UMAP, t-SNE)
- **Test de GNN** (Graph Neural Network) pour analyser les relations acheteurs-fournisseurs comme un graphe : les communautés isolées ou trop denses peuvent signaler des comportements suspects

### 🧠 Module 4 (optionnel) – Interface RAG & exploration conversationnelle
**🔍 Objectif** : Permettre à un utilisateur de poser des questions naturelles sur la base

**Exemple** : *"Quels sont les marchés de cybersécurité passés en 2023 dans le 69 ?"*

**🧰 Stack** :
- LangChain, FAISS, embeddings open-source
- Construction d'un index vectoriel des résumés de marché
- Agent simple (ou chatbot) pour répondre aux requêtes de type : mots-clés, département, procédure, fourchette, etc.

---

## 🖥️ Interface utilisateur (Streamlit)

Une app simple avec 3 à 4 pages :
- **Estimation** d'un montant pour un nouveau marché
- **Suggestion** de marchés similaires  
- **Détection** d'anomalies
- **(Bonus)** Chatbot exploratoire pour naviguer dans la base

---

## 📁 Données

- **Source** : Données ouvertes DECP (data.gouv.fr)
- **État** : Déjà prétraitées (normalisation, nettoyage) + schéma de base structuré
- **Variables utilisées** : intitulé, description, acheteur, fournisseur, CPV, procédure, date, montant, localisation...

---

## 🧰 Stack technique

- **Langage** : Python 3.9+
- **ML/Data** : `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
- **Clustering** : `scikit-learn`, `umap-learn`
- **Anomalies** : `isolation-forest`, `dbscan`
- **RAG** : `langchain`, `faiss-cpu`, `sentence-transformers`
- **Interface** : `streamlit`
- **Infrastructure** : Docker, Docker Compose

---

## 🚀 Installation et lancement

### 📋 Prérequis

- **Python 3.10.6** via [pyenv](https://github.com/pyenv/pyenv)
- **Git**
- **Docker** et **Docker Compose** (optionnel)

### 🐍 Installation avec pyenv (recommandé)

```bash
# Cloner le dépôt
# Se positionner dans le dossier souhaité dans le terminal
git clone https://github.com/RonanB400/decp_ml.git
cd decp_ml

# Installer Python 3.10.6 avec pyenv (si pas déjà fait)
pyenv install 3.10.6

# Créer et activer votre environnement virtuel
pyenv virtualenv 3.10.6 decp_ml_env
pyenv local decp_ml_env

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔐 Configuration des variables d'environnement

```bash
# Copier le fichier d'exemple des variables d'environnement
cp .env.example .env

# Éditer le fichier .env avec vos clés API
nano .env  # ou votre éditeur préféré

# (Optionnel) Si vous utilisez direnv pour l'auto-chargement
cp .envrc.example .envrc
direnv allow
```

### 🚀 Lancement de l'application

```bash
# Lancer l'application Streamlit
streamlit run app/main.py

# L'application sera accessible sur http://localhost:8501
```

### 🐳 Installation avec Docker (alternative)

```bash
# Construire et lancer les containers
docker-compose up --build

# L'application sera accessible sur http://localhost:8501
# Jupyter Lab sera accessible sur http://localhost:8888
```

---

## 📁 Structure du projet

```
decp_ml/
├── 📂 data/                    # Données sources (SQLite, CSV)
│   ├── decp.sqlite            # Base DECP principale
│   └── datalab.sqlite         # Base traitée
├── 📂 notebooks/              # Analyses exploratoires
├── 📂 src/                    # Code source principal
│   ├── 📂 estimation/         # Module 1: Estimation montants
│   ├── 📂 clustering/         # Module 2: Marchés similaires  
│   ├── 📂 anomalies/          # Module 3: Détection anomalies
│   └── 📂 rag/               # Module 4: Interface RAG
├── 📂 models/                 # Modèles entraînés sauvegardés
│   ├── 📂 estimation/
│   ├── 📂 clustering/
│   └── 📂 anomalies/
├── 📂 app/                    # Interface Streamlit
│   ├── 📂 pages/             # Pages de l'application
│   ├── 📂 components/        # Composants réutilisables
│   └── 📂 utils/             # Utilitaires interface
├── 📂 tests/                  # Tests unitaires
├── 📂 docs/                   # Documentation
├── 📂 config/                 # Fichiers de configuration
├── 📂 scripts/                # Scripts utilitaires
├── .env.example               # Template des variables d'environnement
├── .envrc.example             # Template direnv (optionnel)
├── .python-version            # Version Python pour pyenv
├── requirements.txt           # Dépendances Python
├── docker-compose.yml         # Configuration Docker
├── Dockerfile                 # Image Docker
└── README.md                  # Ce fichier
```

---

## 📖 Explication détaillée de la structure

### 🎯 **Dossiers principaux**

#### 📂 **`src/`** - Code source principal
- **`src/estimation/`** - Module 1 : Estimation des montants (Personne 1)
  - Classification par fourchettes de montants
  - Feature engineering sur CPV, procédure, localisation
  - Modèles : XGBoost, Random Forest, SVM

- **`src/clustering/`** - Module 2 : Recherche de marchés similaires (Personne 2)
  - Vectorisation TF-IDF des descriptions
  - Clustering KMeans, DBSCAN
  - Moteur de recherche de proximité

- **`src/anomalies/`** - Module 3 : Détection d'anomalies (Personnes 3 & 4)
  - Isolation Forest pour détecter les points rares
  - DBSCAN pour identifier les outliers
  - GNN pour analyser les relations acheteurs-fournisseurs
  - Visualisation avec UMAP, t-SNE

- **`src/rag/`** - Module 4 : Interface RAG (optionnel)
  - Indexation vectorielle avec FAISS
  - LangChain pour les requêtes en langage naturel
  - Embeddings des descriptions de marchés

#### 📂 **`app/`** - Interface utilisateur Streamlit
- **`app/main.py`** - Application principale avec navigation entre modules
- **`app/pages/`** - Pages spécifiques (estimation, clustering, anomalies, RAG)
- **`app/components/`** - Composants réutilisables (graphiques, widgets, formulaires)
- **`app/utils/`** - Utilitaires pour l'interface (helpers, formatage)

#### 📂 **`models/`** - Modèles entraînés sauvegardés
- **`models/estimation/`** - Modèles de classification des montants (.pkl, .joblib)
- **`models/clustering/`** - Modèles de clustering et vectoriseurs
- **`models/anomalies/`** - Modèles de détection d'anomalies

### 🔧 **Dossiers de support**

#### 📂 **`data/`** - Données (existant)
- **`datalab.sqlite`** - Base de données traitée
- **`processed/`** - Données preprocessées pour chaque module

#### 📂 **`config/`** - Configuration
- **`config.yaml`** - Paramètres centralisés (seuils, chemins, hyperparamètres)
- Configuration des modèles, RAG, interface Streamlit

#### 📂 **`notebooks/`** - Analyses exploratoires
(ajoutez vos initiales dans le nom de vos notebooks)
- Notebooks Jupyter pour l'exploration des données
- Prototypage des modèles
- Analyses statistiques et visualisations

#### 📂 **`tests/`** - Tests unitaires
- Tests pour valider chaque module
- Tests d'intégration de l'interface
- Validation des modèles

#### 📂 **`docs/`** - Documentation
- Documentation technique détaillée
- Guides utilisateur
- Rapports d'analyse
- Images, logos pour l'interface
- Fichiers CSS personnalisés
- Schémas et diagrammes

#### 📂 **`scripts/`** - Scripts utilitaires
- Scripts de preprocessing des données
- Scripts de déploiement
- Utilitaires de maintenance


### ⚙️ **Fichiers de configuration**

- **`requirements.txt`** - Dépendances Python avec versions spécifiques
- **`Dockerfile`** - Image Docker pour conteneurisation
- **`docker-compose.yml`** - Orchestration des services (app + jupyter)
- **`.env.example`** - Template des variables d'environnement (clés API, secrets)
- **`.envrc.example`** - Template direnv pour auto-chargement des variables
- **`.python-version`** - Version Python fixée pour pyenv
- **`README.md`** - Documentation principale du projet

### 🔐 **Gestion des secrets et variables d'environnement**

#### Variables d'environnement nécessaires :
```bash
# API Keys (exemples)
OPENAI_API_KEY=sk-...                    # Pour le module RAG
HUGGINGFACE_API_TOKEN=hf_...             # Pour les embeddings
STREAMLIT_SECRET_KEY=your-secret-key     # Pour les sessions Streamlit

# Configuration base de données
DATABASE_URL=sqlite:///data/datalab.sqlite  # Chemin vers la base DECP
LOG_LEVEL=INFO                           # Niveau de logging

# Configuration modèles
MODEL_CACHE_DIR=./models                 # Dossier de cache des modèles
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### Fichiers à ne PAS commiter :
- **`.env`** - Variables personnelles (ajouté au .gitignore)
- **`.envrc`** - Configuration direnv personnelle (ajouté au .gitignore)
- **`models/*.pkl`** - Modèles entraînés volumineux

### 🚀 **Workflow de développement**

1. **Développement** : Chaque personne code dans son module `src/`
2. **Tests** : Création de tests unitaires dans `tests/`
3. **Sauvegarde** : Modèles entraînés dans `models/`
4. **Documentation** : Ajout de docs dans `docs/`
5. **Intégration** : Interface commune dans `app/`
6. **Déploiement** : Via Docker avec `docker-compose up`


---

## 📌 Livrables attendus

- ✅ Scripts de preprocessing et feature engineering
- ✅ Modèles entraînés et sauvegardés (`.pkl`, `.joblib`)
- ✅ Base vectorielle et index FAISS (pour RAG)
- ✅ Application Web interactive (Streamlit)
- ✅ Documentation technique et guide utilisateur
- ✅ Tests unitaires et validation des modèles
- ✅ Rapport d'analyse et recommandations

---


