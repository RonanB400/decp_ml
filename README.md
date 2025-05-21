# Analyse et surveillance des Marchés Publics et Accès aux Données via RAG

Ce projet vise à valoriser les données des marchés publics français à travers deux approches complémentaires : la **prédiction automatique du montant des marchés et l'analyse d'anomalies** et l’**accès conversationnel aux données** via un module RAG (Retrieval-Augmented Generation). Il repose sur l’exploitation de la base DECP (Données essentielles de la commande publique).

---

## 🧠 Objectifs du projet

1. **Estimation de montants** : prédire le montant estimé d’un marché public à partir de ses caractéristiques (code CPV, acheteur, durée, type de procédure, etc.).
2. **Détection d’anomalies** : identifier des marchés présentant des montants atypiques ou incohérents selon les tendances observées.
3. **Classification** : déterminer automatiquement la catégorie du marché (travaux, fournitures, services) à partir de ses métadonnées.
4. **Module RAG** : permettre des requêtes en langage naturel sur la base DECP et générer des synthèses personnalisées.
5. **Interface interactive** : proposer une interface Web (formulaire + chatbot) pour interagir avec les modèles prédictifs et le module RAG.

---

## 🗃️ Données utilisées

- **Source** : [Base DECP (SQLite)](https://www.data.gouv.fr/fr/datasets/r/43f54982-da60-4eb7-aaaf-ba935396209b)
- **Schéma relationnel** : [dbdiagram.io](https://dbdiagram.io/d/DATALAB-V4-67f00d0c4f7afba18464f539)
- **Champs clés** :
  - `objet`, `montant`, `date_notification`, `acheteur_id`, `nature`, `type_procedure`, `code_cpv`, `lieu_execution`, etc.

---

## 🧩 Modules du projet

### 📦 Module 1 : Estimation de montants
- Objectif : prédire un montant à partir des caractéristiques du marché.
- Modèles : Régression linéaire, Random Forest, XGBoost, MLP.
- Techniques : encodage de variables, évaluation (RMSE, MAE).

### 🧭 Module 2 : Détection d’anomalies
- Objectif : repérer les marchés dont les montants dévient des tendances habituelles.
- Méthodes : z-score, IQR, clustering, Isolation Forest.
- Intérêt : détecter des erreurs, irrégularités ou pratiques atypiques.

### 🏷️ Module 3 : Classification du type de marché
- Objectif : prédire la classe (`Travaux`, `Services`, `Fournitures`).
- Approches : SVM, régression logistique, arbres de décision.
- Gestion des classes déséquilibrées.

### 🔍 Module 4 : RAG (Retrieval-Augmented Generation)
- Objectif : interroger la base DECP en langage naturel.
- Technologies : FAISS, LangChain ou solution maison.
- Indexation : objets, descriptions, CCAG, lots.
- Exemples de requêtes :
  - *"Quels types de marchés sont passés par l’acheteur X en 2023 ?"*
  - *"Donne-moi un résumé des marchés contenant le mot cybersécurité."*

### 💬 Module 5 : Interface interactive
- Objectif : proposer une interface pour tester les prédictions et les requêtes RAG.
- Technologies : Streamlit ou FastAPI.
- Fonctionnalités :
  - Formulaire de prédiction à partir d’un marché fictif ou partiellement rempli.
  - Chatbot pour interroger la base via RAG.
  - (Bonus) Utilisation du prédicteur comme tool au sein d’un agent RAG.

---

## 🧰 Stack technique

- **Langage** : Python
- **Librairies principales** : `pandas`, `scikit-learn`, `xgboost`, `faiss`, `langchain`, `matplotlib`, `streamlit`, `fastapi`
- **Infrastructure** : Docker, Docker Compose
- **Interface Web** : Streamlit ou FastAPI + frontend léger

---

## 🚀 Installation et lancement

```bash
# Cloner le dépôt
git clone https://github.com/RonanB400/decp_ml.git
cd decp_ml

# Construire les containers Docker
docker-compose up --build
```

## 📁 Installation et lancement

├── data/                  # Données sources (SQLite, CSV)
├── notebooks/             # Analyses exploratoires
├── models/                # Modèles entraînés
├── app/                   # Code de l’interface (Streamlit / FastAPI)
├── rag/                   # Module RAG et indexation
├── predict/               # Modèles de prédiction
├── classify/              # Modèles de classification
├── detect/                # Détection d’anomalies
├── docker-compose.yml
└── README.md


## 📌 Livrables attendus
- Scripts de nettoyage et modélisation
- Modèles sauvegardés (.pkl, .joblib)
- Base vectorielle et index FAISS
- Application Web conteneurisée
- Documentation technique

