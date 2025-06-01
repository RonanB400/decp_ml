import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration de la page
st.set_page_config(
    page_title="🧾 Analyse des marchés publics DECP",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le style
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #1f77b4, #17a2b8);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🧾 Analyse des marchés publics et détection d'anomalies</h1>
        <p><strong>Projet de Data Science – Bootcamp Le Wagon, Batch #1992</strong></p>
        <p>👥 Équipe de 4 personnes – en collaboration avec <strong>Anticor</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu de navigation
    with st.sidebar:
        st.image("assets/logo.png" if os.path.exists("assets/logo.png") else "https://via.placeholder.com/200x100/1f77b4/white?text=DECP+ML", width=200)
        
        selected = option_menu(
            "Navigation",
            ["🏠 Accueil", "📊 Estimation", "🔍 Marchés similaires", "⚠️ Anomalies", "🤖 Chatbot RAG"],
            icons=['house', 'bar-chart', 'search', 'exclamation-triangle', 'robot'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
    
    # Navigation vers les différentes pages
    if selected == "🏠 Accueil":
        show_home()
    elif selected == "📊 Estimation":
        show_estimation()
    elif selected == "🔍 Marchés similaires":
        show_clustering()
    elif selected == "⚠️ Anomalies":
        show_anomalies()
    elif selected == "🤖 Chatbot RAG":
        show_rag()

def show_home():
    """Page d'accueil avec présentation du projet"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h2>🎯 Objectifs du projet</h2>
            <ul>
                <li><strong>Renforcer la transparence</strong> dans la commande publique</li>
                <li><strong>Aider les collectivités</strong> à mieux estimer les montants de leurs futurs marchés</li>
                <li><strong>Identifier les comportements atypiques</strong> dans les données ouvertes des marchés publics</li>
                <li><strong>Faciliter l'exploration</strong> des données via un chatbot RAG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h2>🔧 Modules disponibles</h2>
            <p><strong>📊 Estimation des montants</strong> : Prédiction de fourchettes de montants pour de nouveaux marchés</p>
            <p><strong>🔍 Marchés similaires</strong> : Recherche des 10 marchés les plus proches d'un marché donné</p>
            <p><strong>⚠️ Détection d'anomalies</strong> : Identification des marchés atypiques ou suspects</p>
            <p><strong>🤖 Chatbot RAG</strong> : Exploration conversationnelle de la base de données</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Statistiques</h3>
            <p>Chargement des données en cours...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h3>🤝 Partenariat</h3>
            <p>Ce projet est développé en collaboration avec <strong>Anticor</strong>, 
            association de lutte contre la corruption.</p>
        </div>
        """, unsafe_allow_html=True)

def show_estimation():
    """Module d'estimation des montants"""
    st.header("📊 Estimation des montants de marché")
    
    st.info("🚧 Module en cours de développement - Personne 1")
    
    with st.expander("ℹ️ À propos de ce module"):
        st.markdown("""
        **Objectif** : Prédire une tranche réaliste de montant pour un nouveau marché public
        
        **Approche** :
        - Transformation en problème de classification (fourchettes de montant)
        - Feature engineering sur CPV, procédure, localisation, acheteur
        - Modèles : XGBoost, Random Forest, SVM
        """)
    
    # Interface utilisateur pour la prédiction
    st.subheader("🔮 Prédire le montant d'un marché")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Intitulé du marché", placeholder="Ex: Fourniture de matériel informatique")
        st.text_area("Description détaillée", placeholder="Description du marché...")
        st.selectbox("Code CPV", ["48000000 - Progiciels et systèmes d'information", 
                                  "45000000 - Travaux de construction", 
                                  "50000000 - Services de réparation"])
    
    with col2:
        st.selectbox("Type de procédure", ["Appel d'offres ouvert", "Procédure adaptée", "Marché négocié"])
        st.selectbox("Type d'acheteur", ["Commune", "Département", "Région", "État"])
        st.selectbox("Localisation", ["Paris (75)", "Lyon (69)", "Marseille (13)"])
    
    if st.button("🔍 Estimer le montant", type="primary"):
        st.success("Fourchette estimée : **90 000€ - 220 000€** (Procédure formalisée européenne)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Montant minimum", "90 000€")
        with col2:
            st.metric("Montant médian", "155 000€")
        with col3:
            st.metric("Montant maximum", "220 000€")

def show_clustering():
    """Module de recherche de marchés similaires"""
    st.header("🔍 Recherche de marchés similaires")
    
    st.info("🚧 Module en cours de développement - Personne 2")
    
    with st.expander("ℹ️ À propos de ce module"):
        st.markdown("""
        **Objectif** : Trouver les 10 marchés passés les plus proches d'un marché donné
        
        **Approche** :
        - Vectorisation TF-IDF des descriptions + features numériques
        - Clustering (KMeans, DBSCAN) 
        - Recherche de proximité (NearestNeighbors)
        """)

def show_anomalies():
    """Module de détection d'anomalies"""
    st.header("⚠️ Détection d'anomalies")
    
    st.info("🚧 Module en cours de développement - Personnes 3 & 4")
    
    with st.expander("ℹ️ À propos de ce module"):
        st.markdown("""
        **Objectif** : Repérer les marchés anormaux (montants suspects, comportements hors normes)
        
        **Méthodes** :
        - **Isolation Forest** : analyse non supervisée pour repérer les points rares
        - **DBSCAN** : identifie les marchés hors des clusters denses
        - **GNN** : analyse des relations acheteurs-fournisseurs comme un graphe
        - **Visualisation** avec UMAP, t-SNE
        """)

def show_rag():
    """Module RAG pour l'exploration conversationnelle"""
    st.header("🤖 Chatbot RAG - Exploration conversationnelle")
    
    st.info("🚧 Module optionnel en cours de développement")
    
    with st.expander("ℹ️ À propos de ce module"):
        st.markdown("""
        **Objectif** : Permettre des questions en langage naturel sur la base DECP
        
        **Technologies** : LangChain, FAISS, embeddings open-source
        
        **Exemples de requêtes** :
        - "Quels sont les marchés de cybersécurité passés en 2023 dans le 69 ?"
        - "Montrez-moi les marchés suspects détectés cette année"
        """)
    
    # Interface de chat simulée
    st.text_input("💬 Posez votre question sur les marchés publics", 
                  placeholder="Ex: Quels sont les plus gros marchés de travaux en 2023 ?")
    
    if st.button("🚀 Poser la question"):
        st.chat_message("user").write("Quels sont les plus gros marchés de travaux en 2023 ?")
        st.chat_message("assistant").write("🚧 Fonctionnalité en cours de développement...")

if __name__ == "__main__":
    main() 