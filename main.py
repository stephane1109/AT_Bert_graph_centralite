import os
import math
import re
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import networkx as nx
from pyvis.network import Network
from collections import Counter
import community.community_louvain as community_louvain
import spacy

# ---------------------------------------------------------------------
# Configuration spaCy et modèle CamemBERT
# ---------------------------------------------------------------------
spacy_nlp = spacy.load("fr_core_news_sm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LENGTH = 512
NB_NEIGHBORS_DEFAULT = 20
TOP_N_DEFAULT = 100

MODEL_NAME = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------
# Variables globales
# ---------------------------------------------------------------------
selected_filepath = ""
text_widget = None

# Options Tkinter pour le prétraitement et l'embedding
pooling_method = None      # "mean", "weighted", "sif"
var_intra = None           # booléen, optionnel pour arêtes intracommunautaires
var_stopwords = None       # booléen
var_lemmatisation = None   # booléen

# Sélection de la mesure de centralité ("degree", "betweenness", "closeness", "eigenvector")
centrality_method = None

# Paramètre k pour l'approche k‑NN (nombre de voisins connectés par nœud)
# Ce paramètre sera saisi via l'interface

# ---------------------------------------------------------------------
# Fonctions d'affichage et de sélection de fichier
# ---------------------------------------------------------------------
def afficher_message(msg):
    """Affiche un message dans la zone de texte ou dans la console."""
    global text_widget
    if text_widget is not None:
        text_widget.insert(tk.END, msg + "\n")
        text_widget.see(tk.END)
    else:
        print(msg)

def selectionner_fichier():
    """Ouvre une boîte de dialogue pour sélectionner un fichier texte."""
    global selected_filepath
    path = filedialog.askopenfilename(filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")])
    if path:
        selected_filepath = path
        afficher_message("Fichier sélectionné : " + selected_filepath)

# ---------------------------------------------------------------------
# Prétraitement du texte
# ---------------------------------------------------------------------
def pretraiter_phrase(phrase):
    """Traite la phrase avec spaCy en appliquant la lemmatisation et le retrait des stopwords."""
    doc = spacy_nlp(phrase)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return " ".join(tokens)

def extraire_termes_frequents(texte, top_n):
    """Extrait les termes les plus fréquents (NOUN, PROPN) du texte."""
    doc = spacy_nlp(texte)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    freq = Counter(tokens)
    return dict(freq.most_common(top_n))

def normaliser_texte(text):
    """Normalise le texte en supprimant les lignes indésirables et les espaces superflus."""
    lignes = text.splitlines()
    lignes_filtrees = [l for l in lignes if not l.strip().startswith("****")]
    texte_filtre = " ".join(lignes_filtrees)
    return re.sub(r'\s+', ' ', texte_filtre).strip().lower()

def split_text_into_sentences(text):
    """Découpe le texte en phrases en se basant sur la ponctuation (. ! ?)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------------------------------------------------------------------
# Fonctions d'embedding (avec CamemBERT)
# Nous conservons Mean pooling, Weighted pooling et SIF pooling.
# ---------------------------------------------------------------------
def encoder_phrase(phrase):
    """Encode une phrase via CamemBERT selon la méthode de pooling choisie."""
    from collections import Counter
    phrase_pretraitee = pretraiter_phrase(phrase)
    inputs = tokenizer(phrase_pretraitee, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    method = pooling_method.get()
    if method == "mean":
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)
    elif method == "weighted":
        tokens_pretraite = phrase_pretraitee.split()
        freq_dict = Counter(tokens_pretraite)
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                word = token[1:]
                weights.append(freq_dict.get(word, 1))
            else:
                weights.append(weights[-1] if weights else 1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * weights_tensor).sum(dim=1)
        normalization = weights_tensor.sum()
        emb = weighted_embeds / (normalization if normalization != 0 else 1)
        return emb.cpu().numpy().squeeze(0)
    elif method == "sif":
        a = 0.001
        tokens_pretraitee = phrase_pretraitee.split()
        if len(tokens_pretraitee) == 0:
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)
        from collections import Counter
        counts = Counter(tokens_pretraitee)
        total_tokens = len(tokens_pretraitee)
        sif_weights_list = [a / (a + counts[token] / total_tokens) for token in tokens_pretraitee]
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        sif_weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                sif_weights.append(sif_weights_list.pop(0) if sif_weights_list else 1)
            else:
                sif_weights.append(sif_weights[-1] if sif_weights else 1)
        sif_weights_tensor = torch.tensor(sif_weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        sif_weights_tensor = sif_weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * sif_weights_tensor).sum(dim=1)
        normalization = sif_weights_tensor.sum()
        emb = weighted_embeds / (normalization if normalization != 0 else 1)
        return emb.cpu().numpy().squeeze(0)
    else:
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)

def encoder_contextuel_simplifie(texte, mot_cle):
    """
    Calcule l'embedding du mot-clé en contexte en moyennant les embeddings
    de toutes les phrases contenant le mot.
    Si aucune phrase n'est trouvée, encode simplement le mot-clé.
    """
    sentences = split_text_into_sentences(texte)
    pertinentes = [s for s in sentences if mot_cle.lower() in s.lower()]
    afficher_message(f"Nombre de phrases contextuelles pour '{mot_cle}' : {len(pertinentes)}")
    if not pertinentes:
        return encoder_phrase(mot_cle)
    embeddings = [encoder_phrase(s) for s in pertinentes]
    return np.mean(embeddings, axis=0)

def encoder_terme_par_contexte(terme, texte):
    """
    Pour un terme donné, encode les phrases le contenant et retourne
    (embedding moyen, liste des phrases contextuelles).
    """
    sentences = split_text_into_sentences(texte)
    context_sentences = [s for s in sentences if terme.lower() in s.lower()]
    if not context_sentences:
        return encoder_phrase(terme), []
    embeddings = [encoder_phrase(s) for s in context_sentences]
    return np.mean(embeddings, axis=0), context_sentences

def cosine_similarity(v1, v2):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    Retourne 0 si la valeur calculée est négative.
    """
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return max(sim, 0)

# ---------------------------------------------------------------------
# Construction des voisins contextuels
# ---------------------------------------------------------------------
def construire_voisins_contextuel(texte, mot_cle, embedding_keyword):
    """
    Extrait les termes fréquents du texte et, pour chacun, calcule la similarité cosinus
    avec l'embedding du mot-clé. Retourne une liste de tuples (terme, similarité, passages),
    le dictionnaire de fréquences et un cache des embeddings.
    """
    top_n = int(entry_nb_termes.get())
    freq_dict = extraire_termes_frequents(texte, top_n)
    candidats = list(freq_dict.keys())
    afficher_message(f"DEBUG: {len(candidats)} candidats fréquents extraits du corpus.")
    voisins = []
    cache = {}
    for idx, t in enumerate(candidats):
        if t.lower() == mot_cle.lower():
            continue
        if t in cache:
            emb_t, passages = cache[t]
        else:
            emb_t, passages = encoder_terme_par_contexte(t, texte)
            cache[t] = (emb_t, passages)
        sim = cosine_similarity(embedding_keyword, emb_t)
        afficher_message(f"Candidat : {t} - Similarité : {sim:.4f}")
        voisins.append((t, sim, passages))
        progress_bar['maximum'] = len(candidats)
        progress_bar['value'] = idx + 1
    voisins = sorted(voisins, key=lambda x: x[1], reverse=True)[:int(entry_voisins.get())]
    return voisins, freq_dict, cache

# ---------------------------------------------------------------------
# Construction du graphe complet via approche k-NN
# ---------------------------------------------------------------------
def construire_graphe_knn(mot_cle, voisins, cache, emb_keyword, k):
    """
    Construit un graphe complet en utilisant une approche k-NN.
    Pour chaque nœud (mot-clé et voisins), on connecte ce nœud à ses k plus proches voisins,
    selon la similarité cosinus calculée à partir des embeddings.
    """
    G = nx.Graph()
    embeddings = {}
    # Ajout du mot-clé
    G.add_node(mot_cle)
    embeddings[mot_cle] = emb_keyword
    # Ajout des voisins
    for mot, sim, passages in voisins:
        G.add_node(mot)
        embeddings[mot] = cache[mot][0]
    nodes = list(G.nodes())
    for node in nodes:
        similarities = []
        for other in nodes:
            if node == other:
                continue
            sim_val = cosine_similarity(embeddings[node], embeddings[other])
            similarities.append((other, sim_val))
        # Tri par similarité décroissante et conserve les k plus proches
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        for i in range(min(k, len(similarities))):
            other, sim_val = similarities[i]
            if not G.has_edge(node, other):
                G.add_edge(node, other, weight=sim_val)
    return G

# ---------------------------------------------------------------------
# Construction du graphe pour l'affichage via Pyvis
# ---------------------------------------------------------------------
def colorer_communautes(G, freq_dict):
    """
    Applique l'algorithme de Louvain pour détecter les communautés et assigne une couleur à chaque nœud.
    """
    partition = community_louvain.best_partition(G)
    palette = ["#8A2BE2", "#9370DB", "#BA55D3", "#DA70D6", "#D8BFD8"]
    for node in G.nodes():
        comm = partition.get(node, 0)
        color = palette[comm % len(palette)]
        G.nodes[node]["frequency"] = freq_dict.get(node, 0)
        G.nodes[node]["color"] = color
    return G, partition

def assigner_layout_classique(G):
    """Positionne les nœuds avec spring_layout et adapte les coordonnées pour Pyvis."""
    positions = nx.spring_layout(G, seed=42)
    positions_dict = {}
    for node, coord in positions.items():
        positions_dict[node] = {"x": float(coord[0]*1000), "y": float(coord[1]*1000)}
    return positions_dict

def nx_vers_pyvis(G, positions):
    """
    Convertit le graphe en un graphique Pyvis.
    Les labels affichent le score de centralité et la taille des nœuds est proportionnelle à ce score.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options('{ "physics": { "enabled": false } }')
    for node, data in G.nodes(data=True):
        centralite = data.get("centralite", 0)
        # Taille proportionnelle à la centralité (multiplication par un facteur pour une meilleure visualisation)
        size = 20 + centralite * 200
        color = data.get("color", "#8A2BE2")
        label = f"{node}\nCent: {centralite:.4f}"
        pos = positions.get(node, {"x":300, "y":300})
        net.add_node(node, label=label, title=label, x=pos["x"], y=pos["y"], color=color, size=size)
    for u, v, data in G.edges(data=True):
        weight = float(data.get("weight", 0))
        net.add_edge(u, v, value=weight, title=f"Sim: {weight:.4f}")
    return net

def sauvegarder_resultats_avec_centralite(voisins, centralite, filename):
    """
    Sauvegarde dans un fichier texte la liste des voisins avec leur similarité (par rapport au mot-clé)
    et leur score de centralité.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for mot, sim, passages in voisins:
            cent_score = centralite.get(mot, 0)
            f.write(f"{mot}\tSim: {sim:.4f}\tCentralité: {cent_score:.4f}\n")
            if passages:
                for passage in passages:
                    f.write(f"    {passage}\n")
            f.write("\n")

# ---------------------------------------------------------------------
# Fonctions de centralité
# ---------------------------------------------------------------------
def calculer_centralite(G, measure):
    """
    Calcule la centralité du graphe G selon la mesure sélectionnée.
    measure peut être "degree", "betweenness", "closeness" ou "eigenvector".
    """
    if measure == "degree":
        return nx.degree_centrality(G)
    elif measure == "betweenness":
        return nx.betweenness_centrality(G)
    elif measure == "closeness":
        return nx.closeness_centrality(G)
    elif measure == "eigenvector":
        return nx.eigenvector_centrality_numpy(G)
    else:
        return nx.degree_centrality(G)

def ajouter_centralite(G, measure):
    """
    Calcule la centralité selon la mesure choisie et l'ajoute aux nœuds.
    Retourne le graphe modifié et le dictionnaire de centralité.
    """
    cent_scores = calculer_centralite(G, measure)
    for node, score in cent_scores.items():
        G.nodes[node]["centralite"] = score
    return G, cent_scores

# ---------------------------------------------------------------------
# Analyse par mot-clé avec graphe complet via k-NN
# ---------------------------------------------------------------------
def analyser_fichier_mot_cle(texte):
    """
    Analyse par mot-clé :
      1. Récupère le mot-clé depuis l'interface.
      2. Calcule l'embedding contextuel du mot-clé.
      3. Extrait les termes fréquents et calcule leur similarité cosinus pour constituer les voisins.
      4. Construit un graphe complet via k-NN (chaque nœud est connecté à ses k plus proches voisins).
      5. Calcule la centralité du graphe selon la mesure choisie.
      6. Affiche le graphe via Pyvis et sauvegarde un fichier HTML et un fichier texte de résultats.
    """
    mot_cle = entry_noeud_central.get().strip().lower()
    afficher_message(f"Analyse du mot-clé : {mot_cle}")

    # Calcul de l'embedding contextuel du mot-clé
    emb_keyword = encoder_contextuel_simplifie(texte, mot_cle)
    afficher_message(f"Embedding du mot-clé (norme) : {np.linalg.norm(emb_keyword):.4f}")

    # Extraction des voisins (termes fréquents et similarité cosinus)
    voisins, freq_dict, cache = construire_voisins_contextuel(texte, mot_cle, emb_keyword)
    nb_voisins = int(entry_voisins.get()) if entry_voisins.get().isdigit() else NB_NEIGHBORS_DEFAULT
    voisins = voisins[:nb_voisins]
    afficher_message(f"{len(voisins)} voisins positifs sélectionnés.")

    # Construction du graphe complet via k-NN
    k = int(entry_knn.get()) if entry_knn.get().isdigit() else 5
    G_complet = construire_graphe_knn(mot_cle, voisins, cache, emb_keyword, k)
    # Ajout des fréquences aux nœuds
    for node in G_complet.nodes():
        G_complet.nodes[node]["frequency"] = freq_dict.get(node, 0)
    # Coloration par communauté
    G_complet, partition = colorer_communautes(G_complet, freq_dict)

    # Calcul de la centralité selon la mesure choisie
    G_complet, centralite = ajouter_centralite(G_complet, centrality_method.get())

    # Identification du nœud avec la centralité la plus élevée
    max_node = max(centralite, key=centralite.get)
    # Attribuer une couleur spéciale (rouge) à ce nœud
    G_complet.nodes[max_node]["color"] = "#FF0000"
    afficher_message(f"Le nœud avec la centralité la plus élevée est : {max_node} (Score: {centralite[max_node]:.4f})")

    # Calcul de la centralité selon la mesure choisie
    G_complet, centralite = ajouter_centralite(G_complet, centrality_method.get())
    positions = assigner_layout_classique(G_complet)
    net = nx_vers_pyvis(G_complet, positions)
    html_file = f"graph_complet_{pooling_method.get()}_{centrality_method.get()}.html"
    net.write_html(html_file)
    afficher_message("Graphe complet généré : " + os.path.abspath(html_file))

    # Sauvegarde d'un fichier texte avec les scores de similarité et de centralité
    results_file = f"resultats_semantiques_{centrality_method.get()}.txt"
    sauvegarder_resultats_avec_centralite(voisins, centralite, results_file)
    afficher_message("Fichier de résultats semantiques généré : " + os.path.abspath("resultats_semantiques.txt"))

# ---------------------------------------------------------------------
# Lanceur principal
# ---------------------------------------------------------------------
def analyser_fichier():
    """Lit le fichier sélectionné, normalise le texte et lance l'analyse par mot-clé."""
    if not selected_filepath:
        afficher_message("Erreur : veuillez sélectionner un fichier.")
        return
    with open(selected_filepath, "r", encoding="utf-8") as f:
        texte_brut = f.read().strip()
    texte_normalise = normaliser_texte(texte_brut)
    analyser_fichier_mot_cle(texte_normalise)

# ---------------------------------------------------------------------
# Interface Tkinter (Analyse par mot-clé avec graphe complet k-NN)
# ---------------------------------------------------------------------
root = tk.Tk()
root.title("Analyse textuelle – Graphe complet k-NN à partir d'un mot-clé")
root.geometry("700x1300")

# Cadre des paramètres de fichier et k-NN
frame_params = ttk.LabelFrame(root, text="Paramètres", padding="10")
frame_params.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
ttk.Button(frame_params, text="Sélectionner un fichier", command=selectionner_fichier).grid(row=0, column=0, columnspan=2, pady=5)
ttk.Label(frame_params, text="Nombre de voisins/termes (max) :").grid(row=1, column=0, sticky=tk.W)
entry_voisins = ttk.Entry(frame_params, width=10)
entry_voisins.insert(0, "20")
entry_voisins.grid(row=1, column=1, sticky=tk.W)
ttk.Label(frame_params, text="Nombre de termes à analyser :").grid(row=2, column=0, sticky=tk.W)
entry_nb_termes = ttk.Entry(frame_params, width=10)
entry_nb_termes.insert(0, "100")
entry_nb_termes.grid(row=2, column=1, sticky=tk.W)
ttk.Label(frame_params, text="Valeur k pour k-NN :").grid(row=3, column=0, sticky=tk.W)
entry_knn = ttk.Entry(frame_params, width=10)
entry_knn.insert(0, "5")
entry_knn.grid(row=3, column=1, sticky=tk.W)

# Cadre des options de prétraitement
frame_pretraitement = ttk.LabelFrame(root, text="Options de prétraitement", padding="10")
frame_pretraitement.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
var_stopwords = tk.BooleanVar(value=True)
var_lemmatisation = tk.BooleanVar(value=True)
ttk.Checkbutton(frame_pretraitement, text="Utiliser stopwords", variable=var_stopwords).grid(row=0, column=0, sticky=tk.W)
ttk.Checkbutton(frame_pretraitement, text="Utiliser lemmatisation", variable=var_lemmatisation).grid(row=0, column=1, sticky=tk.W)

# Cadre des options d'embedding (seulement Mean, Weighted et SIF)
frame_embedding = ttk.LabelFrame(root, text="Méthode d'embedding", padding="10")
frame_embedding.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
pooling_method = tk.StringVar(value="mean")
ttk.Radiobutton(frame_embedding, text="Mean pooling", variable=pooling_method, value="mean").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Weighted pooling (fréquence)", variable=pooling_method, value="weighted").grid(row=0, column=1, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="SIF pooling", variable=pooling_method, value="sif").grid(row=0, column=2, sticky=tk.W)

# Cadre pour la sélection de la mesure de centralité
frame_centralite = ttk.LabelFrame(root, text="Mesure de centralité", padding="10")
frame_centralite.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
centrality_method = tk.StringVar(value="degree")
ttk.Radiobutton(frame_centralite, text="Degré", variable=centrality_method, value="degree").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_centralite, text="Intermédiarité", variable=centrality_method, value="betweenness").grid(row=0, column=1, sticky=tk.W)
ttk.Radiobutton(frame_centralite, text="Proximité", variable=centrality_method, value="closeness").grid(row=0, column=2, sticky=tk.W)
ttk.Radiobutton(frame_centralite, text="Eigenvecteur", variable=centrality_method, value="eigenvector").grid(row=0, column=3, sticky=tk.W)

# Cadre pour définir le mot-clé
frame_analysis = ttk.LabelFrame(root, text="Mot-clé", padding="10")
frame_analysis.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
ttk.Label(frame_analysis, text="Mot-clé :").grid(row=0, column=0, sticky=tk.W)
entry_noeud_central = ttk.Entry(frame_analysis, width=20)
entry_noeud_central.insert(0, "soins")
entry_noeud_central.grid(row=0, column=1, sticky=tk.W)
ttk.Label(frame_analysis, text="(Le mot-clé défini ici sera analysé)").grid(row=0, column=2, sticky=tk.W)

# Bouton de lancement, barre de progression et zone de texte
ttk.Button(root, text="Lancer l'analyse", command=analyser_fichier).grid(row=5, column=0, pady=10)
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
progress_bar.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
text_widget = scrolledtext.ScrolledText(root, width=80, height=20)
text_widget.grid(row=7, column=0, padx=10, pady=10)

root.mainloop()
