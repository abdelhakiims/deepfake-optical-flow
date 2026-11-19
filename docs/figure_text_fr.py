"""French text of the figures (used by make_figures.py --lang fr)."""

FR = {
    "(brown circle: the face; arrows: flow)": "(cercle marron : le visage ; flèches : le flot)",
    "20 face images\n3 x 128 x 128": "20 images du visage\n3 x 128 x 128",
    "20 flow maps\n2 x 128 x 128\n(head motion removed)": "20 cartes de flot\n2 x 128 x 128\n(tête compensée)",
    "6 clips of 21 consecutive frames, only where every scan frame is green, spread over the video, never overlapping":
        "6 extraits de 21 images consécutives, seulement là où toutes les images analysées sont vertes, répartis sur la vidéo, sans chevauchement",
    "Affine fit: the head's rigid motion": "Ajustement affine : le mouvement rigide de la tête",
    "After: the box follows the tracked face": "Après : le cadre suit le visage suivi",
    "At scan 5 the detector missed person A; the track continues because A is found again close to the same place within 3 scans.":
        "À l'analyse 5, le détecteur a raté la personne A ; la trajectoire continue car A est retrouvée près du même endroit en moins de 3 analyses.",
    "Before: one fixed box for the whole clip": "Avant : un cadre fixe pour tout l'extrait",
    "Box = detected face (x 1.0)\nthe jaw-line seam is cut off": "Cadre = visage détecté (x 1,0)\nla couture de la mâchoire est coupée",
    "Box x 1.3 (used)\nthe seam around the face is kept": "Cadre x 1,3 (utilisé)\nla couture autour du visage est gardée",
    "Celeb-DF (unseen dataset)": "Celeb-DF (jeu non vu)",
    "Celeb-DF (unseen)": "Celeb-DF (non vu)",
    "Clips of 21\nconsecutive crops": "Extraits de\n21 images\nconsécutives",
    "Consecutive frames: every slip is visible": "Images consécutives : chaque glissement est visible",
    "Current version: split by person; up to 6 non-overlapping clips per video":
        "Version actuelle : découpage par personne ; jusqu'à 6 extraits sans chevauchement par vidéo",
    "DFD (unseen)": "DFD (non vu)",
    "DFD, unseen actors": "DFD, acteurs non vus",
    "Face tracking\n(YuNet, 8/s)": "Suivi du visage\n(YuNet, 8/s)",
    "FaceForensics++ test": "Test FaceForensics++",
    "First version: windows of 20 flow maps, a new one every 10 maps, split at random":
        "Première version : fenêtres de 20 cartes de flot, une nouvelle toutes les 10 cartes, découpage au hasard",
    "Flow only": "Flot seul",
    "Full model\n(affine removal)": "Modèle complet\n(suppression affine)",
    "Images only": "Images seules",
    "Interview: guest facing the camera,\ninterviewer seen from the side": "Interview : invité face à la caméra,\nintervieweur vu de profil",
    "LSTM\n2 layers, 128": "LSTM\n2 couches, 128",
    "Mean motion\nremoval": "Suppression du\nmouvement moyen",
    "Mean over time\n(no LSTM)": "Moyenne dans le\ntemps (sans LSTM)",
    "Models trained on DFD (7-fold by actor)": "Modèles entraînés sur DFD (7 plis par acteur)",
    "No motion\nremoval": "Sans suppression\ndu mouvement",
    "One frame in 6 (5 per second): the slips disappear": "Une image sur 6 (5 par seconde) : les glissements disparaissent",
    "RAFT optical\nflow (20 maps)": "Flot optique\nRAFT (20 cartes)",
    "Raw flow: head moves, turns, comes closer\n(+ a small mouth movement)": "Flot brut : la tête bouge, tourne, se rapproche\n(+ un petit mouvement de la bouche)",
    "Real / Fake": "Vrai / Faux",
    "Remove head\nmotion (affine)": "Retrait du\nmouvement de\nla tête (affine)",
    "ResNet18\n2-channel first layer\n(from ImageNet filters)": "ResNet18\n1re couche à 2 canaux\n(filtres ImageNet)",
    "ResNet18\nImageNet weights": "ResNet18\npoids ImageNet",
    "ResNet18 flow +\nResNet18 image": "ResNet18 flot +\nResNet18 image",
    "Residual = raw - fit:\nonly the mouth movement is left": "Résidu = brut - ajustement :\nseul le mouvement de la bouche reste",
    "Small face in the background": "Petit visage à l'arrière-plan",
    "Trained on FaceForensics++: the six versions (one seed each)": "Entraînés sur FaceForensics++ : les six versions (une graine chacune)",
    "Two people facing the camera": "Deux personnes face à la caméra",
    "Video": "Vidéo",
    "after the 5-frame average (steady)": "après la moyenne sur 5 images (stable)",
    "ambiguous: frame not used": "ambiguë : image non utilisée",
    "area < 50%": "< 50 %",
    "box centre from the detector (shaky)": "centre du cadre donné par le détecteur (tremblant)",
    "box centre x (pixels)": "centre du cadre x (pixels)",
    "chance": "hasard",
    "dropout 0.3\nLinear 128 -> 2": "dropout 0,3\nLinear 128 -> 2",
    "eye distance / width ~ 0.05": "distance des yeux / largeur ~ 0,05",
    "eye distance / width ~ 0.19 to 0.30": "distance des yeux / largeur ~ 0,19 à 0,30",
    "eye distance / width ~ 0.38": "distance des yeux / largeur ~ 0,38",
    "face slides\nout of the box": "le visage sort\ndu cadre",
    "face stays\ncentred": "le visage reste\ncentré",
    "flow map index in one video. Window 3 (test) shares maps 20-29 with window 2 and maps 30-39 with window 4, both in training.":
        "numéro de carte de flot dans une vidéo. La fenêtre 3 (test) partage les cartes 20-29 avec la fenêtre 2 et 30-39 avec la fenêtre 4, toutes deux à l'entraînement.",
    "flow map t -> t+1: one arrow per point\n(dx = left/right, dy = up/down)": "carte de flot t -> t+1 : une flèche par point\n(dx = gauche/droite, dy = haut/bas)",
    "frame": "image",
    "frame (25 per second)": "image (25 par seconde)",
    "frame number (a 10-second video at 25 frames per second = 250 frames)": "numéro d'image (une vidéo de 10 secondes à 25 images par seconde = 250 images)",
    "frame t": "image t",
    "frame t+1": "image t+1",
    "frames used": "images utilisées",
    "frontal: used": "de face : utilisé",
    "frontality e.g. 0.07": "frontalité 0,07",
    "frontality e.g. 0.4": "frontalité 0,4",
    "green track = person A (large, frontal: the main subject).  orange track = person B (small, seen from the side).":
        "trajectoire verte = personne A (grande, de face : le sujet principal).  trajectoire orange = personne B (petite, de profil).",
    "head movement": "mouvement de la tête",
    "join\n20 x 1,024": "concaténation\n20 x 1 024",
    "last hidden state\n(or mean over the 20 steps)": "dernier état caché\n(ou moyenne sur les 20 étapes)",
    "main subject = the guest": "sujet principal = l'invité",
    "missed": "raté",
    "orange dotted line = where a swapped face is blended in": "pointillés orange = où un visage échangé est fondu",
    "position of a point of the face": "position d'un point du visage",
    "profile: not used": "profil : non utilisé",
    "real / fake": "vrai / faux",
    "real head position": "position réelle de la tête",
    "red bands = frames where\nthe mask slips": "bandes rouges = images où\nle masque glisse",
    "scan frames (about 8 per second): green = main face usable, red = not usable (turned away, second person, ...)":
        "images analysées (environ 8 par seconde) : vert = visage principal utilisable, rouge = inutilisable (détourné, seconde personne, ...)",
    "the head moved right, the mouth opened": "la tête s'est déplacée à droite, la bouche s'est ouverte",
    "three-quarter: used if >= 0.25": "trois quarts : utilisé si >= 0,25",
    "used: the other face is small": "utilisée : l'autre visage est petit",
    "video AUC": "AUC vidéo",
    "video AUC (95% interval)": "AUC vidéo (intervalle à 95 %)",
    "videos of person A: training only": "vidéos de la personne A : entraînement seulement",
    "videos of person B: test only": "vidéos de la personne B : test seulement",
    "who is wearing the mask?": "qui porte le masque ?",
    "training loss": "perte d'entraînement",
    "validation loss": "perte de validation",
    "best epoch": "meilleure époque",
    "epoch": "époque",
    "loss": "perte",
    "Full model, FaceForensics++": "Modèle complet, FaceForensics++",
    "Full model": "Modèle complet",
    "No motion removal": "Sans suppression du mouvement",
    "Mean motion removal": "Suppression du mouvement moyen",
    "Mean over time": "Moyenne dans le temps",
    "validation video AUC": "AUC vidéo de validation",
    "FaceForensics++ validation, six versions": "Validation FaceForensics++, six versions",
    "DFD, the 7 folds": "DFD, les 7 plis",
}

PREFIX = [("clip ", "extrait "), ("fold ", "pli "), ("frame ", "image "), ("scan ", "analyse "),
          ("window ", "fenêtre ")]
WORDS = {": train": " : entraînement", ": test": " : test"}


def translate(text):
    import re
    if not text or not text.strip():
        return text
    if text in FR:
        return FR[text]
    if re.fullmatch(r"[−\-+]?\d+\.\d+%?", text.strip()):
        return text.replace(".", ",")
    out = text
    for a, b in PREFIX:
        if out.startswith(a):
            out = b + out[len(a):]
            break
    for a, b in WORDS.items():
        out = out.replace(a, b)
    return out
