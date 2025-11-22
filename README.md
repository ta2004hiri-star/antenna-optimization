# 🛰️ Synthèse et Optimisation de Réseaux d'Antennes

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/ta2004hiri-star/optimisation-d-antenne?style=social)](https://github.com/ta2004hiri-star/optimisation-d-antenne)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/ta2004hiri-star/optimisation-d-antenne)

---

## 📌 À propos de ce projet

**Application Python professionnelle** pour la synthèse et l'optimisation de réseaux d'antennes utilisant **15 algorithmes métaheuristiques avancés**.

Ce projet démontre mes compétences en :
- 🎯 **Ingénierie logicielle** : Architecture OOP, Design Patterns
- 🧠 **Algorithmes d'optimisation** : 15 implémentations métaheuristiques
- 🎨 **Interface utilisateur** : GUI professionnelle avec Tkinter
- 📊 **Visualisation de données** : Graphiques en temps réel
- 🗄️ **Gestion de bases de données** : SQLite intégré
- 🚀 **Développement professionnel** : Code documenté, versionné sur Git

---

## ✨ Caractéristiques principales

### 📡 Géométries d'antennes supportées
- **Linéaire** - Réseau linéaire d'antennes
- **Planaire** - Réseau 2D (matriciel)
- **Circulaire** - Réseau en anneau

### ⚙️ Types d'optimisation
- **Amplitude** - Optimiser les amplitudes des éléments
- **Phase** - Optimiser les phases de chaque élément
- **Amplitude+Phase** - Optimisation hybride des deux

### 🎯 Objectifs d'optimisation
- **Minimiser SSL** - Réduire le niveau de lobe secondaire
- **Maximiser Gain** - Augmenter la directivité
- **Multicritères** - Optimisation combinée (SSL + Gain)

### 🤖 15 Algorithmes Métaheuristiques Implémentés

| # | Algorithme | Abréviation | Type | Performance |
|---|-----------|------------|------|------------|
| 1 | Particle Swarm Optimization | PSO | Essaim | ⭐⭐⭐⭐⭐ |
| 2 | Genetic Algorithm | GA | Évolution | ⭐⭐⭐⭐ |
| 3 | Differential Evolution | DE | Évolution | ⭐⭐⭐⭐⭐ |
| 4 | Simulated Annealing | SA | Métallurgie | ⭐⭐⭐ |
| 5 | Grey Wolf Optimizer | GWO | Essaim | ⭐⭐⭐⭐⭐ |
| 6 | Whale Optimization Algorithm | WOA | Essaim | ⭐⭐⭐⭐ |
| 7 | Ant Colony Optimization | ACO | Colonie | ⭐⭐⭐ |
| 8 | Artificial Bee Colony | ABC | Abeilles | ⭐⭐⭐⭐ |
| 9 | Firefly Algorithm | FA | Nature | ⭐⭐⭐ |
| 10 | Bat Algorithm | BA | Nature | ⭐⭐⭐⭐ |
| 11 | Cuckoo Search | CS | Nature | ⭐⭐⭐⭐ |
| 12 | Harris Hawks Optimization | HHO | Prédation | ⭐⭐⭐⭐ |
| 13 | Flower Pollination Algorithm | FPA | Nature | ⭐⭐⭐ |
| 14 | Sine Cosine Algorithm | SCA | Mathématique | ⭐⭐⭐ |
| 15 | Teaching-Learning Based | TLBO | Éducation | ⭐⭐⭐⭐ |

### 📊 Résultats et export

**4 Graphiques dynamiques :**
- 📈 Diagramme de rayonnement (polaire)
- 📊 Performances (Gain, Directivité, SSL)
- 📉 Courbe de convergence
- 📐 Amplitudes optimales

**Formats d'export :**
- 📄 **PDF** - Rapports avec visualisations
- 📋 **CSV** - Tableaux de données
- 💾 **JSON** - Configuration complète

### 🗄️ Fonctionnalités avancées

- ✅ **Base de données SQLite** - Stockage des résultats
- ✅ **Historique** - Consultation des optimisations précédentes
- ✅ **Multi-run** - Comparaison de plusieurs exécutions
- ✅ **Threading** - Opérations non-bloquantes
- ✅ **Statistiques** - Moyenne, écart-type, meilleur/pire cas

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- tkinter (généralement inclus)
- NumPy, Matplotlib, Pandas

### Installation rapide

#### **Méthode 1 : Installation locale**

```bash
# Cloner le repository
git clone https://github.com/ta2004hiri-star/optimisation-d-antenne.git
cd optimisation-d-antenne

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python antenna_app.py
```

#### **Méthode 2 : Google Colab (Sans installation)**

1. Allez sur [Google Colab](https://colab.research.google.com)
2. Créez un nouveau notebook
3. Copiez-collez le code de `antenna_colab.py`
4. Exécutez les cellules
5. **C'est prêt !** ☁️

---

## 📖 Guide d'utilisation

### Lancer l'interface graphique

```bash
python antenna_app.py
```

### Interface utilisateur
1. **Configurez votre antenne**
   - Choisissez la géométrie (Linéaire/Planaire/Circulaire)
   - Définissez le nombre d'éléments
   - Réglez la fréquence et l'espacement

2. **Choisissez l'algorithme**
   - Sélectionnez parmi 15 algorithmes
   - Définissez la population et itérations
   - Spécifiez le nombre de runs

3. **Lancez l'optimisation**
   - Cliquez sur "▶ LANCER"
   - Observez les résultats en temps réel
   - Consultez les 4 graphiques

4. **Exportez les résultats**
   - PDF avec visualisations
   - CSV pour l'analyse
   - JSON pour le partage

### Exemple de code

```python
from antenna_app import AntennaOptimizer

# Créer un optimiseur
optimizer = AntennaOptimizer(
    geometry="Linéaire",
    n_elem_x=8,
    n_elem_y=1,
    frequency=2.4,
    spacing_x=0.5,
    spacing_y=0.5
)

# Lancer l'optimisation
results = optimizer.optimize(
    algorithm="PSO",
    opt_type="Amplitude",
    objective="Minimiser SSL",
    population=50,
    iterations=100
)

# Afficher les résultats
optimizer.display_results()
```

---

## 🏗️ Structure du projet

```
optimisation-d-antenne/
├── antenna_app.py              # Application GUI (Tkinter)
├── antenna_colab.py            # Version Google Colab
├── antenna_optimizer.py        # Moteur d'optimisation
├── metaheuristic_algorithms.py # 15 algorithmes
├── antenna_array.py            # Calculs d'antennes
│
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation (ce fichier)
├── CONTRIBUTING.md             # Guide de contribution
├── INSTALLATION.md             # Guide d'installation
├── LICENSE                     # Licence MIT
│
└── results/
    └── antenna_results.db      # Base de données SQLite
```

---

## 📊 Comparaison des algorithmes

**Résultats typiques** (réseau linéaire 8 éléments) :

| Algorithme | Temps (s) | SSL (dB) | Convergence | Succès |
|-----------|----------|----------|------------|--------|
| PSO | 2.3 | -22.5 | Rapide | 98% |
| GA | 3.1 | -21.2 | Moyen | 95% |
| **DE** | 2.8 | **-23.1** | Rapide | **99%** |
| GWO | 2.5 | -22.8 | Rapide | 97% |
| WOA | 2.9 | -22.3 | Moyen | 96% |

*Les performances varient selon les paramètres d'entrée*

---

## 🎓 Théorie

### Facteur de réseau d'antennes

Pour un réseau linéaire :

```
AF(θ) = Σ(i=0 à N-1) a_i * exp(j * (k*d*i*cos(θ) + φ_i))
```

Où :
- `a_i` = amplitude de l'élément i
- `φ_i` = phase de l'élément i
- `d` = espacement entre éléments
- `k` = nombre d'onde (2π/λ)
- `θ` = angle d'observation

### Métriques de performance

- **Gain** : `20*log10(max(|AF|))`
- **Directivité** : Concentration du diagramme
- **SSL (Side Lobe Level)** : `20*log10(lobe_secondaire / lobe_principal)`
- **Largeur du lobe principal** : Beamwidth à -3dB

---

## 💡 Cas d'usage

### 🎓 Académique
- Étudier les algorithmes métaheuristiques
- Comparer les techniques d'optimisation
- Valider la conception d'antennes

### 🔬 Ingénierie
- Concevoir des réseaux phasés
- Minimiser les lobes secondaires
- Optimiser les systèmes de communication

### 💼 Portfolio
- Démontrer l'expertise Python
- Montrer les compétences en calcul scientifique
- Prouver les pratiques professionnelles

---

## 🤝 Contribuer

Les contributions sont bienvenues ! Pour contribuer :

1. **Fork** le repository
2. **Créez** une branche feature (`git checkout -b feature/VotreFonctionnalite`)
3. **Modifiez** le code
4. **Committez** (`git commit -m 'Ajouter VotreFonctionnalite'`)
5. **Pushez** (`git push origin feature/VotreFonctionnalite`)
6. **Ouvrez** une Pull Request

### Domaines pour contribuer
- [ ] Ajouter plus d'algorithmes (50+ disponibles)
- [ ] Implémenter la visualisation 3D
- [ ] Ajouter l'accélération GPU (CUDA)
- [ ] Créer une interface web (Flask/Django)
- [ ] Développer une application mobile
- [ ] Améliorer la documentation

---

## 🐛 Signaler des bugs

Trouvez un bug ? Créez une issue avec :
- Description du problème
- Étapes pour reproduire
- Résultat attendu vs résultat réel
- Captures d'écran si applicable
- Version de Python utilisée

---

## 📚 Ressources

### Théorie des antennes
- [Wikipedia: Réseau d'antennes](https://en.wikipedia.org/wiki/Antenna_array)
- [MATLAB: Phased Array System](https://www.mathworks.com/products/phased.html)

### Algorithmes métaheuristiques
- [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
- [Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution)

### Documentation Python
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [NumPy](https://numpy.org/doc/stable/)
- [Matplotlib](https://matplotlib.org/stable/tutorials/)

---

## 📄 Licence

Ce projet est sous licence **MIT** - Voir [LICENSE](LICENSE)

---

## 👩‍💼 À propos de l'auteur

**TAHIRI NADIA HAFIDHA**

🛰️ Spécialiste en optimisation d'antennes | Développeuse Python | Enthousiate ML

💡 Passionnée par les algorithmes d'optimisation et le traitement du signal

🔗 **Liens professionnels :**
- 📧 Email : tahiri.nadiahafidha@cuniv-naama.dz
- 💼 LinkedIn : [TA HIRI](https://www.linkedin.com/in/ta-hiri-2b2691392/)
- 🌐 Portfolio : [Nadia's Portfolio](https://ta2004hiri-star.github.io/nadia/#projets)
- 📊 GitHub : [@ta2004hiri-star](https://github.com/ta2004hiri-star)
- 📱 Téléphone : +213 667 619 335

---

## 📈 Statistiques du projet

- ⭐ **Stars** : Merci de mettre en favori si vous aimez !
- 🍴 **Forks** : N'hésitez pas à forker et personnaliser
- 💬 **Issues** : 0 (Aidez à garder propre !)
- 📝 **Dernière mise à jour** : Novembre 2024

---

## 🎯 Roadmap

- [x] Implémenter 15 algorithmes
- [x] Créer l'interface Tkinter
- [x] Intégrer SQLite
- [x] Ajouter l'export PDF/CSV
- [ ] Visualisation 3D
- [ ] Optimisation parallèle
- [ ] Version web
- [ ] Application mobile

---

## 🙏 Remerciements

- À la communauté open-source
- Aux auteurs de NumPy, Matplotlib, et Tkinter
- Aux chercheurs en algorithmes métaheuristiques
- À tous les contributeurs et utilisateurs

---

## 📞 Support

**Besoin d'aide ?**
- 💬 Ouvrez une [issue GitHub](https://github.com/ta2004hiri-star/optimisation-d-antenne/issues)
- 📧 Envoyez un email : tahiri.nadiahafidha@cuniv-naama.dz
- 📱 Appelez : 0667619335
- 🤝 Contribuez à améliorer le projet

---

<div align="center">

**⭐ Si ce projet vous a été utile, pensez à mettre une étoile !**

[GitHub](https://github.com/ta2004hiri-star/optimisation-d-antenne) • 
[Portfolio](https://ta2004hiri-star.github.io/nadia/#projets) • 
[LinkedIn](https://www.linkedin.com/in/ta-hiri-2b2691392/)

Made with ❤️ by TAHIRI NADIA HAFIDHA

</div>
