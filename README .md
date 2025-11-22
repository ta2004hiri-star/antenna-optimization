# 🛰️ Synthèse et Optimisation de Réseaux d'Antennes

## 📋 Description

Application complète pour la **synthèse et optimisation de réseaux d'antennes** utilisant des **algorithmes métaheuristiques avancés**.

Cette application permet d'optimiser les paramètres d'un réseau d'antennes (amplitudes, phases, positions) pour minimiser le niveau de lobe secondaire (SSL), maximiser le gain, ou atteindre des objectifs multicritères.

## ✨ Caractéristiques

### 📡 Géométries supportées :
- **Linéaire** - Réseau linéaire d'antennes
- **Planaire** - Réseau 2D
- **Circulaire** - Réseau en anneau

### ⚙️ Types d'optimisation :
- **Amplitude** - Optimiser les amplitudes
- **Phase** - Optimiser les phases
- **Amplitude+Phase** - Optimiser les deux

### 🎯 Objectifs :
- **Minimiser SSL** - Réduire le niveau de lobe secondaire
- **Maximiser Gain** - Augmenter le gain
- **Multicritères** - Combiner SSL + Gain

### 🧠 Algorithmes (15 implémentés) :

| Algorithme | Code | Type |
|-----------|------|------|
| Particle Swarm Optimization | PSO | Essaim |
| Ant Colony Optimization | ACO | Colonie |
| Artificial Bee Colony | ABC | Abeilles |
| Genetic Algorithm | GA | Évolution |
| Differential Evolution | DE | Évolution |
| Simulated Annealing | SA | Métallurgie |
| Firefly Algorithm | FA | Lucioles |
| Bat Algorithm | BA | Chauve-souris |
| Cuckoo Search | CS | Coucou |
| Grey Wolf Optimizer | GWO | Loups |
| Harris Hawks Optimization | HHO | Faucons |
| Whale Optimization Algorithm | WOA | Baleines |
| Flower Pollination Algorithm | FPA | Pollinisation |
| Sine Cosine Algorithm | SCA | Trigonométrie |
| Teaching-Learning Based Optimization | TLBO | Éducation |

## 🚀 Installation

### Prérequis :
```bash
pip install numpy matplotlib pandas ipywidgets
```

## 📱 Utilisation

### Via Google Colab (Recommandé - Gratuit!)

1. Allez sur [Google Colab](https://colab.research.google.com)
2. Ouvrez un nouveau notebook
3. Copiez-collez le code de `antenna_colab.py`
4. Exécutez les cellules

```python
# Exemple basique
optimizer = AntennaOptimizer(
    geometry="Linéaire",
    n_elem_x=8,
    n_elem_y=1,
    frequency=2.4,
    spacing_x=0.5,
    spacing_y=0.5
)

results = optimizer.optimize(
    algorithm="PSO",
    opt_type="Amplitude",
    objective="Minimiser SSL",
    population=50,
    iterations=100
)

optimizer.display_results()
```

### Localement (Python)

```bash
python antenna_colab.py
```

## 📊 Résultats

L'application affiche 4 graphiques :

1. **Pattern de Rayonnement** - Diagramme polaire du champ rayonné
2. **Performances** - Gain, Directivité, SSL en dB
3. **Convergence** - Évolution du fitness au fil des itérations
4. **Amplitudes Optimales** - Amplitudes finales de chaque élément

## 📈 Métriques de Performance

- **Gain** - Amplification du signal (dB)
- **Directivité** - Concentration du rayonnement
- **SSL** - Niveau de Lobe Secondaire (dB)
- **Lobe Principal** - Largeur du lobe principal (°)

## 🔧 Paramètres Configurables

| Paramètre | Plage | Défaut |
|-----------|-------|--------|
| Nombre d'éléments (X) | 2-20 | 8 |
| Nombre d'éléments (Y) | 1-20 | 1 |
| Fréquence (GHz) | 0.1-10 | 2.4 |
| Espacement (λ) | 0.1-2.0 | 0.5 |
| Population | 10-200 | 50 |
| Itérations | 10-500 | 100 |
| Nombre de Runs | 1-20 | 1 |

## 📚 Théorie

### Réseau d'Antennes Linéaire

Le facteur de réseau pour un réseau linéaire est :

```
AF(θ) = Σ(i=0 to N-1) a_i * exp(j * (k*d*i*cos(θ) + φ_i))
```

Où :
- `a_i` : amplitude de l'élément i
- `φ_i` : phase de l'élément i
- `d` : espacement entre éléments
- `k` : nombre d'onde (2π/λ)
- `θ` : angle d'observation

### Optimisation Multicritères

Fonction objective combinée :
```
f(x) = minimiser(SSL) + w * maximiser(Gain)
```

## 📄 Fichiers

- `antenna_colab.py` - Code principal
- `README.md` - Documentation (ce fichier)
- `requirements.txt` - Dépendances Python

## 🎓 Cas d'Usage

✅ **Recherche académique** - Étudier les algorithmes d'optimisation
✅ **Conception d'antennes** - Optimiser les réseaux réels
✅ **Portfolio professionnel** - Montrer vos compétences
✅ **Prototypage** - Tester rapidement des configurations

## 📊 Comparaison des Algorithmes

Pour comparer les algorithmes :

```python
algorithms = ["PSO", "GA", "DE", "GWO", "WOA"]
results = {}

for algo in algorithms:
    print(f"Test {algo}...")
    res = optimizer.optimize(algorithm=algo, iterations=100)
    results[algo] = res['metrics']['ssl']

# Afficher le meilleur
print("Meilleur algorithme:", min(results, key=results.get))
```

## 🔬 Améliorations Futures

- [ ] Support GPU pour calculs parallèles
- [ ] Export PDF avec rapport complet
- [ ] Visualisation 3D des réseaux
- [ ] Base de données d'historique
- [ ] Interface web interactive
- [ ] Plus d'algorithmes (50+)
- [ ] Support des éléments non-isotropes

## 📝 Licence

MIT License - Libre d'utilisation

## 👨‍💼 À propos

Développé comme projet de recherche doctorale en synthèse d'antennes par algorithmes métaheuristiques.

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📧 Contact

Pour les questions ou suggestions, ouvrez une issue sur GitHub.

## 🙏 Remerciements

- NumPy et Matplotlib pour les calculs et visualisations
- Google Colab pour l'environnement gratuit
- Les auteurs des algorithmes implémentés

---

**⭐ Si ce projet vous a été utile, mettez-le en favori sur GitHub !**
