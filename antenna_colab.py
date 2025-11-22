"""
Synthèse et Optimisation de Réseaux d'Antennes
Utilisant des Algorithmes Métaheuristiques

Auteur: [Votre Nom]
Date: 2024
Description: Application complète pour optimiser les réseaux d'antennes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import widgets, VBox, HBox, Output
from IPython.display import display, clear_output, HTML

# ==================== CLASSE ANTENNE ====================
class AntennaArray:
    """Classe pour calculer les propriétés d'un réseau d'antennes"""
    
    def __init__(self, geometry, n_elem_x, n_elem_y, frequency, spacing_x, spacing_y):
        self.geometry = geometry
        self.n_elem_x = n_elem_x
        self.n_elem_y = n_elem_y
        self.frequency = frequency
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.wavelength = 3e8 / (frequency * 1e9)
    
    def calculate_pattern(self, amplitudes, phases):
        """Calculer le pattern de rayonnement"""
        theta = np.linspace(0, 2*np.pi, 360)
        pattern = np.zeros_like(theta, dtype=complex)
        
        if self.geometry == "Linéaire":
            for i in range(self.n_elem_x):
                x_pos = i * self.spacing_x * self.wavelength
                phase_shift = 2 * np.pi * x_pos * np.cos(theta) / self.wavelength
                pattern += amplitudes[i] * np.exp(1j * (phases[i] + phase_shift))
        else:
            for i in range(self.n_elem_x):
                for j in range(self.n_elem_y):
                    idx = i * self.n_elem_y + j
                    if idx < len(amplitudes):
                        x_pos = i * self.spacing_x * self.wavelength
                        phase_shift = 2*np.pi*x_pos*np.cos(theta)/self.wavelength
                        pattern += amplitudes[idx] * np.exp(1j * (phases[idx] + phase_shift))
        return theta, np.abs(pattern)
    
    def calculate_metrics(self, amplitudes, phases):
        """Calculer les métriques de performance"""
        theta, pattern = self.calculate_pattern(amplitudes, phases)
        gain = 20 * np.log10(np.max(pattern) + 1e-10)
        main_lobe_width = 2 * np.arccos(0.5) * 180/np.pi
        sorted_pattern = np.sort(pattern)[::-1]
        ssl = 20 * np.log10(sorted_pattern[len(sorted_pattern)//2] / sorted_pattern[0] + 1e-10)
        directivity = 10 * np.log10(sorted_pattern[0] ** 2 + 1e-10)
        return {'gain': gain, 'directivity': directivity, 'ssl': ssl, 'main_lobe_width': main_lobe_width, 'pattern': pattern, 'theta': theta}

# ==================== ALGORITHMES MÉTAHEURISTIQUES ====================
class MetaHeuristicAlgorithms:
    """Implémentation de 15 algorithmes métaheuristiques"""
    
    @staticmethod
    def PSO(obj_func, n_vars, bounds, pop=50, iters=100):
        """Particle Swarm Optimization"""
        lb, ub = bounds
        particles = np.random.uniform(lb, ub, (pop, n_vars))
        velocities = np.random.uniform(-1, 1, (pop, n_vars))
        fitness = np.array([obj_func(p) for p in particles])
        best_particle = particles[np.argmin(fitness)].copy()
        best_fitness = np.min(fitness)
        history = []
        for iteration in range(iters):
            w = 0.9 - (iteration / iters) * 0.5
            for i in range(pop):
                r1, r2 = np.random.random(n_vars), np.random.random(n_vars)
                velocities[i] = w*velocities[i] + 2*r1*(particles[i]-best_particle) + 2*r2*(best_particle-particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                new_fit = obj_func(particles[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    if new_fit < best_fitness:
                        best_fitness, best_particle = new_fit, particles[i].copy()
            history.append(best_fitness)
        return best_particle, best_fitness, history
    
    @staticmethod
    def GA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Genetic Algorithm"""
        lb, ub = bounds
        population = np.random.uniform(lb, ub, (pop, n_vars))
        history = []
        for generation in range(iters):
            fitness = np.array([obj_func(ind) for ind in population])
            elite_idx = np.argsort(fitness)[:pop//2]
            new_pop = population[elite_idx].copy()
            for _ in range(pop//2):
                p1, p2 = np.random.choice(elite_idx, 2, replace=False)
                child = (population[p1] + population[p2]) / 2 + np.random.normal(0, (ub-lb)*0.01, n_vars)
                new_pop = np.vstack([new_pop, np.clip(child, lb, ub)])
            population = new_pop[:pop]
            history.append(np.min([obj_func(ind) for ind in population]))
        best_idx = np.argmin([obj_func(ind) for ind in population])
        return population[best_idx], obj_func(population[best_idx]), history
    
    @staticmethod
    def DE(obj_func, n_vars, bounds, pop=50, iters=100):
        """Differential Evolution"""
        lb, ub = bounds
        population = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(ind) for ind in population])
        history = []
        for iteration in range(iters):
            for i in range(pop):
                a, b, c = np.random.choice(pop, 3, replace=False)
                mutant = np.clip(population[a] + 0.8 * (population[b] - population[c]), lb, ub)
                mut_fit = obj_func(mutant)
                if mut_fit < fitness[i]:
                    population[i], fitness[i] = mutant, mut_fit
            history.append(np.min(fitness))
        return population[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def SA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Simulated Annealing"""
        lb, ub = bounds
        current = np.random.uniform(lb, ub, n_vars)
        current_fit = obj_func(current)
        best, best_fit = current.copy(), current_fit
        history = []
        for iteration in range(iters):
            temp = 1.0 * (1 - iteration / iters)
            neighbor = np.clip(current + np.random.normal(0, (ub-lb)*0.1, n_vars), lb, ub)
            neighbor_fit = obj_func(neighbor)
            if neighbor_fit < current_fit or np.random.random() < np.exp(-(neighbor_fit-current_fit)/(temp+1e-10)):
                current, current_fit = neighbor, neighbor_fit
            if current_fit < best_fit:
                best, best_fit = current.copy(), current_fit
            history.append(best_fit)
        return best, best_fit, history
    
    @staticmethod
    def GWO(obj_func, n_vars, bounds, pop=50, iters=100):
        """Grey Wolf Optimizer"""
        lb, ub = bounds
        wolves = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(w) for w in wolves])
        history = []
        for iteration in range(iters):
            alpha_idx = np.argmin(fitness)
            for i in range(pop):
                A = 2 * np.random.random(n_vars) - 1
                C = 2 * np.random.random(n_vars)
                D = np.abs(C * wolves[alpha_idx] - wolves[i])
                wolves[i] = np.clip(wolves[alpha_idx] - A * D, lb, ub)
                fitness[i] = obj_func(wolves[i])
            history.append(np.min(fitness))
        return wolves[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def WOA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Whale Optimization Algorithm"""
        lb, ub = bounds
        whales = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(w) for w in whales])
        history = []
        best_idx = np.argmin(fitness)
        for iteration in range(iters):
            for i in range(pop):
                A = 2 * np.random.random(n_vars) - 1
                C = 2 * np.random.random(n_vars)
                D = np.abs(C * whales[best_idx] - whales[i])
                whales[i] = np.clip(whales[best_idx] - A * D, lb, ub)
                new_fit = obj_func(whales[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
            best_idx = np.argmin(fitness)
            history.append(fitness[best_idx])
        return whales[best_idx], fitness[best_idx], history
    
    @staticmethod
    def ACO(obj_func, n_vars, bounds, pop=50, iters=100):
        """Ant Colony Optimization"""
        lb, ub = bounds
        history = []
        best_fit = float('inf')
        best_sol = None
        for iteration in range(iters):
            ants = np.random.uniform(lb, ub, (pop, n_vars))
            fits = np.array([obj_func(ant) for ant in ants])
            best_ant_idx = np.argmin(fits)
            if fits[best_ant_idx] < best_fit:
                best_fit = fits[best_ant_idx]
                best_sol = ants[best_ant_idx].copy()
            history.append(best_fit)
        return best_sol, best_fit, history
    
    @staticmethod
    def ABC(obj_func, n_vars, bounds, pop=50, iters=100):
        """Artificial Bee Colony"""
        lb, ub = bounds
        food = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(f) for f in food])
        history = []
        for iteration in range(iters):
            for i in range(pop):
                k = np.random.randint(pop)
                candidate = food[i] + np.random.uniform(-1, 1, n_vars) * (food[i] - food[k])
                candidate = np.clip(candidate, lb, ub)
                cand_fit = obj_func(candidate)
                if cand_fit < fitness[i]:
                    food[i], fitness[i] = candidate, cand_fit
            history.append(np.min(fitness))
        return food[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def FA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Firefly Algorithm"""
        lb, ub = bounds
        fireflies = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(f) for f in fireflies])
        history = []
        for iteration in range(iters):
            for i in range(pop):
                for j in range(pop):
                    if fitness[j] < fitness[i]:
                        fireflies[i] += 0.5 * (fireflies[j] - fireflies[i])
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        fitness[i] = obj_func(fireflies[i])
            history.append(np.min(fitness))
        return fireflies[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def BA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Bat Algorithm"""
        lb, ub = bounds
        bats = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(b) for b in bats])
        best_bat = bats[np.argmin(fitness)].copy()
        best_fit = np.min(fitness)
        history = []
        for iteration in range(iters):
            for i in range(pop):
                bats[i] = bats[i] + 0.1 * (best_bat - bats[i])
                bats[i] = np.clip(bats[i], lb, ub)
                new_fit = obj_func(bats[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best_bat, best_fit = bats[i].copy(), new_fit
            history.append(best_fit)
        return best_bat, best_fit, history
    
    @staticmethod
    def CS(obj_func, n_vars, bounds, pop=50, iters=100):
        """Cuckoo Search"""
        lb, ub = bounds
        nests = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(n) for n in nests])
        history = []
        for iteration in range(iters):
            for i in range(pop):
                new_nest = nests[i] + 0.01 * np.random.randn(n_vars)
                new_nest = np.clip(new_nest, lb, ub)
                new_fit = obj_func(new_nest)
                if new_fit < fitness[i]:
                    nests[i], fitness[i] = new_nest, new_fit
            history.append(np.min(fitness))
        return nests[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def HHO(obj_func, n_vars, bounds, pop=50, iters=100):
        """Harris Hawks Optimization"""
        lb, ub = bounds
        hawks = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(h) for h in hawks])
        history = []
        for iteration in range(iters):
            for i in range(pop):
                hawks[i] = hawks[i] + 0.1 * np.random.random(n_vars)
                hawks[i] = np.clip(hawks[i], lb, ub)
                fitness[i] = obj_func(hawks[i])
            history.append(np.min(fitness))
        return hawks[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def FPA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Flower Pollination Algorithm"""
        lb, ub = bounds
        flowers = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(f) for f in flowers])
        history = []
        for iteration in range(iters):
            best_idx = np.argmin(fitness)
            for i in range(pop):
                flowers[i] = flowers[i] + 0.1 * (flowers[best_idx] - flowers[i])
                flowers[i] = np.clip(flowers[i], lb, ub)
                new_fit = obj_func(flowers[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
            history.append(np.min(fitness))
        return flowers[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def SCA(obj_func, n_vars, bounds, pop=50, iters=100):
        """Sine Cosine Algorithm"""
        lb, ub = bounds
        population = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(x) for x in population])
        history = []
        for iteration in range(iters):
            best_idx = np.argmin(fitness)
            for i in range(pop):
                population[i] = population[best_idx] + np.sin(np.random.random()) * 0.1
                population[i] = np.clip(population[i], lb, ub)
                new_fit = obj_func(population[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
            history.append(np.min(fitness))
        return population[np.argmin(fitness)], np.min(fitness), history
    
    @staticmethod
    def TLBO(obj_func, n_vars, bounds, pop=50, iters=100):
        """Teaching-Learning Based Optimization"""
        lb, ub = bounds
        population = np.random.uniform(lb, ub, (pop, n_vars))
        fitness = np.array([obj_func(x) for x in population])
        history = []
        for iteration in range(iters):
            teacher_idx = np.argmin(fitness)
            for i in range(pop):
                population[i] = population[i] + 0.1 * (population[teacher_idx] - population[i])
                population[i] = np.clip(population[i], lb, ub)
                new_fit = obj_func(population[i])
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
            history.append(np.min(fitness))
        return population[np.argmin(fitness)], np.min(fitness), history

# ==================== CLASSE OPTIMISATION ====================
class AntennaOptimizer:
    """Classe principale pour l'optimisation"""
    
    def __init__(self, geometry, n_elem_x, n_elem_y, frequency, spacing_x, spacing_y):
        self.antenna = AntennaArray(geometry, n_elem_x, n_elem_y, frequency, spacing_x, spacing_y)
        self.n_total = n_elem_x * n_elem_y
        self.results = None
    
    def optimize(self, algorithm="PSO", opt_type="Amplitude", objective="Minimiser SSL", population=50, iterations=100, amp_min=0, amp_max=1):
        """Lancer l'optimisation"""
        n_vars = self.n_total if opt_type == "Amplitude" else self.n_total * 2
        
        def objective_func(x):
            if opt_type == "Amplitude":
                amplitudes = np.clip(x, amp_min, amp_max)
                phases = np.zeros(self.n_total)
            else:
                amplitudes = np.clip(x[:self.n_total], amp_min, amp_max)
                phases = x[self.n_total:]
            metrics = self.antenna.calculate_metrics(amplitudes, phases)
            if objective == "Minimiser SSL":
                return -metrics['ssl']
            elif objective == "Maximiser Gain":
                return -metrics['gain']
            else:
                return -metrics['ssl'] + 0.5 * (metrics['gain'] - 20)
        
        bounds = (np.zeros(n_vars), np.ones(n_vars))
        algo_methods = {
            "PSO": MetaHeuristicAlgorithms.PSO, "ACO": MetaHeuristicAlgorithms.ACO,
            "ABC": MetaHeuristicAlgorithms.ABC, "GA": MetaHeuristicAlgorithms.GA,
            "DE": MetaHeuristicAlgorithms.DE, "SA": MetaHeuristicAlgorithms.SA,
            "FA": MetaHeuristicAlgorithms.FA, "BA": MetaHeuristicAlgorithms.BA,
            "CS": MetaHeuristicAlgorithms.CS, "GWO": MetaHeuristicAlgorithms.GWO,
            "HHO": MetaHeuristicAlgorithms.HHO, "WOA": MetaHeuristicAlgorithms.WOA,
            "FPA": MetaHeuristicAlgorithms.FPA, "SCA": MetaHeuristicAlgorithms.SCA,
            "TLBO": MetaHeuristicAlgorithms.TLBO
        }
        
        if algorithm not in algo_methods:
            return None
        algo_func = algo_methods[algorithm]
        best, fitness, history = algo_func(objective_func, n_vars, bounds, population, iterations)
        amplitudes = np.clip(best[:self.n_total] if opt_type != "Amplitude" else best, amp_min, amp_max)
        phases = best[self.n_total:] if opt_type != "Amplitude" else np.zeros(self.n_total)
        metrics = self.antenna.calculate_metrics(amplitudes, phases)
        self.results = {
            'algorithm': algorithm, 'opt_type': opt_type, 'objective': objective,
            'amplitudes': amplitudes, 'phases': phases, 'metrics': metrics,
            'fitness_history': history
        }
        return self.results
    
    def display_results(self):
        """Afficher les résultats"""
        if not self.results:
            return
        metrics = self.results['metrics']
        print(f"\n✅ OPTIMISATION RÉUSSIE!")
        print(f"Algorithme: {self.results['algorithm']} | Type: {self.results['opt_type']} | Objectif: {self.results['objective']}")
        print(f"Gain: {metrics['gain']:.2f} dB | Directivité: {metrics['directivity']:.2f} dB | SSL: {metrics['ssl']:.2f} dB")
        
        fig = plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        ax1.plot(metrics['theta'], metrics['pattern'], 'b-', linewidth=2)
        ax1.set_title("Pattern")
        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(['Gain', 'Dir', 'SSL'], [metrics['gain'], metrics['directivity'], metrics['ssl']], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title("Performances")
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(self.results['fitness_history'], 'g-', linewidth=2)
        ax3.set_title("Convergence")
        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(range(len(self.results['amplitudes'])), self.results['amplitudes'], color='#9B59B6')
        ax4.set_title("Amplitudes")
        plt.tight_layout()
        plt.show()

# ==================== INTERFACE ====================
display(HTML("<h1 style='color:blue'>🛰️ SYNTHÈSE D'ANTENNES</h1>"))

geometry = widgets.Dropdown(options=['Linéaire', 'Planaire', 'Circulaire'], value='Linéaire', description='Géométrie:', style={'description_width': '120px'})
n_elem_x = widgets.IntSlider(value=8, min=2, max=20, description='Éléments X:', style={'description_width': '120px'})
n_elem_y = widgets.IntSlider(value=1, min=1, max=20, description='Éléments Y:', style={'description_width': '120px'})
frequency = widgets.FloatSlider(value=2.4, min=0.1, max=10, description='Fréquence:', style={'description_width': '120px'})
spacing_x = widgets.FloatSlider(value=0.5, min=0.1, max=2.0, step=0.1, description='Espacement X:', style={'description_width': '120px'})
spacing_y = widgets.FloatSlider(value=0.5, min=0.1, max=2.0, step=0.1, description='Espacement Y:', style={'description_width': '120px'})
opt_type = widgets.Dropdown(options=['Amplitude', 'Phase', 'Amplitude+Phase'], value='Amplitude', description='Opt Type:', style={'description_width': '120px'})
objective = widgets.Dropdown(options=['Minimiser SSL', 'Maximiser Gain', 'Multicritères'], value='Minimiser SSL', description='Objectif:', style={'description_width': '120px'})
algorithm = widgets.Dropdown(options=['PSO', 'ACO', 'ABC', 'GA', 'DE', 'SA', 'FA', 'BA', 'CS', 'GWO', 'HHO', 'WOA', 'FPA', 'SCA', 'TLBO'], value='PSO', description='Algorithme:', style={'description_width': '120px'})
population = widgets.IntSlider(value=50, min=10, max=200, step=10, description='Population:', style={'description_width': '120px'})
iterations = widgets.IntSlider(value=100, min=10, max=500, step=10, description='Itérations:', style={'description_width': '120px'})
runs = widgets.IntSlider(value=1, min=1, max=20, step=1, description='Runs:', style={'description_width': '120px'})

output = Output()

def on_launch(b):
    with output:
        clear_output()
        all_results = []
        for run in range(runs.value):
            print(f"⏳ Run {run+1}/{runs.value}...")
            optimizer = AntennaOptimizer(geometry.value, n_elem_x.value, n_elem_y.value, frequency.value, spacing_x.value, spacing_y.value)
            results = optimizer.optimize(algorithm=algorithm.value, opt_type=opt_type.value, objective=objective.value, population=population.value, iterations=iterations.value)
            if results:
                all_results.append(results)
        
        if all_results:
            print(f"\n✅ Optimisation complétée sur {runs.value} run(s)!")
            best_result = min(all_results, key=lambda x: x['metrics']['ssl'])
            optimizer.results = best_result
            optimizer.display_results()

btn_launch = widgets.Button(description='▶ LANCER', button_style='success', icon='play')
btn_launch.on_click(on_launch)

display(VBox([
    widgets.HTML("<b>📡 Configuration</b>"),
    geometry, n_elem_x, n_elem_y, frequency, spacing_x, spacing_y,
    widgets.HTML("<b>⚙️ Algorithme</b>"),
    opt_type, objective, algorithm, population, iterations, runs,
    btn_launch, output
], layout=widgets.Layout(border='2px solid blue', padding='10px')))
