import sys
sys.path.insert(0, '.')
from src.simulation.monte_carlo import MonteCarloEngine
mc = MonteCarloEngine()
mc.run_simulation({})
s = mc.get_summary()
print(f"P(Colapso) = {s['collapse_probability']}%")
print(f"Semaforo = {s['traffic_light']}")
print(f"Score medio = {s['mean_score']}")
print(f"Distribucion scores: {s['score_distribution']}")
print(f"Por anno: {s['yearly_collapse_prob']}")
