import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import itertools
import sys
import random
from itertools import product

# =============================================================================
# Physical constants (SI)
# =============================================================================
R_GAS = 8.314				 # J/(mol*K)
k_BOLTZ = 1.380649e-23		 # J/K
h_PLANCK = 6.62607015e-34	 # J*s

# =============================================================================
# Global switches for water activity model
# =============================================================================
USE_NONIDEAL_WATER_ACTIVITY = False
A_W_THRESHOLD = 1.0
A_W_POWER = 1.0
A_W_MIN = 1e-30

seed_0=10

# =============================================================================
# Water concentration law (physical time)
# =============================================================================
def water_concentration(t, mode="constant", params=None):
	"""
	Return [H2O](t) in *physical time* units.

	mode = "constant":
		params = {"c": value}

	mode = "square":
		params = {
			"c_dry": 0.055,
			"c_wet": 55.0,
			"period": 86400.0,	   # seconds
			"duty_cycle": 0.5	   # fraction dry
		}

	mode = "custom":
		params = {"func": callable(t_phys) -> concentration}
	"""
	if params is None:
		params = {}

	if mode == "constant":
		return float(params.get("c", 1.0))

	elif mode == "square":
		c_dry = float(params.get("c_dry", 0.01))
		c_wet = float(params.get("c_wet", 55.0))
		period = float(params.get("period", 10.0))
		duty = float(params.get("duty_cycle", 0.5))
		
		# Force ONLY the initial instant to be dry
# 		if np.isclose(t, 0.0):
# 			return c_dry
# 
# 		phase = (t % period) / period
# 		return c_wet if phase < duty else c_dry

		phase = (t % period) / period
		return c_wet if phase < duty else c_dry
		#return c_dry if phase < duty else c_wet

	elif mode == "custom":
		func = params.get("func", None)
		if func is None:
			raise ValueError("mode='custom' requires params['func']")
		return float(func(t))

	else:
		raise ValueError(f"Unknown water mode: {mode}")


def water_concentration_derivative(t, mode="constant", params=None, eps=1e-3):
	"""
	Numerical derivative d[H2O]/dt in physical time.
	For discontinuous 'square' we return 0 and handle jumps via rescaling.
	"""
	if params is None:
		params = {}

	if mode == "square":
		return 0.0

	t_plus = t + eps
	t_minus = t - eps
	c_plus = water_concentration(t_plus, mode=mode, params=params)
	c_minus = water_concentration(t_minus, mode=mode, params=params)

	den = max(abs(c_plus), abs(c_minus), 1e-12)
	rel_jump = abs(c_plus - c_minus) / den
	if rel_jump > 0.5:
		return 0.0

	return (c_plus - c_minus) / (2.0 * eps)


# =============================================================================
# Non-ideal water activity model (THERMO ONLY via reverse term)
# =============================================================================
def water_activity(c_water):
	c = float(c_water)
	if not USE_NONIDEAL_WATER_ACTIVITY:
		return max(c, A_W_MIN)

	if c < A_W_THRESHOLD:
		aw = c**A_W_POWER
	else:
		aw = c

	return max(aw, A_W_MIN)


# =============================================================================
# Temperature law (physical time)
# =============================================================================

def temperature(t, mode="constant", params=None):
	"""
	Return T(t) in Kelvin.

	t is physical time in seconds.

	Modes
	-----
	mode="constant":
		params = {"T": 298.15}	# Kelvin

	mode="piecewise":
		params = {
			"times": [t0, t1, t2, ...],	  # seconds, strictly increasing
			"temps": [T0, T1, ...],		  # one per interval
			"unit": "K" or "C"
		}

	mode="cycle_with_perturbation":
		params = {
			"T_base": 25.0,			 # baseline temperature
			"unit": "C",			 # "C" or "K"
			"cycle_days": 10,
			"cycle_length_days": 1.0,
			"perturb_day": 3,		 # human counting: day 3
			"delta_T": 40.0			 # increase by 40 deg C or K
		}
	"""
	if params is None:
		params = {}

	if mode == "constant":
		Tval = params.get("T", 298.15)
		return float(Tval)

	elif mode == "piecewise":
		times = params.get("times", None)
		temps = params.get("temps", None)
		unit = str(params.get("unit", "K")).upper()

		if times is None or temps is None:
			raise ValueError("mode='piecewise' requires params['times'] and params['temps']")

		times = np.asarray(times, dtype=float)
		temps = np.asarray(temps, dtype=float)

		if times.ndim != 1 or temps.ndim != 1:
			raise ValueError("'times' and 'temps' must be 1D")
		if times.size < 2:
			raise ValueError("'times' must have at least 2 points")
		if temps.size != times.size - 1:
			raise ValueError("'temps' must have len(times)-1 entries")
		if np.any(np.diff(times) <= 0):
			raise ValueError("'times' must be strictly increasing")

		if t <= times[0]:
			Tval = temps[0]
		elif t >= times[-1]:
			Tval = temps[-1]
		else:
			k = int(np.searchsorted(times, t, side="right") - 1)
			Tval = temps[k]

		if unit == "C":
			return float(Tval + 273.15)
		return float(Tval)

	elif mode == "cycle_with_perturbation":
		unit = str(params.get("unit", "C")).upper()

		T_base = float(params.get("T_base", 25.0))
		cycle_days = int(params.get("cycle_days", 10))
		cycle_length_days = float(params.get("cycle_length_days", 1.0))
		perturb_day = int(params.get("perturb_day", 3))
		delta_T = float(params.get("delta_T", 40.0))

		seconds_per_day = 24 * 3600
		cycle_length = cycle_length_days * seconds_per_day
		total_time = cycle_days * cycle_length

		# Human day counting:
		# day 1 = t from 0 to 1 day
		# day 2 = t from 1 to 2 days
		# day 3 = t from 2 to 3 days
		perturb_start = (perturb_day - 1) * cycle_length
		perturb_end = perturb_day * cycle_length

		Tval = T_base

		if perturb_start <= t < perturb_end:
			Tval += delta_T

		# Optional: keep constant after simulated window
		if t < 0 or t > total_time:
			Tval = T_base

		if unit == "C":
			return float(Tval + 273.15)
		return float(Tval)

	else:
		raise ValueError(f"Unknown temperature mode: {mode}")
# =============================================================================
# Species generation
# =============================================================================
def generate_sequences(M, L_max):
	"""
	Generate all sequences over alphabet {0,...,M-1} with lengths 1..L_max.
	Returns sequences list and seq_to_idx dict.
	"""
	alphabet = [str(a) for a in range(M)]
	sequences = []
	for L in range(1, L_max + 1):
		for tup in itertools.product(alphabet, repeat=L):
			sequences.append("".join(tup))
	seq_to_idx = {s: i for i, s in enumerate(sequences)}
	return sequences, seq_to_idx

def random_oligomer_configuration(M, desired_length=None, L_max=None, rng=None):
	"""
	Return a random oligomer configuration as a string.

	Parameters
	----------
	M : int
		Number of monomer types. The alphabet is {"0", "1", ..., str(M-1)}.
	desired_length : int or None, default None
		Desired oligomer length.
		- If an integer is given, the returned sequence has exactly that length.
		- If None, a random length is chosen from 2 to L_max inclusive.
	L_max : int or None, default None
		Maximum allowed length when desired_length is None.
		Ignored if desired_length is provided.
	rng : random.Random or None
		Optional random number generator. If None, a new one is used.

	Returns
	-------
	str
		Random oligomer configuration, e.g. "012", "000", "1101".
	"""

	if rng is None:
		rng = random.Random()

	if not isinstance(M, int) or M < 1:
		raise ValueError("M must be an integer >= 1.")

	if desired_length is not None:
		if not isinstance(desired_length, int) or desired_length < 1:
			raise ValueError("desired_length must be an integer >= 1.")
		L = desired_length
	else:
		if L_max is None:
			raise ValueError("If desired_length is None, you must provide L_max.")
		if not isinstance(L_max, int) or L_max < 2:
			raise ValueError("L_max must be an integer >= 2.")
		L = rng.randint(2, L_max)

	alphabet = [str(i) for i in range(M)]
	return "".join(rng.choice(alphabet) for _ in range(L))
# =============================================================================
# Parameter lookup helpers (bond types and exchange type-pairs)
# =============================================================================
def _make_bondtype_param_lookup(x):
	"""
	Convert user input into a function f(bond_type)->value.

	Allowed inputs:
	  - scalar: same for all types
	  - list/tuple length 1..3:
		  [type0, type1, type2plus]
		If len=2: [type0, type1plus]
		If len=1: [type0]
	Rule for type>=K: use last entry.
	"""
	if isinstance(x, (int, float, np.floating)):
		val = float(x)
		return lambda bond_type: val

	x_list = list(x)
	if len(x_list) == 0:
		raise ValueError("Parameter list must be non-empty.")

	if len(x_list) == 1:
		val = float(x_list[0])
		return lambda bond_type: val

	if len(x_list) == 2:
		v0 = float(x_list[0])
		v1p = float(x_list[1])
		return lambda bond_type: (v0 if bond_type == 0 else v1p)

	v0 = float(x_list[0])
	v1 = float(x_list[1])
	v2p = float(x_list[2])

	def f(bond_type):
		if bond_type == 0:
			return v0
		elif bond_type == 1:
			return v1
		else:
			return v2p
	return f


def _make_bondpair_param_lookup(x, M):
	"""
	Convert user input into f(new_type, old_type) -> value.

	Allowed:
	  - scalar: constant
	  - dict: keys (new,old) tuples or "new,old"
	  - square matrix (K,K): bucketed with idx=min(type,K-1)
	"""
	if isinstance(x, (int, float, np.floating)):
		val = float(x)
		return lambda new_t, old_t: val

	if isinstance(x, dict):
		def f(new_t, old_t):
			if (new_t, old_t) in x:
				return float(x[(new_t, old_t)])
			key_str = f"{new_t},{old_t}"
			if key_str in x:
				return float(x[key_str])
			raise KeyError(f"Missing exchange parameter for (new,old)=({new_t},{old_t})")
		return f

	arr = np.array(x, dtype=float)
	if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
		raise ValueError(f"Exchange parameter matrix must be square. Got {arr.shape}.")
	K = arr.shape[0]

	def f(new_t, old_t):
		new_i = new_t if new_t < K else (K - 1)
		old_i = old_t if old_t < K else (K - 1)
		return float(arr[new_i, old_i])
	return f


# =============================================================================
# Reaction builders (Condensation, Hydrolysis and Exchange)
# =============================================================================
def build_reactions_all_oligomers_thermo(
	sequences,
	seq_to_idx,
	L_max,
	deltaG_act_kcal,
	deltaG_std_kcal,
	random_factor_range=(0.5, 1.5),
	seed=seed_0+10,
):
	"""
	Condensation/Hydrolysis:
		s_i + s_j <-> s_ij + H2O

	We store:
		dG_act_J (ΔG‡), dG_std_J (ΔG°), alpha
	and compute:
		k_r(T) = alpha*(kBT/h)*exp(-ΔG‡/RT)
		K_eq(T) = exp(-ΔG°/RT)
		k_f(T) = k_r(T)*K_eq(T)
	"""
	rng = np.random.default_rng(seed)
	reactions = []
	N = len(sequences)

	act_of_type = _make_bondtype_param_lookup(deltaG_act_kcal)
	std_of_type = _make_bondtype_param_lookup(deltaG_std_kcal)

	for i in range(N):
		s_i = sequences[i]
		L_i = len(s_i)

		for j in range(N):
			s_j = sequences[j]
			L_j = len(s_j)

			if L_i + L_j > L_max:
				continue

			s_new = s_i + s_j
			k_idx = seq_to_idx[s_new]

			bond_type = int(s_j[0])	 # first monomer of appended fragment

			dG_act_J = act_of_type(bond_type) * 4184.0
			dG_std_J = std_of_type(bond_type) * 4184.0
			alpha = rng.uniform(random_factor_range[0], random_factor_range[1])

			reactions.append({
				'i': i, 'j': j, 'k': k_idx,
				'dG_act_J': dG_act_J,
				'dG_std_J': dG_std_J,
				'alpha': alpha,
				'is_dimer': (i == j),
			})

	return reactions


def build_exchange_reactions_thermo(
	sequences,
	seq_to_idx,
	L_max,
	M,
	deltaGx_act_kcal,
	deltaGx_std_kcal,
	random_factor_range=(0.5, 1.5),
	seed=seed_0+10,
):
	"""
	Exchange reactions via cutting oligomer i = A + B (len(i) >= 2).

	We store:
		dGx_act_J, dGx_std_J, alpha
	and compute k(T) inside RHS.
	"""
	rng = np.random.default_rng(seed)
	act_pair = _make_bondpair_param_lookup(deltaGx_act_kcal, M)
	std_pair = _make_bondpair_param_lookup(deltaGx_std_kcal, M)

	ex_rxns = []
	seen = set()
	N = len(sequences)

	for i_idx in range(N):
		i = sequences[i_idx]
		L_i = len(i)
		if L_i < 2:
			continue

		for cut in range(1, L_i):
			A = i[:cut]
			B = i[cut:]
			old_type = int(B[0])

			A_idx = seq_to_idx[A]
			B_idx = seq_to_idx[B]

			for j_idx in range(N):
				j = sequences[j_idx]

				# Channel (1): i + j <-> (A+j) + B
				if len(A) + len(j) <= L_max:
					C_seq = A + j
					C_idx = seq_to_idx[C_seq]
					D_idx = B_idx
					new_type = int(j[0])

					react_pair = tuple(sorted((i_idx, j_idx)))
					prod_pair = tuple(sorted((C_idx, D_idx)))
					key = (react_pair, prod_pair, "Aplusj")

					if key not in seen:
						seen.add(key)
						dGx_act_J = act_pair(new_type, old_type) * 4184.0
						dGx_std_J = std_pair(new_type, old_type) * 4184.0
						alpha = rng.uniform(random_factor_range[0], random_factor_range[1])

						ex_rxns.append({
							'a': i_idx, 'b': j_idx, 'c': C_idx, 'd': D_idx,
							'dGx_act_J': dGx_act_J,
							'dGx_std_J': dGx_std_J,
							'alpha': alpha,
							'new_type': new_type, 'old_type': old_type
						})

				# Channel (2): i + j <-> (j+B) + A
				if len(j) + len(B) <= L_max:
					C_seq = j + B
					C_idx = seq_to_idx[C_seq]
					D_idx = A_idx
					new_type = int(B[0])

					react_pair = tuple(sorted((i_idx, j_idx)))
					prod_pair = tuple(sorted((C_idx, D_idx)))
					key = (react_pair, prod_pair, "jplusB")

					if key not in seen:
						seen.add(key)
						dGx_act_J = act_pair(new_type, old_type) * 4184.0
						dGx_std_J = std_pair(new_type, old_type) * 4184.0
						alpha = rng.uniform(random_factor_range[0], random_factor_range[1])

						ex_rxns.append({
							'a': i_idx, 'b': j_idx, 'c': C_idx, 'd': D_idx,
							'dGx_act_J': dGx_act_J,
							'dGx_std_J': dGx_std_J,
							'alpha': alpha,
							'new_type': new_type, 'old_type': old_type
						})

	return ex_rxns


def build_exchange_matrices(deltaG_std_kcal, deltaG_exch_act_kcal):
	G_std = np.asarray(deltaG_std_kcal, dtype=float)
	G_exch = np.asarray(deltaG_exch_act_kcal, dtype=float)

	deltaGx_std_kcal = G_std[:, None] - G_std[None, :]

	deltaGx_act_kcal = (
		G_exch[None, :] +
		np.maximum(0.0, -deltaGx_std_kcal)
	)

	return deltaGx_act_kcal, deltaGx_std_kcal


# =============================================================================
# CATALYSIS
# =============================================================================
def build_periodic_masks(sequences, reactions, pattern_specs, period):
	"""
	pattern_specs: list of tuples (interval_index, reactant1_str, reactant2_str, product_str)
	period: int, pattern period in *physical time units* (same units used for interval indexing).
	"""
	n_reactions = len(reactions)

	triple_to_idx = {}
	for r_idx, r in enumerate(reactions):
		s_i = sequences[r['i']]
		s_j = sequences[r['j']]
		s_k = sequences[r['k']]
		triple_to_idx[(s_i, s_j, s_k)] = r_idx

	masks = {}
	for (interval, s1, s2, sprod) in pattern_specs:
		if interval < 0 or interval >= period:
			raise ValueError(f"Interval {interval} outside period [0,{period-1}]")
		key = (s1, s2, sprod)
		if key not in triple_to_idx:
			raise ValueError(f"Requested enhanced reaction {key} not found.")
		r_idx = triple_to_idx[key]

		if interval not in masks:
			masks[interval] = np.zeros(n_reactions, dtype=bool)
		masks[interval][r_idx] = True

	base_mask = np.zeros(n_reactions, dtype=bool)
	return masks, base_mask


def build_periodic_mask_from_specs(sequences, reactions, pattern_specs, period):
	"""
	pattern_specs: list of tuples (interval_index, reactant1_str, reactant2_str, product_str)
	period: int, pattern period in model-interval indexing space.
	"""
	n_reactions = len(reactions)

	triple_to_idx = {}
	for r_idx, r in enumerate(reactions):
		s_i = sequences[r['i']]
		s_j = sequences[r['j']]
		s_k = sequences[r['k']]
		triple_to_idx[(s_i, s_j, s_k)] = r_idx

	masks = {}
	for (interval, s1, s2, sprod) in pattern_specs:
		if interval < 0 or interval >= period:
			raise ValueError(f"Interval {interval} outside period [0,{period-1}]")
		key = (s1, s2, sprod)
		if key not in triple_to_idx:
			raise ValueError(f"Requested catalyzed reaction {key} not found.")
		r_idx = triple_to_idx[key]

		if interval not in masks:
			masks[interval] = np.zeros(n_reactions, dtype=bool)
		masks[interval][r_idx] = True

	base_mask = np.zeros(n_reactions, dtype=bool)
	return masks, base_mask


def generate_vectors_by_k_sequence(
	t_span,
	dt_pattern: float,
	size_set: int,
	n_monomers: int,
	max_total_len: int,
	k_step: int = 2,
	seed: int | None = None,
):
	rng = random.Random(seed)

	if n_monomers < 1:
		raise ValueError("n_monomers must be >= 1.")
	if max_total_len < 2:
		raise ValueError("max_total_len must be >= 2.")
	if size_set < 1:
		raise ValueError("size_set must be >= 1.")
	if dt_pattern <= 0:
		raise ValueError("dt_pattern must be > 0.")
	if k_step < 1:
		raise ValueError("k_step must be >= 1.")

	period = int(round((t_span[1] - t_span[0]) / dt_pattern))
	k_max = period - 1
	alphabet = [str(i) for i in range(n_monomers)]

	valid_pairs = []
	for L1 in range(1, max_total_len):
		for L2 in range(1, max_total_len - L1 + 1):
			for p1 in product(alphabet, repeat=L1):
				s1 = "".join(p1)
				for p2 in product(alphabet, repeat=L2):
					s2 = "".join(p2)
					valid_pairs.append((s1, s2))

	if not valid_pairs:
		raise RuntimeError("No valid (s1,s2) pairs.")

	k_values = list(range(0, k_max + 1, k_step))

	vectors = []
	for k in k_values:
		for _ in range(size_set):
			s1, s2 = rng.choice(valid_pairs)
			vectors.append((k, s1, s2, s1 + s2))

	return vectors


def build_periodic_catalysis_sets(
	n_monomers: int,
	max_total_len: int,
	size_set: int,
	phase_step: int = 2,
	period_sets: int = 10,
	seed: int | None = None,
):
	"""
	Returns a dict: phase_k -> list[(s1,s2,s1+s2)]	where phase_k in {0,2,4,...}<period_sets.
	The set for any actual bin k is obtained via phase_k = k % period_sets.
	"""
	rng = random.Random(seed)

	alphabet = [str(i) for i in range(n_monomers)]

	valid_pairs = []
	for L1 in range(1, max_total_len):
		for L2 in range(1, max_total_len - L1 + 1):
			for p1 in product(alphabet, repeat=L1):
				s1 = "".join(p1)
				for p2 in product(alphabet, repeat=L2):
					s2 = "".join(p2)
					valid_pairs.append((s1, s2))

	if not valid_pairs:
		raise RuntimeError("No valid (s1,s2) pairs.")

	phase_bins = list(range(0, period_sets, phase_step))
	periodic_sets = {}
	for phase_k in phase_bins:
		periodic_sets[phase_k] = [
			(s1, s2, s1 + s2) for (s1, s2) in (rng.choice(valid_pairs) for _ in range(size_set))
		]

	return periodic_sets, period_sets


def expand_periodic_sets_to_specs_with_order(
	t_span,
	dt_pattern: float,
	periodic_sets: dict[int, list[tuple[str, str, str]]],
	period_sets: int,
	phase_step: int = 2,
	order=None,
):
	"""
	Same as expand_periodic_sets_to_specs, but allows you to permute the order
	of which phase-bin set is used within each cycle.

	order can be:
	  - None: identity order (A,B,C,D,...)
	  - list of indices into phase_bins, e.g. [1,0,3,2] meaning B,A,D,C
	  - list of phase-bin values, e.g. [2,0,6,4] meaning (phase_k=2),(0),(6),(4)
	"""
	n_bins_total = int(round((t_span[1] - t_span[0]) / dt_pattern))

	phase_bins = list(range(0, period_sets, phase_step))
	if len(phase_bins) == 0:
		return []

	if order is None:
		order_phase_bins = phase_bins
	else:
		if all(isinstance(x, int) for x in order) and all(0 <= x < len(phase_bins) for x in order):
			order_phase_bins = [phase_bins[i] for i in order]
		else:
			order_phase_bins = list(order)

	for pb in order_phase_bins:
		if pb not in periodic_sets:
			raise KeyError(f"phase bin {pb} not in periodic_sets. Available keys: {sorted(periodic_sets.keys())}")

	cycle_len = len(order_phase_bins)

	specs = []
	for k in range(n_bins_total):
		pos_in_cycle = k % cycle_len
		phase_k = order_phase_bins[pos_in_cycle]
		for (s1, s2, s12) in periodic_sets[phase_k]:
			specs.append((k, s1, s2, s12))

	return specs


# =============================================================================
# Species-mediated catalysis helpers
# =============================================================================
def catalytic_multiplier(
	C,
	catalytic_list,
	mode="linear",
	reactant_indices=None,
	catalysis_threshold_fraction=0.0
):
	"""
	catalytic_list entries can be either:
	  (cat_idx, strength)
	or
	  {"cat_idx": ..., "strength": ...}

	If reactant_indices is provided, catalysis only applies when
	  [cat] >= catalysis_threshold_fraction * max([reactants])
	"""
	if not catalytic_list:
		return 1.0

	active_total = 0.0

	for item in catalytic_list:
		if isinstance(item, dict):
			cat_idx = int(item["cat_idx"])
			strength = float(item["strength"])
		else:
			cat_idx, strength = item
			cat_idx = int(cat_idx)
			strength = float(strength)

		cat_conc = C[cat_idx]

		activation = 1.0
		if reactant_indices is not None and len(reactant_indices) > 0:
			ref = max(float(C[idx]) for idx in reactant_indices)
			thr = catalysis_threshold_fraction * ref
			activation = cat_conc / (cat_conc + thr) if thr > 0.0 else 1.0

		active_total += activation * strength * cat_conc

	if mode == "linear":
		return 1.0 + active_total

	elif mode == "saturating":
		return 1.0 + active_total / (1.0 + active_total)

	else:
		raise ValueError(f"Unknown catalytic mode: {mode}")


def build_species_catalysis_map(
	sequences,
	seq_to_idx,
	reactions,
	catalysis_rules=None,
	hydrolysis_bondtype_catalysts=None,
	max_hydrolysis_target_len=None,
	hydrolysis_sequence_catalysts=None,
):
	"""
	Returns:
		species_catalysis_map[r_idx] = {
			"fwd": [ ... ],
			"bwd": [ ... ]
		}

	Two independent mechanisms can populate the map:

	1) Explicit reaction rules via catalysis_rules:
	   [
		 {
		   "catalyst": "112",
		   "direction": "fwd",
		   "targets": [("01","21","0121")],
		   "strength": 100.0
		 },
		 ...
	   ]

	2) Automatic hydrolysis catalysis by broken-bond type via hydrolysis_bondtype_catalysts:
	   {
		 2: ("000", 1000.0),
		 0: ("111", 1000.0),
		 1: ("222", 1000.0),
	   }

	3) Automatic hydrolysis catalysis by target sequence via hydrolysis_sequence_catalysts:
	   [
		 {
		   "catalyst": "01210",
		   "strength": 100000.0,
		   "target_length": 5,
		   "exclude_self": True
		 },
		 {
		   "catalyst": "11220",
		   "strength": 100000.0,
		   "target_length": 5,
		   "exclude_self": True
		 },
	   ]
	"""
	species_catalysis_map = {
		r_idx: {"fwd": [], "bwd": []}
		for r_idx in range(len(reactions))
	}

	# ---------- explicit per-reaction rules ----------
	if catalysis_rules is not None:
		triple_to_idx = {}
		for r_idx, r in enumerate(reactions):
			s_i = sequences[r["i"]]
			s_j = sequences[r["j"]]
			s_k = sequences[r["k"]]
			triple_to_idx[(s_i, s_j, s_k)] = r_idx

		for rule in catalysis_rules:
			catalyst = str(rule["catalyst"])
			direction = str(rule["direction"]).lower()
			strength = float(rule.get("strength", 1.0))
			targets = rule["targets"]

			if catalyst not in seq_to_idx:
				raise KeyError(f"Catalyst species '{catalyst}' not found in sequences.")
			if direction not in ("fwd", "bwd"):
				raise ValueError(f"Unknown direction '{direction}'. Use 'fwd' or 'bwd'.")

			cat_idx = seq_to_idx[catalyst]

			for tgt in targets:
				key = tuple(str(x) for x in tgt)
				if key not in triple_to_idx:
					raise KeyError(f"Target reaction {key} not found.")
				r_idx = triple_to_idx[key]
				species_catalysis_map[r_idx][direction].append({
					"cat_idx": cat_idx,
					"strength": strength,
				})

	# ---------- automatic hydrolysis catalysis by broken-bond type ----------
	if hydrolysis_bondtype_catalysts is not None:
		for bond_type, cat_info in hydrolysis_bondtype_catalysts.items():
			if isinstance(cat_info, (list, tuple)) and len(cat_info) == 2:
				catalyst_species, strength = cat_info
			else:
				raise ValueError(
					"hydrolysis_bondtype_catalysts values must be tuples like "
					"{bond_type: ('catalyst_species', strength)}"
				)

			catalyst_species = str(catalyst_species)
			strength = float(strength)

			if catalyst_species not in seq_to_idx:
				raise KeyError(f"Catalyst species '{catalyst_species}' not found in sequences.")
			cat_idx = seq_to_idx[catalyst_species]

			for r_idx, r in enumerate(reactions):
				s_j = sequences[r["j"]]
				s_k = sequences[r["k"]]

				if len(s_j) < 1:
					continue

				broken_bond_type = int(s_j[0])

				if broken_bond_type != int(bond_type):
					continue

				if max_hydrolysis_target_len is not None and len(s_k) > int(max_hydrolysis_target_len):
					continue

				species_catalysis_map[r_idx]["bwd"].append({
					"cat_idx": cat_idx,
					"strength": strength,
				})

	# ---------- automatic hydrolysis catalysis by target sequence ----------
	if hydrolysis_sequence_catalysts is not None:
		for rule in hydrolysis_sequence_catalysts:
			catalyst_species = str(rule["catalyst"])
			strength = float(rule["strength"])
			target_length = rule.get("target_length", None)
			exclude_self = bool(rule.get("exclude_self", True))

			if catalyst_species not in seq_to_idx:
				raise KeyError(f"Catalyst species '{catalyst_species}' not found in sequences.")
			cat_idx = seq_to_idx[catalyst_species]

			for r_idx, r in enumerate(reactions):
				s_k = sequences[r["k"]]	  # hydrolysis target

				if target_length is not None and len(s_k) != int(target_length):
					continue

				if exclude_self and s_k == catalyst_species:
					continue

				species_catalysis_map[r_idx]["bwd"].append({
					"cat_idx": cat_idx,
					"strength": strength,
				})

	return species_catalysis_map


# =============================================================================
# Assembly-based protection against catalytic hydrolysis
# =============================================================================
def build_assembly_protection_rules(seq_to_idx, assembly_pairs=None, default_threshold_fraction=0.1):
	"""
	Input examples:
	  [
		("000", "122"),
		("111", "020"),
		("222", "110"),
	  ]
	"""
	if assembly_pairs is None:
		return []

	rules_out = []

	for entry in assembly_pairs:
		if isinstance(entry, dict):
			catalyst_species = str(entry["catalyst"])
			partner_species = str(entry["partner"])
			threshold_fraction = float(entry.get("threshold_fraction", default_threshold_fraction))
		else:
			if len(entry) != 2:
				raise ValueError(
					"assembly_pairs entries must be either "
					"('catalyst','partner') or "
					"{'catalyst':..., 'partner':..., 'threshold_fraction':...}"
				)
			catalyst_species = str(entry[0])
			partner_species = str(entry[1])
			threshold_fraction = float(default_threshold_fraction)

		if catalyst_species not in seq_to_idx:
			raise KeyError(f"Assembly protection catalyst species '{catalyst_species}' not found.")
		if partner_species not in seq_to_idx:
			raise KeyError(f"Assembly protection partner species '{partner_species}' not found.")

		rules_out.append({
			"catalyst_idx": seq_to_idx[catalyst_species],
			"partner_idx": seq_to_idx[partner_species],
			"threshold_fraction": threshold_fraction,
		})

	return rules_out


def catalytic_multiplier_with_assembly_protection(
	C,
	catalytic_list,
	target_idx,
	assembly_protection_rules=None,
	mode="linear",
	reactant_indices=None,
	catalysis_threshold_fraction=0.0,
):
	if not catalytic_list:
		return 1.0

	if assembly_protection_rules is None:
		assembly_protection_rules = []

	protection = 0.0

	for rule in assembly_protection_rules:
		rule_cat = rule["catalyst_idx"]
		rule_partner = rule["partner_idx"]
		thr = float(rule["threshold_fraction"])

		# protect BOTH members of the pair from ANY hydrolysis catalyst
		if target_idx not in (rule_cat, rule_partner):
			continue

		c1 = float(C[rule_cat])
		c2 = float(C[rule_partner])

		prod = c1 * c2
		protection = max(protection, prod / (prod + thr))
		break

	filtered_total = 0.0
	for item in catalytic_list:
		if isinstance(item, dict):
			cat_idx = int(item["cat_idx"])
			strength = float(item["strength"])
		else:
			cat_idx, strength = item
			cat_idx = int(cat_idx)
			strength = float(strength)

		cat_conc = C[cat_idx]

		activation = 1.0
		if reactant_indices is not None and len(reactant_indices) > 0:
			ref = max(float(C[idx]) for idx in reactant_indices)
			thr = catalysis_threshold_fraction * ref
			activation = cat_conc / (cat_conc + thr) if thr > 0.0 else 1.0

		filtered_total += activation * (1.0 - protection) * strength * cat_conc

	if mode == "linear":
		return 1.0 + filtered_total
	elif mode == "saturating":
		return 1.0 + filtered_total / (1.0 + filtered_total)
	else:
		raise ValueError(f"Unknown catalytic mode: {mode}")


# =============================================================================
# Core ODE system (time-dependent T, [H2O])
# =============================================================================
def ode_system_thermo_driven(
	t_phys,
	C,
	reactions,
	exchange_reactions,
	masks_fwd,
	base_mask_fwd,
	enh_factor_fwd,
	masks_bwd,
	base_mask_bwd,
	enh_factor_bwd,
	period,
	water_mode,
	water_params,
	volume_coupled=True,
	temp_mode="constant",
	temp_params=None,
	interval_index_override=None,
	all_cat_fwd=False,
	all_cat_bwd=False,
	species_catalysis_map=None,
	species_catalysis_mode="linear",
	assembly_protection_rules=None,
	catalysis_threshold_fraction=0.0,
):
	"""
	Condensation/Hydrolysis:
		s_i + s_j <-> s_k + H2O
		v_f = k_f(T) [i][j]
		v_r = k_r(T) [k][H2O](t)

	Exchange:
		a + b <-> c + d
		v_f = k_f(T) [a][b]
		v_r = k_r(T) [c][d]
	"""
	dCdt = np.zeros_like(C)

	if interval_index_override is None:
		interval_index = int(np.floor(t_phys))
	else:
		interval_index = int(interval_index_override)

	interval_mod = interval_index % period
	mask_fwd = masks_fwd.get(interval_mod, base_mask_fwd)
	mask_bwd = masks_bwd.get(interval_mod, base_mask_bwd)

	c_water = water_concentration(t_phys, mode=water_mode, params=water_params)
	T_now = temperature(t_phys, mode=temp_mode, params=temp_params)

	kappa = 1.0
	kBT_over_h = kappa * (k_BOLTZ * T_now) / h_PLANCK

	# --- Condensation/Hydrolysis ---
	for r_idx, r in enumerate(reactions):
		i = r['i']
		j = r['j']
		k = r['k']
		alpha = r['alpha']
		dG_act_J = r['dG_act_J']
		dG_std_J = r['dG_std_J']

		k_r = alpha * (kBT_over_h * np.exp(-dG_act_J / (R_GAS * T_now)))
		K_eq = np.exp(-dG_std_J / (R_GAS * T_now))
		k_f = k_r * K_eq

		# external / periodic catalysis
		if all_cat_fwd:
			mult_fwd = float(enh_factor_fwd)
		else:
			mult_fwd = float(enh_factor_fwd) if mask_fwd[r_idx] else 1.0

		if all_cat_bwd:
			mult_bwd = float(enh_factor_bwd)
		else:
			mult_bwd = float(enh_factor_bwd) if mask_bwd[r_idx] else 1.0

		# species-mediated catalysis
		if species_catalysis_map is not None and r_idx in species_catalysis_map:
			catinfo = species_catalysis_map[r_idx]

			mult_fwd *= catalytic_multiplier(
				C,
				catinfo["fwd"],
				mode=species_catalysis_mode,
				reactant_indices=[i, j],
				catalysis_threshold_fraction=catalysis_threshold_fraction,
			)

			mult_bwd *= catalytic_multiplier_with_assembly_protection(
				C,
				catinfo["bwd"],
				target_idx=k,
				assembly_protection_rules=assembly_protection_rules,
				mode=species_catalysis_mode,
				reactant_indices=[k],
				catalysis_threshold_fraction=catalysis_threshold_fraction,
			)

		k_f_eff = k_f * mult_fwd
		k_r_eff = k_r * mult_bwd

		v_f = k_f_eff * C[i] * C[j]
		a_w = water_activity(c_water)
		v_r = k_r_eff * C[k] * a_w

		if r['is_dimer']:
			dCdt[i] += 2.0 * v_r - 2.0 * v_f
			dCdt[k] += v_f - v_r
		else:
			dCdt[i] += v_r - v_f
			dCdt[j] += v_r - v_f
			dCdt[k] += v_f - v_r

	# --- Exchange ---
	for rx in exchange_reactions:
		a = rx['a']
		b = rx['b']
		c = rx['c']
		d = rx['d']
		alpha = rx['alpha']
		dGx_act_J = rx['dGx_act_J']
		dGx_std_J = rx['dGx_std_J']

		k_r = alpha * (kBT_over_h * np.exp(-dGx_act_J / (R_GAS * T_now)))
		K_eq = np.exp(-dGx_std_J / (R_GAS * T_now))
		k_f = k_r * K_eq

		v_f = k_f * C[a] * C[b]
		v_r = k_r * C[c] * C[d]

		if a == b:
			dCdt[a] += -2.0 * v_f + 2.0 * v_r
		else:
			dCdt[a] += -v_f + v_r
			dCdt[b] += -v_f + v_r

		if c == d:
			dCdt[c] += 2.0 * v_f - 2.0 * v_r
		else:
			dCdt[c] += v_f - v_r
			dCdt[d] += v_f - v_r

	# --- Continuous volume coupling (smooth water laws only) ---
	if volume_coupled and c_water > 0.0:
		dc_water_dt = water_concentration_derivative(
			t_phys, mode=water_mode, params=water_params, eps=1e-3
		)
		if dc_water_dt != 0.0:
			scale_rate = dc_water_dt / c_water
			dCdt -= scale_rate * C

	return dCdt


# =============================================================================
# Initial condition helper
# =============================================================================
def build_uniform_initial_C0(sequences, seq_to_idx, M, L_max, C_total, init_length=1):
	"""
	Uniform initial condition across all sequences of a given length.
	Total concentration C_total is split evenly among all M^init_length sequences.
	"""
	if init_length < 1:
		raise ValueError("init_length must be >= 1")
	if init_length > L_max:
		raise ValueError(f"init_length={init_length} exceeds L_max={L_max}. Increase L_max.")

	N_species = len(sequences)
	C0 = np.zeros(N_species)

	n_bins = M ** init_length
	C_each = C_total / n_bins

	for s in sequences:
		if len(s) == init_length:
			C0[seq_to_idx[s]] = C_each

	return C0


def apply_monomer_initials(C0, seq_to_idx, M, monomer_C0=None, init_length=1):
	"""
	Set explicit initial concentrations for monomers (init_length=1) or, more generally,
	for all sequences of length 'init_length' using a vector in lexicographic order.
	"""
	if monomer_C0 is None:
		return C0

	monomer_C0 = np.asarray(monomer_C0, dtype=float)

	n_bins = M ** init_length
	if monomer_C0.size != n_bins:
		raise ValueError(
			f"monomer_C0 must have length M**init_length = {n_bins} "
			f"(got {monomer_C0.size})."
		)

	C0 = C0.copy()

	alphabet = [str(i) for i in range(M)]
	k = 0
	for tup in itertools.product(alphabet, repeat=init_length):
		s = "".join(tup)
		if s not in seq_to_idx:
			raise KeyError(f"Initial species '{s}' not in seq_to_idx (check L_max and M).")
		C0[seq_to_idx[s]] = float(monomer_C0[k])
		k += 1

	return C0


# =============================================================================
# Square-wave volume coupling: instantaneous rescaling at phase boundaries
# =============================================================================
def square_switch_times(t0_phys, tf_phys, period, duty_cycle):
	"""
	Switches at n*period and n*period + duty*period.
	"""
	switches = []
	n_start = int(np.floor(t0_phys / period)) - 1
	n_end = int(np.ceil(tf_phys / period)) + 1

	for n in range(n_start, n_end + 1):
		t_a = n * period
		t_b = n * period + duty_cycle * period
		if t0_phys < t_a < tf_phys:
			switches.append(t_a)
		if t0_phys < t_b < tf_phys:
			switches.append(t_b)

	return sorted(set(switches))


def rescale_concentrations_for_water_jump(C, c_old, c_new):
	"""
	Enforce C ∝ 1/[H2O] across an instantaneous water jump:
	  C_new = C_old * (c_old / c_new)
	"""
	if c_new <= 0:
		raise ValueError("Water concentration must be > 0.")
	
	return C * (c_old / c_new)


# =============================================================================
# Feeding of oligomers
# =============================================================================
def apply_feed_dict(C, seq_to_idx, feed_dict, feeding_type='constant_add'):
	"""
	Instantaneously add concentrations for multiple species:
	  feed_dict = {"1": 0.02, "0": 0.01, "10": 0.005, ...}
	"""
	if feed_dict is None:
		return C

	C = C.copy()
	for sp, amt in feed_dict.items():
		amt = float(amt)
		if amt == 0.0:
			continue
		sp = str(sp)
		if sp not in seq_to_idx:
			raise KeyError(f"Feed species '{sp}' not found in sequences (check M and L_max).")
		if feeding_type == 'constant_add':
			C[seq_to_idx[sp]] += amt
		elif feeding_type == 'constant_total':
			C[seq_to_idx[sp]] = amt
		else:
			raise ValueError("feeding_type must be 'constant_add' or 'constant_total'")
	return C


def normalize_feed_schedule(t_span, feed_times_model=None, feed_amounts=None):
	"""
	Returns a dict: {t_feed_model: feed_dict_to_apply_at_that_time}
	"""
	if feed_times_model is None:
		return {}

	t0, tf = float(t_span[0]), float(t_span[1])
	times = [float(x) for x in feed_times_model]

	if feed_amounts is None:
		raise ValueError("If feed_times_model is provided, you must provide feed_amounts (dict or list of dicts).")

	if isinstance(feed_amounts, dict):
		per_time = [dict(feed_amounts) for _ in times]
	else:
		if len(feed_amounts) != len(times):
			raise ValueError("feed_amounts list must have same length as feed_times_model.")
		per_time = [dict(d) for d in feed_amounts]

	schedule = {}
	for t, d in zip(times, per_time):
		if not (t0 < t < tf):
			continue
		if t not in schedule:
			schedule[t] = {}
		for sp, amt in d.items():
			sp = str(sp)
			schedule[t][sp] = schedule[t].get(sp, 0.0) + float(amt)

	return schedule


# =============================================================================
# Simulation driver (time-dependent T, [H2O])
# =============================================================================
def simulate_thermo_pattern(
	pattern_specs_fwd=None,
	pattern_specs_bwd=None,
	dt_pattern=0.5,
	period=1,
	L_max=6,
	M=2,
	C0_total=1.0,
	init_length=1,
	deltaG_act_kcal=25.0,
	deltaG_std_kcal=3.0,
	enable_exchange=True,
	deltaGx_act_kcal=25.0,
	deltaGx_std_kcal=0.0,
	random_factor_range=(0.5, 1.5),
	seed=seed_0+10,
	t_span=(0.0, 100.0),
	time_scale=1.0,
	n_points=2000,
	enh_factor_fwd=1.0,
	enh_factor_bwd=1.0,
	water_mode="constant",
	water_params=None,
	volume_coupled=True,
	reference_water=55.0,
	temp_mode="constant",
	temp_params=None,
	FEED_ENABLED=False,
	feed_times_model=None,
	feed_amounts=None,
	monomer_C0=None,
	feeding_type='constant_add',
	all_cat_fwd=False,
	all_cat_bwd=False,
	catalysis_rules=None,
	hydrolysis_bondtype_catalysts=None,
	max_hydrolysis_target_len=None,
	hydrolysis_sequence_catalysts=None,
	species_catalysis_mode="linear",
	assembly_pairs=None,
	assembly_threshold_fraction=0.1,
	catalysis_threshold_fraction=0.0,
):
	if water_params is None:
		water_params = {}
	if temp_params is None:
		temp_params = {}
	if pattern_specs_fwd is None:
		pattern_specs_fwd = []
	if pattern_specs_bwd is None:
		pattern_specs_bwd = []

	# --- Species ---
	sequences, seq_to_idx = generate_sequences(M, L_max)
	seq_lengths = np.array([len(s) for s in sequences], dtype=float)

	# --- Reactions ---
	reactions = build_reactions_all_oligomers_thermo(
		sequences, seq_to_idx, L_max,
		deltaG_act_kcal=deltaG_act_kcal,
		deltaG_std_kcal=deltaG_std_kcal,
		random_factor_range=random_factor_range,
		seed=seed,
	)

	if enable_exchange:
		exchange_reactions = build_exchange_reactions_thermo(
			sequences=sequences,
			seq_to_idx=seq_to_idx,
			L_max=L_max,
			M=M,
			deltaGx_act_kcal=deltaGx_act_kcal,
			deltaGx_std_kcal=deltaGx_std_kcal,
			random_factor_range=random_factor_range,
			seed=seed + 999,
		)
	else:
		exchange_reactions = []

	# --- Driving masks ---
	masks_fwd, base_mask_fwd = build_periodic_mask_from_specs(
		sequences, reactions, pattern_specs_fwd, period
	)
	masks_bwd, base_mask_bwd = build_periodic_mask_from_specs(
		sequences, reactions, pattern_specs_bwd, period
	)

	# --- Species-mediated catalysis map ---
	species_catalysis_map = build_species_catalysis_map(
		sequences=sequences,
		seq_to_idx=seq_to_idx,
		reactions=reactions,
		catalysis_rules=catalysis_rules,
		hydrolysis_bondtype_catalysts=hydrolysis_bondtype_catalysts,
		max_hydrolysis_target_len=max_hydrolysis_target_len,
		hydrolysis_sequence_catalysts=hydrolysis_sequence_catalysts,
	)

	# --- Assembly-based protection against catalytic hydrolysis ---
	assembly_protection_rules = build_assembly_protection_rules(
		seq_to_idx=seq_to_idx,
		assembly_pairs=assembly_pairs,
		default_threshold_fraction=assembly_threshold_fraction,
	)

	# --- Initial concentrations ---
	C0 = build_uniform_initial_C0(
		sequences=sequences,
		seq_to_idx=seq_to_idx,
		M=M,
		L_max=L_max,
		C_total=C0_total,
		init_length=init_length,
	)

	C0 = apply_monomer_initials(
		C0=C0,
		seq_to_idx=seq_to_idx,
		M=M,
		monomer_C0=monomer_C0,
		init_length=init_length,
	)

	# --- Initial scaling ---
	if volume_coupled:
		cw0 = water_concentration(0.0, mode=water_mode, params=water_params)
		if cw0 <= 0.0:
			raise ValueError("Initial water concentration must be > 0 for volume_coupled=True")
		#print (C0,cw0,reference_water,reference_water / cw0)
		C0 *= (reference_water / cw0)
		#print (C0)

	# --- Time grids ---
	t0_model, tf_model = t_span
	t_eval = np.linspace(t0_model, tf_model, n_points)

	# --- Feeding schedule ---
	if FEED_ENABLED:
		feed_schedule = normalize_feed_schedule(
			t_span=t_span,
			feed_times_model=feed_times_model,
			feed_amounts=feed_amounts
		)
	else:
		feed_schedule = {}

	feed_times_unique = sorted(feed_schedule.keys())

	# RHS in model time
	def rhs_model_time(t_model, C):
		t_phys = t_model * time_scale
		interval_index_model = int(np.floor(t_model / dt_pattern))
		dCdt_phys = ode_system_thermo_driven(
			t_phys,
			C,
			reactions=reactions,
			exchange_reactions=exchange_reactions,
			masks_fwd=masks_fwd,
			base_mask_fwd=base_mask_fwd,
			enh_factor_fwd=enh_factor_fwd,
			masks_bwd=masks_bwd,
			base_mask_bwd=base_mask_bwd,
			enh_factor_bwd=enh_factor_bwd,
			period=period,
			water_mode=water_mode,
			water_params=water_params,
			volume_coupled=volume_coupled,
			temp_mode=temp_mode,
			temp_params=temp_params,
			interval_index_override=interval_index_model,
			all_cat_fwd=all_cat_fwd,
			all_cat_bwd=all_cat_bwd,
			species_catalysis_map=species_catalysis_map,
			species_catalysis_mode=species_catalysis_mode,
			assembly_protection_rules=assembly_protection_rules,
			catalysis_threshold_fraction=catalysis_threshold_fraction,
		)
		return time_scale * dCdt_phys

	# --- Piecewise integration if needed ---
	has_driving_fwd = (enh_factor_fwd != 1.0) and (pattern_specs_fwd is not None) and (len(pattern_specs_fwd) > 0)
	has_driving_bwd = (enh_factor_bwd != 1.0) and (pattern_specs_bwd is not None) and (len(pattern_specs_bwd) > 0)
	has_driving = has_driving_fwd or has_driving_bwd
	has_square = (volume_coupled and water_mode == "square")
	has_feed = (len(feed_times_unique) > 0)

	do_piecewise = has_square or has_feed or has_driving

	if do_piecewise:
		boundaries = [t0_model, tf_model]

		if has_driving:
			k_start = int(np.floor(t0_model / dt_pattern)) + 1
			k_end = int(np.floor(tf_model / dt_pattern))
			if k_end >= k_start:
				boundaries.extend([(k * dt_pattern) for k in range(k_start, k_end + 1)])

		switch_model = []
		if has_square:
			period_w = float(water_params.get("period", 10.0))
			duty_w = float(water_params.get("duty_cycle", 0.5))
			switch_phys = square_switch_times(t0_model * time_scale, tf_model * time_scale, period_w, duty_w)
			switch_model = [ts / time_scale for ts in switch_phys]
			boundaries.extend([ts for ts in switch_model if t0_model < ts < tf_model])

		boundaries.extend(feed_times_unique)
		boundaries = sorted(set(boundaries))

		T_all = []
		Y_all = []
		y_current = C0.copy()

		def _is_same_time(a, b, tol=1e-12):
			return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

		for seg_i in range(len(boundaries) - 1):
			seg_t0 = boundaries[seg_i]
			seg_tf = boundaries[seg_i + 1]

			if seg_i == len(boundaries) - 2:
				mask_eval = (t_eval >= seg_t0) & (t_eval <= seg_tf)
			else:
				mask_eval = (t_eval >= seg_t0) & (t_eval < seg_tf)
			t_eval_seg = t_eval[mask_eval]

			if len(t_eval_seg) == 0 or t_eval_seg[0] != seg_t0:
				t_eval_seg = np.insert(t_eval_seg, 0, seg_t0)

			sol = solve_ivp(
				rhs_model_time,
				(seg_t0, seg_tf),
				y_current,
				t_eval=t_eval_seg,
				method="BDF",
				rtol=1e-8,
				atol=1e-10,
			)
			if not sol.success:
				raise RuntimeError("ODE solver failed: " + sol.message)

			if seg_i == 0:
				T_all.extend(sol.t.tolist())
				Y_all.extend(sol.y.T.tolist())
			else:
				T_all.extend(sol.t[1:].tolist())
				Y_all.extend(sol.y.T[1:].tolist())

			y_current = sol.y[:, -1].copy()

			# rescale at square-wave boundary
			if has_square:
				is_switch = any(_is_same_time(seg_tf, ts) for ts in switch_model)
				if is_switch:
					t_phys_boundary = seg_tf * time_scale
					period_w = float(water_params.get("period", 10.0))
					eps = 1e-9 * period_w
					c_old = water_concentration(t_phys_boundary - eps, mode=water_mode, params=water_params)
					c_new = water_concentration(t_phys_boundary + eps, mode=water_mode, params=water_params)
					if c_old != c_new:
						#before = y_current.copy()
						y_current = rescale_concentrations_for_water_jump(y_current, c_old, c_new)

# 						print(
# 							"\nSWITCH at model time", seg_tf,
# 							"c_old =", c_old,
# 							"c_new =", c_new,
# 							"factor =", c_old / c_new,
# 							"max before =", before.max(),
# 							"max after =", y_current.max()
# 						)

			# apply feeding at this boundary
			for t_feed in feed_times_unique:
				if _is_same_time(seg_tf, t_feed):
					y_current = apply_feed_dict(
						y_current,
						seq_to_idx,
						feed_schedule[t_feed],
						feeding_type=feeding_type
					)

			pct = int((seg_i + 1) * 100 / (len(boundaries) - 1))

			t_phys_now = seg_tf * time_scale
			cw_now = water_concentration(t_phys_now, mode=water_mode, params=water_params)
			n_now = y_current * (cw_now / reference_water)
			total_monomer_equiv = float(seq_lengths @ n_now)

			sys.stdout.write(f"\rSimulation {pct}% complete | total={total_monomer_equiv:.2e}")
			sys.stdout.flush()

		sys.stdout.write("\n")
		t_out = np.array(T_all)
		C_out = np.array(Y_all).T

	else:
		sol = solve_ivp(
			rhs_model_time,
			(t0_model, tf_model),
			C0,
			t_eval=t_eval,
			method="BDF",
			rtol=1e-8,
			atol=1e-10,
		)
		if not sol.success:
			raise RuntimeError("ODE solver failed: " + sol.message)
		t_out = sol.t
		C_out = sol.y

	return t_out, C_out, sequences, seq_to_idx, reactions, exchange_reactions


def make_alternating_temperature_phystime(
	t_end_model, time_scale, dt_model=0.5,
	T_low=40.0, T_high=80.0, unit="C"
):
	"""
	Build a piecewise temperature schedule where:
	  - model time runs from 0..t_end_model
	  - the schedule alternates every dt_model (in model units)
	  - returned 'times' are in PHYSICAL seconds because RHS calls temperature(t_phys)
	"""
	t_end_phys = float(t_end_model) * float(time_scale)
	dt_phys = float(dt_model) * float(time_scale)

	times = [0.0]
	temps = []
	t = 0.0
	k = 0

	while t < t_end_phys:
		temps.append(T_high if (k % 2 == 0) else T_low)
		t = min(t + dt_phys, t_end_phys)
		times.append(t)
		k += 1

	return {"times": times, "temps": temps, "unit": unit}


# =============================================================================
# Printing and saving
# =============================================================================
def print_plotted_timeseries_dict_phys_time_only(
	t, C_sol, sequences,
	time_scale=1.0,
	water_mode="constant",
	water_params=None,
	reference_water=55.0,
	use_volume_scaling=False,
	every=1,
	printing_values=False
):
	if not printing_values:
		return

	from pprint import pprint

	if water_params is None:
		water_params = {}

	t = np.asarray(t)
	C_sol = np.asarray(C_sol)

	if C_sol.shape[1] != t.size:
		raise ValueError("C_sol must have shape (n_species, n_time) with n_time == len(t).")

	every = max(1, int(every))
	idx = np.arange(0, t.size, every, dtype=int)

	t_phys = (t[idx] * float(time_scale))
	t_phys_list = t_phys.tolist()

	y = C_sol[:, idx].copy()
	if use_volume_scaling:
		cw = np.array([water_concentration(tp, mode=water_mode, params=water_params) for tp in t_phys_list], dtype=float)
		eps = 1e-30
		y *= (cw[np.newaxis, :] / (float(reference_water) + eps))

	order = list(range(len(sequences)))
	order.sort(key=lambda i: (len(sequences[i]), sequences[i]))

	y_dict = {sequences[i]: y[i, :].tolist() for i in order}
	out = {"t_phys_s": t_phys_list, "y": y_dict}

	print("\n# ===== COPY EVERYTHING BELOW THIS LINE =====\n")
	pprint(out, width=140, sort_dicts=False)
	print("\n# ===== COPY EVERYTHING ABOVE THIS LINE =====\n")


def save_timeseries_to_txt(
	t, C_sol, sequences,
	printing_values_txt=False,
	filename="timeseries_values.txt",
	time_scale=1.0,
	water_mode="constant",
	water_params=None,
	reference_water=55.0,
	use_volume_scaling=False,
	every=1,
	params_dict=None
):
	if not printing_values_txt:
		return

	from datetime import datetime

	if water_params is None:
		water_params = {}
	if params_dict is None:
		params_dict = {}

	t = np.asarray(t)
	C_sol = np.asarray(C_sol)
	if C_sol.shape[1] != t.size:
		raise ValueError("C_sol must have shape (n_species, n_time) with n_time == len(t).")

	every = max(1, int(every))
	idx = np.arange(0, t.size, every, dtype=int)

	t_phys = (t[idx] * float(time_scale))
	t_phys_list = t_phys.tolist()

	y = C_sol[:, idx].copy()
	if use_volume_scaling:
		cw = np.array([water_concentration(tp, mode=water_mode, params=water_params) for tp in t_phys_list], dtype=float)
		eps = 1e-30
		y *= (cw[np.newaxis, :] / (float(reference_water) + eps))

	order = list(range(len(sequences)))
	order.sort(key=lambda i: (len(sequences[i]), sequences[i]))

	auto = {
		"generated": datetime.now().isoformat(timespec="seconds"),
		"n_species": len(sequences),
		"n_timepoints_written": int(idx.size),
		"every": every,
		"time_scale": time_scale,
		"water_mode": water_mode,
		"water_params": water_params,
		"reference_water": reference_water,
		"use_volume_scaling": use_volume_scaling,
		"t_phys_start_s": float(t_phys[0]) if t_phys.size else None,
		"t_phys_end_s": float(t_phys[-1]) if t_phys.size else None,
	}

	with open(filename, "w", encoding="utf-8") as f:
		f.write("# ============================================================\n")
		f.write("# Full time-trajectory output (dict-like)\n")
		f.write("# ============================================================\n\n")

		f.write("[Simulation parameters]\n")
		merged = dict(params_dict)
		merged.update(auto)
		for k in sorted(merged.keys(), key=lambda x: str(x)):
			f.write(f"- {k}: {merged[k]}\n")
		f.write("\n")

		f.write("t_phys_s:\n")
		f.write("  [" + ", ".join(f"{x:.16e}" for x in t_phys_list) + "]\n\n")

		f.write("y:\n")
		for i in order:
			seq = sequences[i]
			f.write(f"	{seq}:\n")
			f.write("	 [" + ", ".join(f"{v:.16e}" for v in y[i, :].tolist()) + "]\n")

	print(f"Wrote time series to: {filename}")


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_final_bar_by_length_log(C_sol, sequences, L_max, title_suffix=""):
	final_C = C_sol[:, -1]
	N_species = len(sequences)

	indices = list(range(N_species))
	indices.sort(key=lambda idx: (len(sequences[idx]), sequences[idx]))

	sorted_seqs = [sequences[i] for i in indices]
	sorted_C = final_C[indices]

	cmap = plt.cm.tab10
	base_colors = {}
	for L in range(1, max(L_max, 1) + 1):
		frac = (L - 1) / max(L_max - 1, 1) if L_max > 1 else 0.0
		base_colors[L] = cmap(frac)

	colors = [base_colors[len(s)] for s in sorted_seqs]

	eps = 1e-15
	sorted_C_plot = sorted_C + eps

	x = np.arange(N_species)

	plt.figure(figsize=(14, 5))
	plt.bar(x, sorted_C_plot, color=colors, width=0.8)
	plt.xticks(x, sorted_seqs, rotation=90)
	plt.xlabel("Sequence")
	plt.ylabel("Final concentration")
	plt.title(f"Final concentrations by sequence {title_suffix}".strip())
	plt.tight_layout()
	plt.show()


def plot_all_species_moles(
	t, C_sol, sequences,
	time_scale,
	water_mode,
	water_params,
	reference_water=55.0,
	use_volume_scaling=False,
	label_prefix="",
	bond_weight=0.65,
	num_bond_types=None
):
	import matplotlib.colors as mcolors

	n_sol = C_sol.copy()

	if use_volume_scaling:
		t_phys = t * time_scale
		c_water_t = np.array([water_concentration(tp, mode=water_mode, params=water_params) for tp in t_phys])
		eps = 1e-30
		n_sol = C_sol * (c_water_t[np.newaxis, :] / (reference_water + eps))

	indices = list(range(len(sequences)))
	indices.sort(key=lambda idx: (len(sequences[idx]), sequences[idx]))

	length_cmaps = {
		1: plt.cm.PuRd,
		2: plt.cm.Blues,
		3: plt.cm.Greens,
		4: plt.cm.Reds,
	}
	extra_cycle = [plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Oranges, plt.cm.YlOrRd, plt.cm.Greys]

	def length_cmap(L):
		if L in length_cmaps:
			return length_cmaps[L]
		return extra_cycle[(L - 5) % len(extra_cycle)]

	if num_bond_types is None:
		all_bt = []
		for s in sequences:
			if len(s) > 1:
				all_bt.extend([int(ch) for ch in s[1:]])
		num_bond_types = (max(all_bt) + 1) if all_bt else 1

	plt.figure(figsize=(12, 6))
	for idx in indices:
		s = sequences[idx]
		L = len(s)

		cmapL = length_cmap(L)
		base_rgb = np.array(mcolors.to_rgb(cmapL(0.55)))

		if len(s) <= 1 or num_bond_types <= 1:
			bond_rgb = base_rgb
		else:
			counts = np.zeros(num_bond_types, dtype=float)
			for ch in s[1:]:
				bt = int(ch)
				if 0 <= bt < num_bond_types:
					counts[bt] += 1.0
			if counts.sum() <= 0:
				bond_rgb = base_rgb
			else:
				dom = int(np.argmax(counts))
				u = 0.25 + 0.60 * (dom / max(num_bond_types - 1, 1))
				bond_rgb = np.array(mcolors.to_rgb(cmapL(u)))

		color = (1.0 - bond_weight) * base_rgb + bond_weight * bond_rgb
		plt.plot(t, n_sol[idx, :], color=color, alpha=0.9, label=f"{label_prefix}{s}")

	plt.xlabel("Time (model units)")
	plt.ylabel("Total moles (arb. units)")
	title_extra = " (volume-scaled)" if use_volume_scaling else " (constant volume)"
	plt.title("All species total moles vs time" + title_extra)
	#plt.legend(fontsize=7, ncol=3)
	plt.tight_layout()
	plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
	# -------------------------------
	# User parameters
	# -------------------------------
	L_max = 2
	M = 3
	init_length = 1
	C0_total = 0.1
	#monomer_C0 = [0.2,0.2]
	monomer_C0 = [0.1,0.1,0.1]#,0.192]#,0.076]#, 0.1]
	seed_0 = 14	   
	rng = random.Random(seed_0)

	# Time
	t_span = (0.0, 14.999)
	time_scale = 86400.0
	n_points = 4000#1000000#4000

	enable_exchange = True

	# Bond-type dependent parameters (0,1,2+)
#	  deltaG_act_kcal = [26.0, 33.0, 35.0]
#	  deltaG_std_kcal = [5.0, 3.0, 3.0]
#	  deltaG_exch_act_kcal = [29.0, 29.0, 360.0]

# 	deltaG_act_kcal = [24.0, 28.0, 35.0]
# 	deltaG_std_kcal = [7.0, 5.0, 3.0]
# 	deltaG_exch_act_kcal = [30.0, 32.0, 36.0]

	deltaG_act_kcal = [25,28.5,35]
	deltaG_std_kcal = [7,5,3]
	deltaG_exch_act_kcal = [31,33,36]

	deltaG_act_kcal = [24,28,35]
	deltaG_std_kcal = [7,5,3]
	deltaG_exch_act_kcal = [30,32,36]

	deltaGx_act_kcal, deltaGx_std_kcal = build_exchange_matrices(
		deltaG_std_kcal=deltaG_std_kcal,
		deltaG_exch_act_kcal=deltaG_exch_act_kcal
	)

	################################# WATER ACTIVITY #########################################
	water_mode = "square"
	water_params = {
		"c_dry": .01,
		"c_wet": 55.0,
		"period": 86400.0,
		"duty_cycle": 0.5,
	}

	volume_coupled = True
	reference_water = 55.0

	USE_NONIDEAL_WATER_ACTIVITY = False
	A_W_THRESHOLD = 1.0
	A_W_POWER = 1.0
	A_W_MIN = 1e-30

	################################# TEMPERATURE ############################################
	temp_mode = "constant"
	temp_params = {"T": 272 + 50}

# 	temp_mode = "cycle_with_perturbation"
# 	temp_params = {
# 		"T_base": 45.0,
# 		"unit": "C",
# 		"cycle_days": 10,
# 		"cycle_length_days": 1.,
# 		"perturb_day": 3,
# 		"delta_T": 0.0
# 	}
# 
# 	temp_mode = "piecewise"
# 	temp_params = make_alternating_temperature_phystime(
# 		t_end_model=t_span[1],
# 		time_scale=time_scale,
# 		dt_model=0.5,
# 		T_low=160.0,
# 		T_high=140.0,
# 		unit="C"
# 	)
	
			
	##################################### FEEDING ############################################
	FEED_ENABLED = False
	feeding_type = 'constant_total'	  # 'constant_add' or 'constant_total'

	feed_times_model = [i + 0.5 for i in range(40)]
	feed_amounts = [
		{"0": 0.1, "1": 0.1, "2": 0.1} for _ in feed_times_model
	]

	#################################### CATALYSIS ###########################################
	dt_pattern = 1.0
	period = int(round((t_span[1] - t_span[0]) / dt_pattern))

	# Periodic external catalysis switches
	all_cat_fwd = False
	all_cat_bwd = False
	enh_factor_fwd = 100000.0
	enh_factor_bwd = 10000000.0

	pattern_fwd_specs = []
	pattern_bwd_specs = []

	###################### CATALYSYS HYDROLYSIS ###############################
	# Catalysis of Hydrolysis: Species-mediated bond-type-targeted
	hydrolysis_bondtype_catalysts = {
#		  2: ("000", enh_factor_bwd),	   #   000 catalyzes all -2 bonds
#		  0: ("111", enh_factor_bwd),	   #   111 catalyzes all -0 bonds
#		  1: ("222", enh_factor_bwd),	 #	 222 catalyzes all -1 bonds
	}
	hydrolysis_bondtype_catalysts = {
#		  2: (random_oligomer_configuration(M=3, desired_length=3,rng=rng), enh_factor_bwd),	   #   000 catalyzes all -2 bonds
#		  0: (random_oligomer_configuration(M=3, desired_length=3,rng=rng), enh_factor_bwd),	   #   111 catalyzes all -0 bonds
#		  1: (random_oligomer_configuration(M=3, desired_length=3,rng=rng), enh_factor_bwd),	#	222 catalyzes all -1 bonds
	}
	#print ('Hydrolyzers:',hydrolysis_bondtype_catalysts)
	max_hydrolysis_target_len = 4

	# Catalysis of Hydrolysis: Species mediated length-target
	hydrolysis_sequence_catalysts = [
#		  {
#			  "catalyst": "01210",
#			  "strength": enh_factor_bwd,
#			  "target_length": 5,
#			  "exclude_self": True,
#		  },
#		  {
#			  "catalyst": "11220",
#			  "strength": enh_factor_bwd,
#			  "target_length": 5,
#			  "exclude_self": True,
#		  },
#		  {
#			  "catalyst": "10220",
#			  "strength": enh_factor_bwd,
#			  "target_length": 5,
#			  "exclude_self": True,
#		  },
	]
	
	# Catalysis of Hydrolysis: Species mediated configuration-target
	#
	#
	#
	#
	
	###################### CATALYSYS CONDENSATION ###############################

	# Catalysis of Condensation: Species mediated configuration-target
	catalysis_rules = [
#		  {
#			  "catalyst": "110",
#			  "direction": "fwd",
#			  "targets": [("01", "210", "01210")],
#			  "strength": enh_factor_fwd,
#		  },
#		  {
#			  "catalyst": "020",
#			  "direction": "fwd",
#			  "targets": [("11", "220", "11220")],
#			  "strength": enh_factor_fwd,
#		  },
#		  {
#			  "catalyst": "122",
#			  "direction": "fwd",
#			  "targets": [("10", "220", "10220")],
#			  "strength": enh_factor_fwd,
#		  },
#		  {
#			  "catalyst": "001",
#			  "direction": "fwd",
#			  "targets": [("21", "210", "21210")],
#			  "strength": 5*enh_factor_fwd,
#		  },
#		  {
#			  "catalyst": "010",
#			  "direction": "fwd",
#			  "targets": [("11", "002", "11002")],
#			  "strength": 5*enh_factor_fwd,
#		  },
	]

	catalysis_threshold_fraction = 0.1 # Catalyst concentration must be at least this fraction of the largest reactant concentration
	species_catalysis_mode = "linear"

	###################### RESISTANCE HYDROLYSIS ###############################
	# Assembly-based protection against catalytic hydrolysis
	assembly_pairs = [
#		  ("000", "122"),
#		  ("111", "020"),
#		  ("222", "110"),
	]
	assembly_pairs = [
#		  (random_oligomer_configuration(M=3, desired_length=3,rng=rng), random_oligomer_configuration(M=3, desired_length=3,rng=rng)),
#		  (random_oligomer_configuration(M=3, desired_length=3,rng=rng), random_oligomer_configuration(M=3, desired_length=3,rng=rng)),
#		  (random_oligomer_configuration(M=3, desired_length=3,rng=rng), random_oligomer_configuration(M=3, desired_length=3,rng=rng)),
	]
	#print ('Assembly pairs:',assembly_pairs)
	# smooth protection = prod / (prod + threshold)
	assembly_threshold_fraction = 0.001**2

	################################## PRINTING FILE #########################################
	print_every = 1
	printing_values = False
	printing_values_txt = True

	##################################### RUN ################################################
	t, C, sequences, seq_to_idx, rxns, exrxns = simulate_thermo_pattern(
		pattern_specs_fwd=pattern_fwd_specs,
		pattern_specs_bwd=pattern_bwd_specs,
		dt_pattern=dt_pattern,
		period=period,
		L_max=L_max,
		M=M,
		C0_total=C0_total,
		init_length=init_length,
		deltaG_act_kcal=deltaG_act_kcal,
		deltaG_std_kcal=deltaG_std_kcal,
		enable_exchange=enable_exchange,
		deltaGx_act_kcal=deltaGx_act_kcal,
		deltaGx_std_kcal=deltaGx_std_kcal,
		random_factor_range=(0.5, 1.5),
		seed=seed_0,
		t_span=t_span,
		time_scale=time_scale,
		n_points=n_points,
		enh_factor_fwd=enh_factor_fwd,
		enh_factor_bwd=enh_factor_bwd,
		water_mode=water_mode,
		water_params=water_params,
		volume_coupled=volume_coupled,
		reference_water=reference_water,
		temp_mode=temp_mode,
		temp_params=temp_params,
		FEED_ENABLED=FEED_ENABLED,
		feed_times_model=feed_times_model,
		feed_amounts=feed_amounts,
		monomer_C0=monomer_C0,
		feeding_type=feeding_type,
		all_cat_fwd=all_cat_fwd,
		all_cat_bwd=all_cat_bwd,
		catalysis_rules=catalysis_rules,
		hydrolysis_bondtype_catalysts=hydrolysis_bondtype_catalysts,
		max_hydrolysis_target_len=max_hydrolysis_target_len,
		hydrolysis_sequence_catalysts=hydrolysis_sequence_catalysts,
		species_catalysis_mode=species_catalysis_mode,
		assembly_pairs=assembly_pairs,
		assembly_threshold_fraction=assembly_threshold_fraction,
		catalysis_threshold_fraction=catalysis_threshold_fraction,
	)

	print("Simulation done.")
	print(f"Species: {len(sequences)}")
	print(f"Condensation/Hydrolysis reactions: {len(rxns)}")
	print(f"Exchange reactions: {len(exrxns)}")

	# -------------------------------
	# Print (optional)
	# -------------------------------
	print_plotted_timeseries_dict_phys_time_only(
		t, C, sequences,
		time_scale=time_scale,
		water_mode=water_mode,
		water_params=water_params,
		reference_water=reference_water,
		use_volume_scaling=True,
		every=print_every,
		printing_values=printing_values
	)

	# -------------------------------
	# Save time series to .txt
	# -------------------------------
	save_timeseries_to_txt(
		t, C, sequences,
		printing_values_txt=printing_values_txt,
		filename="timeseries_values.txt",
		time_scale=time_scale,
		water_mode=water_mode,
		water_params=water_params,
		reference_water=reference_water,
		use_volume_scaling=True,
		every=print_every,
		params_dict={
			"L_max": L_max,
			"M": M,
			"init_length": init_length,
			"C0_total": C0_total,
			"deltaG_act_kcal": deltaG_act_kcal,
			"deltaG_std_kcal": deltaG_std_kcal,
			"enable_exchange": enable_exchange,
			"deltaGx_act_kcal": deltaGx_act_kcal,
			"deltaGx_std_kcal": deltaGx_std_kcal,
			"t_span": t_span,
			"n_points": n_points,
			"enh_factor_fwd": enh_factor_fwd,
			"enh_factor_bwd": enh_factor_bwd,
			"period": period,
			"water_mode": water_mode,
			"water_params": water_params,
			"volume_coupled": volume_coupled,
			"reference_water": reference_water,
			"temp_mode": temp_mode,
			"temp_params": temp_params,
			"seed": seed_0,
			"all_cat_fwd": all_cat_fwd,
			"all_cat_bwd": all_cat_bwd,
			"hydrolysis_bondtype_catalysts": hydrolysis_bondtype_catalysts,
			"max_hydrolysis_target_len": max_hydrolysis_target_len,
			"hydrolysis_sequence_catalysts": hydrolysis_sequence_catalysts,
			"species_catalysis_mode": species_catalysis_mode,
			"assembly_pairs": assembly_pairs,
			"assembly_threshold_fraction": assembly_threshold_fraction,
			"catalysis_rules": catalysis_rules,
			"catalysis_threshold_fraction": catalysis_threshold_fraction,
		},
	)

	# -------------------------------
	# Plots
	# -------------------------------
	plot_all_species_moles(
		t, C, sequences,
		time_scale=time_scale,
		water_mode=water_mode,
		water_params=water_params,
		reference_water=reference_water,
		use_volume_scaling=True,
		label_prefix="n~: "
	)
	plot_final_bar_by_length_log(C, sequences, L_max, title_suffix="(final)")