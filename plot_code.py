import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.colors as mcolors

# =============================================================================
# USER CONTROLS (edit here)
# =============================================================================
FILENAME = "TV_5mers_thio-ester-amide_50.txt"

# Time axis
TIME_UNIT = "days"   # "seconds" or "days"

# Which series to plot
PLOT_MODE = "all"    # "all" or "topN"
TOP_N = 25           # used if PLOT_MODE="topN"
MIN_FINAL = 0.0      # optional filter on final value (applies after TOP_N selection)

# Labeling
Y_LABEL = "Value from file (concentration or n~ depending on how you saved)"

# Legend
LEGEND_NCOL = 3
LEGEND_FONTSIZE = 7

# Color model (MATCHES your plot_all_species_moles)
BOND_WEIGHT = 0.65   # same default you used

# =============================================================================
# File reader (matches save_timeseries_to_txt format)
# =============================================================================
def read_timeseries_from_txt(filename):
	params = {}
	t_phys_s = None
	y = {}
	order = []

	section = None
	current_seq = None
	buffer = []

	def flush_buffer_as_array():
		nonlocal buffer
		s = "".join(buffer).strip()
		buffer = []
		if not s:
			return np.array([])
		return np.array(ast.literal_eval(s), dtype=float)

	def parse_param_value(v_str):
		v_str = v_str.strip()
		try:
			return ast.literal_eval(v_str)
		except Exception:
			return v_str

	with open(filename, "r", encoding="utf-8") as f:
		for raw in f:
			line = raw.rstrip()
			if not line or line.startswith("#"):
				continue

			if line == "[Simulation parameters]":
				section = "params"
				continue
			if line == "t_phys_s:":
				section = "t_phys"
				buffer = []
				continue
			if line == "y:":
				section = "y"
				current_seq = None
				continue

			if section == "params":
				if line.startswith("- "):
					rest = line[2:]
					if ":" in rest:
						k, v = rest.split(":", 1)
						params[k.strip()] = parse_param_value(v)
				continue

			if section == "t_phys":
				if line.lstrip().startswith("[") or buffer:
					buffer.append(line.strip())
					if line.strip().endswith("]"):
						t_phys_s = flush_buffer_as_array()
						section = None
				continue

			if section == "y":
				if line.startswith("  ") and line.endswith(":") and not line.startswith("    "):
					current_seq = line.strip()[:-1]
					y[current_seq] = None
					order.append(current_seq)
					continue

				if current_seq is not None and (line.strip().startswith("[") or buffer):
					buffer.append(line.strip())
					if line.strip().endswith("]"):
						y[current_seq] = flush_buffer_as_array()
						current_seq = None
				continue

	if t_phys_s is None:
		raise ValueError("Could not find/parse 't_phys_s' from file.")
	n_t = t_phys_s.size
	for s in order:
		if y.get(s) is None:
			raise ValueError(f"Could not parse time series for species '{s}'.")
		if y[s].size != n_t:
			raise ValueError(f"Length mismatch for '{s}': len(y)={y[s].size} vs len(t)={n_t}")

	return params, np.asarray(t_phys_s, dtype=float), y, order


# =============================================================================
# Color matching logic (COPIED from your plot_all_species_moles)
# =============================================================================
def infer_num_bond_types(sequences):
	all_bt = []
	for s in sequences:
		if len(s) > 1:
			all_bt.extend([int(ch) for ch in s[1:]])
	return (max(all_bt) + 1) if all_bt else 1

def length_cmap_factory():
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
	return length_cmap

def color_for_sequence(seq, num_bond_types, bond_weight=0.65):
	"""
	Matches your writer-side coloring:
	- Base colormap depends on length
	- "Bond-dominant tint" depends on dominant bond type in s[1:]
	- Final is linear blend with weight bond_weight
	"""
	L = len(seq)
	length_cmap = length_cmap_factory()
	cmapL = length_cmap(L)

	base_rgb = np.array(mcolors.to_rgb(cmapL(0.55)))

	if len(seq) <= 1 or num_bond_types <= 1:
		bond_rgb = base_rgb
	else:
		counts = np.zeros(num_bond_types, dtype=float)
		for ch in seq[1:]:
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
	return color


# =============================================================================
# Plotting from file (NOW color-matched)
# =============================================================================
def plot_timeseries_from_file(filename):
	params, t_phys_s, y_dict, order = read_timeseries_from_txt(filename)

	# Time axis transform
	if TIME_UNIT.lower() == "days":
		t = t_phys_s / 86400.0
		xlabel = "Time (days)"
	else:
		t = t_phys_s
		xlabel = "Time (s)"

	# Select series
	final_vals = np.array([y_dict[s][-1] for s in order], dtype=float)

	if PLOT_MODE == "topN":
		idx_sort = np.argsort(final_vals)[::-1]
		keep = [order[i] for i in idx_sort[:max(1, int(TOP_N))]]
	else:
		keep = list(order)

	if MIN_FINAL > 0.0:
		keep = [s for s in keep if y_dict[s][-1] >= float(MIN_FINAL)]

	# Stable ordering (like your other plotters)
	keep.sort(key=lambda s: (len(s), s))

	# Infer bond-type count for color mapping
	num_bond_types = infer_num_bond_types(keep)

	plt.figure(figsize=(12, 6))
	for s in keep:
		color = color_for_sequence(s, num_bond_types=num_bond_types, bond_weight=BOND_WEIGHT)
		plt.plot(t, y_dict[s], color=color, alpha=0.9, label=s)

	plt.xlabel(xlabel)
	plt.ylabel(Y_LABEL)

	title_bits = []
	if "water_mode" in params:
		title_bits.append(f"water_mode={params['water_mode']}")
	if "water_params" in params:
		title_bits.append(f"water_params={params['water_params']}")
	if "use_volume_scaling" in params:
		title_bits.append(f"use_volume_scaling={params['use_volume_scaling']}")
	title = "All species vs time (from file; color-matched)"
	if title_bits:
		title += " | " + " | ".join(title_bits)
	plt.title(title)

	plt.legend(fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
	plt.tight_layout()
	plt.show()

	return params


def plot_final_bar_by_length_from_file(filename):
	# This bar plot is already close to your original (tab10 by length),
	# so I’m leaving it as-is (since your original bar plot used that scheme).
	params, t_phys_s, y_dict, order = read_timeseries_from_txt(filename)

	order_sorted = sorted(order, key=lambda s: (len(s), s))
	final_sorted = np.array([y_dict[s][-1] for s in order_sorted], dtype=float)

	L_max = max(len(s) for s in order_sorted) if order_sorted else 1

	cmap = plt.cm.tab10
	base_colors = {}
	for L in range(1, L_max + 1):
		frac = (L - 1) / max(L_max - 1, 1) if L_max > 1 else 0.0
		base_colors[L] = cmap(frac)

	colors = [base_colors[len(s)] for s in order_sorted]

	eps = 1e-15
	final_plot = final_sorted + eps

	x = np.arange(len(order_sorted))

	plt.figure(figsize=(14, 5))
	plt.bar(x, final_plot, color=colors, width=0.8)
	plt.xticks(x, order_sorted, rotation=90)
	plt.xlabel("Sequence")
	plt.ylabel("Final value")
	plt.title("Final values by sequence (from file)")
	plt.tight_layout()
	plt.show()

	return params


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
	params = plot_timeseries_from_file(FILENAME)
	plot_final_bar_by_length_from_file(FILENAME)
	print("Done. Parsed parameter keys:", sorted(params.keys(), key=str))
