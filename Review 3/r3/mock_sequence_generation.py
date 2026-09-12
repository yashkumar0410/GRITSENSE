"""Generate temporal sequences from saved graph `.pt` files.

This script loads `graph_frame_<id>.pt` files from `output_graphs/`, extracts a
per-frame vector (uses the saved `global_embedding` when present, otherwise
averages node embeddings), and produces fixed-length sequences using the
existing `TemporalSequenceGenerator`.

It prints a short summary of generated sequences.
"""

from pathlib import Path
import glob
import torch
try:
	import torch_geometric
	# Allowlist torch_geometric Data class for safe unpickling (PyTorch 2.6+)
	try:
		torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
	except Exception:
		# Older/newer PyTorch may use safe_globals context manager; ignore if unavailable
		pass
except Exception:
	# If torch_geometric is not installed, torch.load will likely fail earlier.
	pass
import numpy as np

from temporal import FeaturePreparer, TemporalSequenceGenerator


OUTPUT_GRAPHS = Path(__file__).resolve().parent / "output_graphs"


def _load_graph_frames():
	files = sorted(glob.glob(str(OUTPUT_GRAPHS / "graph_frame_*.pt")))
	frames = []
	frame_ids = []
	for path in files:
		# Load full checkpoint (not weights-only) so torch_geometric Data objects
		# are unpickled correctly. This assumes files are trusted local outputs.
		data = torch.load(path, map_location="cpu", weights_only=False)
		frame_id = data.get("frame_id")
		frame_ids.append(frame_id)
		# Prefer global_embedding if present
		if "global_embedding" in data and data["global_embedding"] is not None:
			emb = data["global_embedding"]
			if isinstance(emb, torch.Tensor):
				vec = emb.detach().cpu().squeeze().numpy()
			else:
				vec = np.asarray(emb)
		else:
			# Fallback: mean over HRN node embeddings or GAT embeddings
			if "hrn_node_embeddings" in data and data["hrn_node_embeddings"] is not None:
				emb = data["hrn_node_embeddings"]
				vec = emb.detach().cpu().mean(dim=0).numpy()
			elif "gat_node_embeddings" in data and data["gat_node_embeddings"] is not None:
				emb = data["gat_node_embeddings"]
				vec = emb.detach().cpu().mean(dim=0).numpy()
			else:
				raise RuntimeError(f"No usable embeddings in {path}")
		frames.append(vec.tolist())
	return frame_ids, frames


def main():
	frame_ids, frames = _load_graph_frames()
	if not frames:
		print("No graph frame files found in", OUTPUT_GRAPHS)
		return

	# The temporal pipeline expects named blocks; use a single unnamed block via
	# prepare_matrix which accepts a plain [num_frames, dim] matrix.
	preparer = FeaturePreparer()
	prepared = preparer.prepare_matrix(frames)

	# Create sequences
	generator = TemporalSequenceGenerator(sequence_length=8, stride=1)
	sequences = generator.generate_sequences(prepared)

	print(f"Loaded {len(frames)} frames from output_graphs")
	print(f"Prepared feature matrix: {len(prepared)} x {len(prepared[0])}")
	print(f"Generated sequences: {len(sequences)} (sequence length {len(sequences[0])})")
	if sequences:
		print("First sequence (first frame vector):", sequences[0][0][:8])

	# Print per-frame vectors for all frames
	print("\nPer-frame feature vectors:")
	for fid, vec in zip(frame_ids, frames):
		print(f"Frame {fid}: {np.array(vec)}")

	# Print all sequences with their corresponding frame-id ranges
	seq_len = generator.sequence_length
	step = generator.stride
	print("\nGenerated sequences with frame-id ranges:")
	for i, seq in enumerate(sequences):
		start_idx = i * step
		frame_range = frame_ids[start_idx : start_idx + seq_len]
		print(f"Sequence {i} frames {frame_range} -> sequence vectors:")
		for j, vec in enumerate(seq):
			print(f"  frame {frame_range[j]}: {np.array(vec)}")


if __name__ == "__main__":
	main()
