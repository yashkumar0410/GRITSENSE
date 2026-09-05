"""Development-only example using synthetic frame features.

This data is not GritSense data and must not be used to train or evaluate a
production temporal model.
"""

from gritsense.temporal import FeaturePreparer, TemporalSequenceGenerator


# Synthetic blocks only: these are not real GritSense features.
mock_blocks = {
	"player": [[frame, frame + 1] for frame in range(20)],
	"ball": [[frame + 2, frame + 3] for frame in range(20)],
}


prepared_features = FeaturePreparer().prepare(mock_blocks)
generator = TemporalSequenceGenerator(sequence_length=4, stride=2)
sequences = generator.generate_sequences(prepared_features)

print(f"Input shape: [{len(prepared_features)}, {len(prepared_features[0])}]")
print(f"Output shape: [{len(sequences)}, {len(sequences[0])}, {len(sequences[0][0])}]")
print(f"First sequence: {sequences[0]}")
