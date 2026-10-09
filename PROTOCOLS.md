# Frozen protocol identifiers

Configuration suffixes such as `k11` and `k26` identify frozen experiment stages. They bind the implementation, data split, label budget, random initialization, and evaluation phase used by the released evidence.

The YAML configurations specify the model and evaluation settings, and the compact results used by the paper are under `outputs/`. This page maps the stage identifiers to their corresponding studies.

The main protocol families are:

- `k11`--`k18`: Light Tunnel Relation Tube validation, held-out evaluation, geometry, correspondence, radius, and noise controls;
- `k22`--`k36`: CausalVerse Slope relation-content and instrumented-interface studies;
- MOVi-B formal validation/held-out: collision-video propagation controls;
- `k43`: resistance spot welding validation/held-out;
- `k44`: independent Light Tunnel label-subset stability.

Exact commands for the principal released interfaces are listed in [COMMANDS.md](COMMANDS.md). The machine-readable `release-manifest.json` records the SHA-256 of each released file other than the manifest itself.
