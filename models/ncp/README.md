# Neural Circuit Policies (NCP)

**Paper:** Lechner, Hasani, Amini, Henzinger, Rus, Grosu, *"Neural circuit
policies enabling auditable autonomy"*, Nature Machine Intelligence 2020.
Picked as the clearest description of the sparse sensory/inter/command/motor
wiring diagram; see [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

NCP takes LTC neurons (see [`../ltc`](../ltc)) and wires them sparsely instead
of fully-connected, in four layers loosely modeled on the *C. elegans* nervous
system: `sensory -> inter -> command (recurrent) -> motor`. Every neuron only
connects to a handful of targets (fixed at construction, not learned), which
cuts parameters drastically and makes the policy's wiring diagram directly
inspectable/auditable.

## Files

- `model.py` -- `NCPWiring` (sparse mask generator) + `NCPCell` (masked LTC
  dynamics) + `NCPModel`.
- `example.py` -- trains on UCI Room Occupancy (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough, plus a plot of the wiring sparsity mask.

## Run it

```bash
pip install -e .
python models/ncp/example.py --device auto
# or open models/ncp/example.ipynb
```
