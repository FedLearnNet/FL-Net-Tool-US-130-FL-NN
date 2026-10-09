# US-130 Federated Readmission Prediction NN

Neural-network variant of `us-130-fl` for 30-day hospital readmission prediction.
The app keeps the same preprocessing, output schema and metric names as the
reference implementation so the existing analysis scripts can still compare:

- isolated single-site training via `local_val_*` client metrics
- FL-Net federated training via per-round `val_*` metrics
- centralized pooled training via `evaluation/train_central.py`

The model is a small scikit-learn `MLPClassifier` with a Kaggle-style diabetes
architecture by default: hidden layers `12,8`, ReLU, Adam, batch size `10`, and
10 training rounds with 15 local iterations each. Federated rounds exchange
weighted layer parameters only; raw rows remain local. The input layer is aligned
by feature name before aggregation, while hidden and output layers use the same
configured architecture on every client.

## Coordinator-only production runs

For an intentionally single-clinic production run, set this on the app container:

```text
FL_RUN__SINGLE_PARTICIPANT_MODE=true
```

The production run lifecycle and local training still execute normally, but the
coordinator uses each round's local model update unchanged. It does not wait for
the V2 relay's reserved client slot, perform a one-item aggregation, or broadcast
through the controller. Do not enable this setting for multi-clinic runs: the
framework validates the coordinator-only topology and fails instead of silently
ignoring real participants.

The global API must separately allow an experiment with one participant, for
example by configuring `flnet.federated-learning.participants.minamount=1` in
that deployment.
