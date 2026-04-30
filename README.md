# Influencer Labeling with PyG

This project is a small synthetic graph experiment I put together to keep the idea easy to understand.

The goal was simple:

- generate one social network
- mark a handful of users as influencers
- turn the graph into PyTorch Geometric data
- compare a basic heuristic with GraphSAGE and GCN

I wanted something I could explain without sounding like I built a giant platform.

## What’s in the graph

- `USER` nodes
- `TOPIC` nodes
- `COMMUNITY` nodes
- directed `FOLLOWS` edges
- directed `INTERACTION` edges
- user-to-topic and user-to-community edges
- binary influencer labels on a small subset of users

The demo graph is a little bigger than the first version, but it is still small enough to read through without getting lost.

## What the code does

- `influencer_lab/synthetic.py` builds the graph and labels
- `influencer_lab/features.py` turns the graph into node features
- `influencer_lab/pyg.py` converts everything into a PyG `Data` object
- `influencer_lab/model.py` trains GraphSAGE and GCN
- `influencer_lab/baselines.py` runs a follower-count baseline
- `influencer_lab/cli.py` ties the whole thing together

## Quick start

```python
from influencer_lab import (
    GCN,
    GraphSAGE,
    build_demo_benchmark,
    evaluate_degree_baseline,
    evaluate_gcn,
    evaluate_graphsage,
    to_pyg_data,
    train_gcn,
    train_graphsage,
)

benchmark = build_demo_benchmark(seed=7)
data = to_pyg_data(benchmark)

graphsage = GraphSAGE(in_channels=data.x.size(1))
gcn = GCN(in_channels=data.x.size(1))

train_graphsage(graphsage, data)
train_gcn(gcn, data)

print(evaluate_graphsage(graphsage, data, data.test_mask & data.user_mask))
print(evaluate_gcn(gcn, data, data.test_mask & data.user_mask))
print(evaluate_degree_baseline(data))
```

## What I liked about this version

I kept it intentionally small and fairly human:

- one synthetic graph
- one task
- one simple baseline
- two common GNNs
- no experiment framework, no training platform, no extra moving parts

That makes it easier to understand, easier to explain, and easier to extend later if I want to.

## Running it

```bash
python3 examples/run_demo.py
python3 -m pytest lab_tests
```

There is also a comparison figure in `docs/influencer_comparison.png`

