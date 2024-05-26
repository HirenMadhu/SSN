#  [ICML 2024]- Unsupervised Parameter-free Simplicial Representation Learning with Scattering Transforms
Code for <tt>SSN</tt> model proposed in the ICML 2024 submission.

## Dependencies

- Python 3.9
- PyTorch 2.0
- dgl 1.0.2.cu113
- gudhi 3.8.0


## Datasets

```
Node classification: 'contact-high-school', 'contact-primary-school' and 'senate-bills'.

Simplicial closure: 'contact-high-school', 'contact-primary-school' and 'email-Enron'.

Simplex prediction: 'madison-restaurant-reviews', 'algebra-questions' and 'geometry-questions'.

Graph classification: 'Proteins', 'NCI1', 'IMDB-B', 'REDDIT-B', and 'REDDIT-M5'.

Trajectory prediction: 'Ocean drifters', 'Mesh' and 'Synthetic'.

```

## Usage
To run the codes, use the following commands:
```python
# Node classification example

python node_classification.py -- data contact-high-school --gpu 0 --dim 3, --J 4, --split 0.2 --include_boundary True

# Simplicial closure example
python simplicial_closure.py -- data contact-high-school --gpu 0 --dim 3, --J 4, --split 0.2 --include_boundary True

# Graph classification example
python graph_classification.py --data proteins --gpu 0 --J 2 --include_boundary True

# Trajectory prediction example
python trajectory_prediction.py --data syn --J 4 --gpu 0

# Simplex prediction example
python simplex_prediction.py -- data madison-restaurant-reviews --gpu 0 --dim 3, --J 4, --split 0.2 --include_boundary True
```