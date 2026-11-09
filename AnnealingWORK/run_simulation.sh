#!/bin/bash
source /home/francesco/data/francesco_true/PycharmProjects/projet_one-main/.venv/bin/activate

for i in {5..9}; do
    python3 simulation.py "$i"
done
