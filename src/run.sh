tau=50

for i in {0..9}; do
    #python3 simulation_dimension.py "$tau" "$i"
    python3 finalSimulation.py "$tau" "$i"
done