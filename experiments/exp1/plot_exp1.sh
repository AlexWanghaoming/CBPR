
#layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
switch_freqs=('inter2' 'inter1' 'intra200' 'intra100')
#switch_freqs=('inter2')

for layout in "${layouts[@]}"; do
  for sf in "${switch_freqs[@]}"; do
    python plot_exp1.py --switch_freq ${sf}
  done
done

