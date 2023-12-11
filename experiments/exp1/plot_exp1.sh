
#layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment' 'coordination_ring')
layouts=('marshmallow_experiment' 'asymmetric_advantages')
#switch_freqs=('inter2' 'inter1' 'intra200' 'intra100')
switch_freqs=('inter2')

for layout in "${layouts[@]}"; do
  for sf in "${switch_freqs[@]}"; do
    python plot_exp1.py --layout ${layout} --switch_freq ${sf}
  done
done

