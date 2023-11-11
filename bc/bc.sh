layouts=('cramped_room' 'asymmetric_advantages' 'marshmallow_experiment')

for layout in "${layouts[@]}"; do
  python bc_hh.py --layout ${layout} --epochs 120 --lr 0.001
done

