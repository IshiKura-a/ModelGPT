seed=(2022 2023 2025 2026)
for s in "${seed[@]}"
do
  python -m baseline.img_cls \
    --seed ${s} \
    --load_ckpt
done