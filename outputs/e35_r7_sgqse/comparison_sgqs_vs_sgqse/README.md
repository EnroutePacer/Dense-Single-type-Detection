# B1 SGQS versus B3 SGQS-E curves

Reproduce from the repository root with:

```powershell
python outputs/e35_r7_sgqse/comparison_sgqs_vs_sgqse/plot_sgqs_vs_sgqse.py
```

The script reads only the sibling B1/B3 user-attachment `console.log` files by
relative path. Input SHA-256 values are:

- B1: `686a2abbc98cd948b9be92f7fcec5613f567d37bdb9e154c4977f12ec0e32bf4`
- B3: `763b4c43883fe030fb6a7b0b9d634dc43096e100e85cc591657a830f6df0ad35`

The CSV and PNG outputs reproduce byte-for-byte with the recorded Python and
Matplotlib stack; PDF page content reproduces, while PDF creation time metadata
can differ. COCO console lines are rounded to three decimals, so these plots
cannot support the exact `+0.000078905207` B3-versus-B1 comparison. Exact
endpoint statistics must use the structured server `log.txt` files.
