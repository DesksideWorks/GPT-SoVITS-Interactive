## Backbones quick introduction


### dit.py
- adaln-zero dit
- embedded timestep as condition
- concatted noised_input + masked_cond + embedded_text, linear proj in
- possible abs pos emb & convnextv2 blocks for embedded text before concat
- possible long skip connection (first layer to last layer)
