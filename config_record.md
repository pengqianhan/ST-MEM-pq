## main_pretrain.py
### model config
Model = ST_MEM(
    seq_len=2250,
    patch_size=75,
    num_leads=12,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=4,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06),
    norm_pix_loss=True)

### model input shape
    print('samples.shape:',samples.shape)##torch.Size([batch_size=16, 12, 2250])
    results = model(samples)
