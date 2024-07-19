

## run main_pretrain.py
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
in 'engine_pretrain.py'

    print('samples.shape:',samples.shape)##torch.Size([batch_size=16, 12, 2250])
    results = model(samples)


for the target sample rate is 250Hz, so the input is 9s

### resample
in 'dataset.py'

    self.resample = T.Resample(target_fs=target_fs) if fs_list is not None else None

in 'transforms.py'
            from scipy.signal import butter, resample, sosfiltfilt, square
            # print(f"Resample from {fs} to {self.target_fs}.")##Resample from 500 to 250.
            # print(f"Original shape: {x.shape}")##Original shape: (12, 5000)
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
            # print(f"Resampled shape: {x.shape}")##Resampled shape: (12, 2500)