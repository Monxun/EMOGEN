import torch
from dalle2_pytorch import DALLE2, Unet, Decoder, CLIP, DecoderTrainer

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# mock data

text = torch.randint(0, 49408, (32, 256)).cuda()
images = torch.randn(32, 3, 256, 256).cuda()

# decoder (with unet)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    text_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    cond_on_text_encodings = True,
).cuda()

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16),
).cuda()

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 1000
).cuda()

decoder_trainer = DecoderTrainer(
    decoder,
    lr = 3e-4,
    wd = 1e-2,
    ema_beta = 0.99,
    ema_update_after_step = 1000,
    ema_update_every = 10,
)

for unet_number in (1, 2):
    loss = decoder_trainer(
        images,
        text = text,
        unet_number = unet_number, # which unet to train on
        max_batch_size = 4         # gradient accumulation - this sets the maximum batch size in which to do forward and backwards pass - for this example 32 / 4 == 8 times
    )

    decoder_trainer.update(unet_number) # update the specific unet as well as its exponential moving average

# after much training
# you can sample from the exponentially moving averaged unets as so

mock_image_embed = torch.randn(32, 512).cuda()
images = decoder_trainer.sample(image_embed = mock_image_embed, text = text) # (4, 3, 256, 256)