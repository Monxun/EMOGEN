import torch
from dalle2_pytorch import Unet, Decoder, DecoderTrainer

# unet for the cascading ddpm

unet1 = Unet(
    dim = 128,
    dim_mults=(1, 2, 4, 8)
).cuda()

unet2 = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

# decoder, which contains the unets

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (256, 512),  # first unet up to 256px, then second to 512px
    timesteps = 1000,
    unconditional = True
).cuda()

# decoder trainer

decoder_trainer = DecoderTrainer(decoder)

# images (get a lot of this)

images = torch.randn(1, 3, 512, 512).cuda()

# feed images into decoder

for i in (1, 2):
    loss = decoder_trainer(images, unet_number = i)
    decoder_trainer.update(unet_number = i)

# do the above for many many many many images
# then it will learn to generate images

images = decoder_trainer.sample(batch_size = 36, max_batch_size = 4) # (36, 3, 512, 512)