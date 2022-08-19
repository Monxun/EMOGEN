from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader

# Create a dataloader directly.
dataloader = create_image_embedding_dataloader(
    tar_url="/path/or/url/to/webdataset/{0000..9999}.tar", # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    embeddings_url="path/or/url/to/embeddings/folder",     # Included if .npy files are not in webdataset. Left out or set to None otherwise
    num_workers=4,
    batch_size=32,
    shard_width=4,                                         # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
    shuffle_num=200,                                       # Does a shuffle of the data with a buffer size of 200
    shuffle_shards=True,                                   # Shuffle the order the shards are read in
    resample_shards=False,                                 # Sample shards with replacement. If true, an epoch will be infinite unless stopped manually
)
for img, emb in dataloader:
    print(img.shape)  # torch.Size([32, 3, 256, 256])
    print(emb.shape)  # torch.Size([32, 512])
    # Train decoder only as shown above

# Or create a dataset without a loader so you can configure it manually
dataset = ImageEmbeddingDataset(
    urls="/path/or/url/to/webdataset/{0000..9999}.tar",
    embedding_folder_url="path/or/url/to/embeddings/folder",
    shard_width=4,
    shuffle_shards=True,
    resample=False
)