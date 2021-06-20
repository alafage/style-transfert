from styletrf import StyleTRF

# Initialize the hyperparameters.
strf = StyleTRF(content_weight=1.0, style_weight=1e3)
print(f"Using device: {strf.device}")
# Load the content and the style images.
strf.fit(
    content="./styletrf/data/content_4.jpg",
    style="./styletrf/data/style_4.jpg",
)

# Generate the target image
strf.train(iterations=10000, out_path="test.jpg", save_every=50)

# Get the target image Torch Tensor.
# target_image = strf.target
