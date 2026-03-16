from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    "openai/clip-vit-large-patch14",
    export=True
)

model.save_pretrained("clip_text_onnx")

print("Export finished.")