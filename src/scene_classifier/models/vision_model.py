from transformers import AutoModelForImageClassification, PreTrainedModel

from scene_classifier.utils.model_config import ModelConfig

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# inputs = processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])




def build_model(model_cfg: ModelConfig, classes: list) -> PreTrainedModel:
    """ 
    Return a pretrained model with a new head for our classes.
    """    
    model = AutoModelForImageClassification.from_pretrained(
                model_cfg.checkpoint,
                id2label = {i: name for i, name in enumerate(classes)},
                label2id= {name: i for i, name in enumerate(classes)},
                ignore_mismatched_sizes=True,
            )
    
    return model
