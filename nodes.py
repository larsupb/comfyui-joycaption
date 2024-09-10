import numpy as np
import torch
import torch.amp.autocast_mode
import os
import requests
import gc
from transformers import (AutoModel, AutoProcessor, AutoTokenizer,
                          PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM)
from pathlib import Path
from PIL import Image

class JoyCaptioning:
    def __init__(self):
        self.CLIP_PATH = "google/siglip-so400m-patch14-384"
        self.LLM_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"  # "
        self.CHECKPOINT_PATH = os.path.join(Path(__file__).parent, Path("models/joycaption/wpkklhc6"))

        self.image_adapter = None
        self.clip_model = None
        self.clip_processor = None
        self.tokenizer = None
        self.text_model = None

        self.acquire_model()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_device": (["cuda", "cpu"], {"default": "cuda"}),
                "clip_device": (["cuda", "cpu"], {"default": "cpu"}),
                "instruction": ("STRING", {"default": "A descriptive caption for this image", "multiline": True}),
            },
        }

    RETURN_TYPES = ("String",)
    FUNCTION = "generate_joycaption"
    CATEGORY = "Tagging"

    def generate_joycaption(self, image: torch.tensor, llm_device='cuda', clip_device='cpu',
                            instruction="A descriptive caption for this image"):
        torch.cuda.empty_cache()

        self.load_model(llm_device, clip_device)

        # Convert the Tensor to a PIL image
        image_np = image.numpy().squeeze()  # Remove the first dimension (batch size of 1)
        # Convert the numpy array back to the original range (0-255) and data type (uint8)
        image_np = (image_np * 255).astype(np.uint8)
        # Create a PIL image from the numpy array
        image = Image.fromarray(image_np, mode="RGB")

        # resize image
        image = resize_image(image)

        # Tokenize the prompt
        prompt = self.tokenizer.encode(instruction + ":\n", return_tensors='pt',
                                       padding=False, truncation=False, add_special_tokens=False)

        # Preprocess image
        image = self.clip_processor(images=image, return_tensors='pt').pixel_values

        # Embed image
        vision_outputs = self.clip_model(pixel_values=image, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2].to('cuda')
        embedded_images = self.image_adapter(image_features)
        embedded_images = embedded_images.to('cuda')

        # Embed prompt
        prompt_embeds = self.text_model.model.embed_tokens(prompt.to('cuda'))
        assert prompt_embeds.shape == (1, prompt.shape[1],
                                       self.text_model.config.hidden_size), \
            f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], self.text_model.config.hidden_size)}"
        embedded_bos = self.text_model.model.embed_tokens(
            torch.tensor([[self.tokenizer.bos_token_id]], device=self.text_model.device, dtype=torch.int64))

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                max_new_tokens=255, do_sample=True, top_k=10, temperature=0.5,
                                                suppress_tokens=None)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)[0]

        self.cleanup()

        return (caption.strip(),)

    def load_model(self, llm_device, clip_device):
        # LLM
        print("Loading LLM")
        self.text_model = AutoModelForCausalLM.from_pretrained(self.LLM_ID, device_map=llm_device)
        self.text_model.eval()
        # LLM Tokenizer
        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_ID, device_map=llm_device, use_fast=False)
        assert (isinstance(self.tokenizer, PreTrainedTokenizer) or
                isinstance(self.tokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(self.tokenizer)}"

        # Load CLIP
        print("Loading CLIP")
        self.clip_processor = AutoProcessor.from_pretrained(self.CLIP_PATH, device_map=clip_device)
        self.clip_model = AutoModel.from_pretrained(self.CLIP_PATH, device_map=clip_device)
        self.clip_model = self.clip_model.vision_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        # Image Adapter
        print("Loading image adapter")
        self.acquire_model()
        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.text_model.config.hidden_size)
        self.image_adapter.load_state_dict(torch.load(os.path.join(self.CHECKPOINT_PATH, "image_adapter.pt")))
        self.image_adapter.eval()
        self.image_adapter.to("cuda")

    def acquire_model(self):
        if (not os.path.exists(self.CHECKPOINT_PATH) or
                not os.path.exists(os.path.join(self.CHECKPOINT_PATH, "image_adapter.pt"))):
            os.makedirs(self.CHECKPOINT_PATH)
            # download the model and its config with requests
            url = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"
            download(url, os.path.join(self.CHECKPOINT_PATH, "image_adapter.pt"))

    def cleanup(self):
        self.text_model.cpu()
        self.clip_model.cpu()
        self.image_adapter.cpu()
        del self.text_model
        del self.tokenizer
        del self.clip_model
        del self.clip_processor
        del self.image_adapter
        gc.collect()
        torch.cuda.empty_cache()


class ImageAdapter(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_features, output_features)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def resize_image(img, max=768):
    # Get the current dimensions of the image
    width, height = img.size

    # Determine the scaling factor to ensure the longest side is 1024 pixels
    if width > height:
        new_width = max
        new_height = int((max / width) * height)
    else:
        new_height = max
        new_width = int((max / height) * width)

    # Resize the image
    img = img.resize((new_width, new_height))
    return img


def download(url, file_path):
    r = requests.get(url)
    # write file to disk
    with open(file_path, "wb") as f:
        f.write(r.content)