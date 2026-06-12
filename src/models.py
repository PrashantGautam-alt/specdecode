import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelLoader:
    """
    Loads a HuggingFace language model and its tokenizer onto the GPU.
    Exists so we have one clean place to handle model loading for both
    the draft model and target model in speculative decoding.
    """

    def __init__(self, model_name: str):
        """
        model_name: HuggingFace model ID, e.g. 'meta-llama/Llama-3.2-1B'
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        Downloads (if needed) and loads the model and tokenizer into GPU memory.
        """
        print(f"Loading {self.model_name} onto {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # half precision saves VRAM without much quality loss
            device_map=self.device,
        )

        self.model.eval()  # disables dropout; we're doing inference not training

        print(f"Loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        return self
