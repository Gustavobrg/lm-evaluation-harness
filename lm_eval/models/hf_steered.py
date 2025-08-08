from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@contextmanager
def steer(
    model: Union[PreTrainedModel, PeftModel], hook_to_steer: dict[str, Callable]
) -> Generator[None, Any, None]:
    """
    Context manager that temporarily hooks models and steers them.

    Args:
        model: The transformer model to hook
        hook_to_steer: Dictionary mapping hookpoints to steering functions

    Yields:
        None
    """

    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            # If output is a tuple (like in some transformer layers), take first element
            if isinstance(output, tuple):
                output = (hook_to_steer[hookpoint](output[0]), *output[1:])  # type: ignore
            else:
                output = hook_to_steer[hookpoint](output)

            return output

        return hook_fn

    handles = []
    hookpoints = list(hook_to_steer.keys())

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name))
            handles.append(handle)

    if len(handles) != len(hookpoints):
        raise ValueError(f"Not all hookpoints could be resolved: {hookpoints}")

    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


@register_model("steered2")
class SteeredModel(HFLM):
    hook_to_steer: dict[str, Callable]

    def __init__(
        self,
        pretrained: str,
        steer_path: str,
        reward_model_path: str = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        HFLM with a steered forward pass.

        To derive steering vectors from a sparse model loadable with sparsify or sae_lens,
        provide the path to a CSV file with the following columns (example rows are provided below):

        loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,sae_id,description,
        sparsify,add,EleutherAI/sae-pythia-70m-32k,layers.3,30,10.0,,,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,240.0,layer_20/width_16k/canonical,increase dogs,

        To load steering vectors directly, provide the path to a pytorch (.pt) file with content in the following format:

        {
            hookpoint: {
                "steering_vector": <torch.Tensor>,
                "steering_coefficient": <float>,
                "action": <Literal["add", "clamp"]>,
                "bias": <torch.Tensor | None>,
            },
            ...
        }
        """
        super().__init__(pretrained=pretrained, device=device, **kwargs)
        

        if steer_path.endswith(".pt") or steer_path.endswith(".pth"):
            with open(steer_path, "rb") as f:
                steer_config: dict[str, dict[str, Any]] = torch.load(
                    f, weights_only=True
                )
        elif steer_path.endswith(".csv"):
            steer_config = self.derive_steer_config(steer_path)
        else:
            raise ValueError(f"Unknown steer file type: {steer_path}")

        # pegamos o primeiro hookpoint e sua lista de infos (lista de dicts)
        self.hookpoint = list(steer_config.keys())[0]
        self.original_steer_info_list = steer_config[self.hookpoint]  # lista de dicts

        # iniciamos vazio, vai ser preenchido no _model_generate
        self.hook_to_steer = {}

        self._load_reward_model(reward_model_path)


    def _load_reward_model(self, reward_model_path: str) -> None:
        """Load the reward model for scoring generations."""

        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            device_map="auto"
        ).eval()

    @classmethod
    def derive_steer_config(cls, steer_path: str):
        import pandas as pd
    
        df = pd.read_csv(steer_path)
        steer_data: dict[str, list[dict[str, Any]]] = {}
    
        if any(df["loader"] == "sparsify"):
            from sparsify import SparseCoder
        if any(df["loader"] == "sae_lens"):
            from sae_lens import SAE
    
            sae_cache = {}
    
            def load_from_sae_lens(sae_release: str, sae_id: str):
                cache_key = (sae_release, sae_id)
                if cache_key not in sae_cache:
                    sae_cache[cache_key] = SAE.from_pretrained(sae_release, sae_id)[0]
                return sae_cache[cache_key]
    
        for _, row in df.iterrows():
            action = row.get("action", "add")
            sparse_name = row["sparse_model"]
            hookpoint = row["hookpoint"]
            feature_index = int(row["feature_index"])
            steering_coefficient = float(row["steering_coefficient"])
            loader = row.get("loader", "sparsify")
    
            if loader == "sparsify":
                name_path = Path(sparse_name)
                sparse_coder = (
                    SparseCoder.load_from_disk(name_path / hookpoint)
                    if name_path.exists()
                    else SparseCoder.load_from_hub(sparse_name, hookpoint)
                )
                assert sparse_coder.W_dec is not None
                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec
    
            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=row["sae_id"]
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")
    
            info = {
                "action": action,
                "steering_coefficient": steering_coefficient,
                "steering_vector": steering_vector,
                "bias": bias,
            }
    
            if hookpoint not in steer_data:
                steer_data[hookpoint] = []
            steer_data[hookpoint].append(info)
    
        return steer_data

    @classmethod
    def clamp(
        cls,
        acts: Tensor,
        steering_vector: Tensor,
        value: float,
        bias: Optional[Tensor] = None,
    ):
        """Clamps a direction of the activations to be the steering vector * the value.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, features]
            steering_vector (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to
            bias (Tensor | None): Optional bias to add to the activations

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """

        if bias is not None:
            acts = acts - bias

        direction = steering_vector / torch.norm(steering_vector)
        proj_magnitude = torch.sum(acts * direction, dim=-1, keepdim=True)
        orthogonal_component = acts - proj_magnitude * direction

        clamped = orthogonal_component + direction * value

        if bias is not None:
            return clamped + bias

        return clamped

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            with steer(self.model, self.hook_to_steer):
                return self.model.forward(*args, **kwargs)

    def _model_call(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        results = []
        decoded_results = []

        context = kwargs.get("context", args[0] if len(args) > 0 else None)
        prompt_text = self.tokenizer.decode(context[0], skip_special_tokens=True)
        
        for steer_info in self.original_steer_info_list:
            action = steer_info["action"]
            coef = steer_info["steering_coefficient"]
            vec = steer_info["steering_vector"].to(self.device).to(self.model.dtype)
            bias = (
                steer_info["bias"].to(self.device).to(self.model.dtype)
                if steer_info["bias"] is not None
                else None
            )
    
            if action == "add":
                self.hook_to_steer = {
                    self.hookpoint: lambda acts, vec=vec: acts + coef * vec
                }
            elif action == "clamp":
                self.hook_to_steer = {
                    self.hookpoint: partial(self.clamp, steering_vector=vec, value=coef, bias=bias)
                }
            else:
                raise ValueError(f"Ação desconhecida: {action}")
    
            with torch.no_grad():
                with steer(self.model, self.hook_to_steer):
                    out = super()._model_generate(*args, **kwargs)
                    results.append(out)

                    # Decodifica a geração
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    decoded_results.append(decoded)

        scores = []
        for decoded in decoded_results:
            print(f"Scoring decoded: {decoded}")
            score = self._score_with_reward_model(decoded)
            print(f"Score: {score:.4f}")
            scores.append(score)
        best_index = scores.index(max(scores))
        print(f"Best decoded (score {scores[best_index]:.4f}): {decoded_results[best_index]}")
    
        return results[best_index]

    def strip_prompt_from_generation(self, prompt: str, generation: str) -> str:
        prompt = prompt.strip()
        generation = generation.strip()
        
        # Se a geração começa com o prompt, remove essa parte
        if generation.startswith(prompt):
            return generation[len(prompt):].strip()
        
        # Se o prompt está parcialmente incluso no início, tenta cortar a maior parte possível
        for i in range(len(prompt), 0, -1):
            if generation.startswith(prompt[:i]):
                return generation[i:].strip()
        
        return generation
    
    def format_ranking_prompt(self, prompt: str, generations: list[str]) -> str:
        prompt_rank = (
            f"Original prompt:\n\n{prompt.strip()}\n\n"
            "Below are several responses generated for the prompt above, each marked with a number in brackets ([number]).\n"
            "Your task is to carefully review all alternatives and select the one that most correctly and appropriately fulfills the requirements of the original prompt by providing the correct answer.\n\n"
        )
        for i, gen in enumerate(generations, 1):
            cleaned = self.strip_prompt_from_generation(prompt, gen)
            prompt_rank += f"[{i}]: {cleaned}\n\n"
        prompt_rank += (
            "Please respond with only the number of the most suitable alternative. "
            "Do not include any explanations, justification, or additional text—your answer should be the number only.\n"
            "Your selected alternative:"
        )
        return prompt_rank
    
    def select_best_generation(self, ranking_prompt: str, generations: list[str]) -> int:
        inputs = self.tokenizer(ranking_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrai índice (pode melhorar com regex)
        for i in range(len(generations)):
            if str(i+1) in response:
                return i
        return 0  # fallback se não identificar

    def _score_with_reward_model(self, prompt: str) -> float:
        """Score a response using the reward model."""
        full_input = f"{prompt}"

        with torch.inference_mode():
            reward_inputs = self.reward_tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            logits = self.reward_model(**reward_inputs).logits
            print(f"Logits shape: {logits}")
            
            score = (logits[1] - logits[0]).item()

            return score
    
    def _score_with_log_likelihood(self, prompt: str, response: str) -> float:
        """Score a response using log likelihood."""
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            prompt_length = len(self.tokenizer.encode(prompt))
            response_log_probs = log_probs[0, prompt_length-1:-1, :]
            response_tokens = input_ids[0, prompt_length:]
            
            token_log_probs = torch.gather(response_log_probs, 1, response_tokens.unsqueeze(1))
            return -token_log_probs.mean().item()