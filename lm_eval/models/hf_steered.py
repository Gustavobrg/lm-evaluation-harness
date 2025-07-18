from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union, List, Dict
import logging

import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_new_tokens: int = 1000
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None

@dataclass
class SteeringConfig:
    """Configuration for SAE feature steering."""
    feature_indices: List[int]
    strength: float = 2.0
    hook_layer: Optional[int] = None

@contextmanager
def steer(
    model: Union[PreTrainedModel, PeftModel], hook_to_steer: dict[str, Callable]
) -> Generator[None, Any, None]:
    """
    Context manager that temporarily hooks models and steers them.
    """
    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            if isinstance(output, tuple):
                output = (hook_to_steer[hookpoint](output[0]), *output[1:])
            else:
                output = hook_to_steer[hookpoint](output)
            return output
        return hook_fn

    handles = []
    hookpoints = list(hook_to_steer.keys())

    for name, module in model.named_modules():
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

@register_model("steered_best_of_n")
class SteeredBestOfNModel(HFLM):
    """HFLM with SAE feature steering and Best of N sampling."""

    def __init__(
        self,
        pretrained: str,
        sae_path: str,
        sae_component: str,
        reward_model_path: str = None,
        device: Optional[str] = None,
        steering_config: Optional[SteeringConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        clean_responses: bool = True,
        scoring_method: str = "reward_model",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        # Load SAE
        self._load_sae(sae_path, sae_component)

        # Load reward model if specified
        self.reward_model = None
        self.reward_tokenizer = None
        if reward_model_path and scoring_method == "reward_model":
            self._load_reward_model(reward_model_path)

        # Configuration
        self.steering_config = steering_config or SteeringConfig(
            feature_indices=[15136, 17456, 46379, 62777],
            strength=4.0
        )
        self.generation_config = generation_config or GenerationConfig()
        self.clean_responses = clean_responses
        self.scoring_method = scoring_method

        # Hook mapping for steering
        self.hook_to_steer = {}

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"SteeredBestOfNModel initialized with {len(self.steering_config.feature_indices)} features")

    def _load_sae(self, sae_path: str, sae_component: str) -> None:
        """Load the Sparse Autoencoder."""
        logger.info(f"Loading SAE: {sae_path}/{sae_component}")
        
        from sae_lens import SAE
        
        self.sae, self.sae_cfg_dict, _ = SAE.from_pretrained(sae_path, sae_component)
        self.sae = self.sae.to(self.device)

    def _load_reward_model(self, reward_model_path: str) -> None:
        """Load the reward model for scoring generations."""
        logger.info(f"Loading reward model: {reward_model_path}")

        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            device_map="auto"
        ).eval()

    def _create_steering_hook(self, feature_idx: int, strength: float):
        """Create a forward hook for SAE feature steering."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                resid = output[0].clone()
                other = output[1:]
            else:
                resid = output.clone()
                other = ()

            # Apply SAE steering
            features = self.sae.encode(resid)
            recon = self.sae.decode(features)
            error = resid - recon

            # Steer the feature
            features[..., feature_idx] = strength
            steered = self.sae.decode(features) + error
            steered = steered.to(resid.dtype)

            return (steered,) + other

        return hook_fn

    @contextmanager
    def _apply_steering_hook(self, feature_idx: int, strength: float):
        """Context manager for temporarily applying steering hook."""
        hook_fn = self._create_steering_hook(feature_idx, strength)
        
        # Get the correct layer based on SAE configuration
        hook_layer = getattr(self.sae.cfg, 'hook_layer', 19)
        target_layer = self.model.model.layers[hook_layer]
        
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    def _generate_with_feature(
        self,
        input_ids: torch.Tensor,
        feature_idx: int,
        strength: float,
        generation_config: GenerationConfig
    ) -> str:
        """Generate text with a specific feature steering."""
        with self._apply_steering_hook(feature_idx, strength):
            output = self.model.generate(
                input_ids,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _clean_response(self, text: str) -> str:
        """Clean the response by removing content before </think> tag."""
        end_tag_index = text.find("</think>")
        if end_tag_index != -1:
            return text[end_tag_index + len("</think>"):].lstrip()
        return text

    def _score_response(self, prompt: str, response: str) -> float:
        """Score a response using the specified scoring method."""
        if self.scoring_method == "reward_model" and self.reward_model is not None:
            return self._score_with_reward_model(prompt, response)
        elif self.scoring_method == "log_likelihood":
            return self._score_with_log_likelihood(prompt, response)
        elif self.scoring_method == "perplexity":
            return self._score_with_perplexity(prompt, response)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

    def _score_with_reward_model(self, prompt: str, response: str) -> float:
        """Score a response using the reward model."""
        full_input = f"Prompt: {prompt}\nResposta: {response}"

        with torch.inference_mode():
            reward_inputs = self.reward_tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            logits = self.reward_model(**reward_inputs).logits

            if logits.shape[-1] == 1:
                score = logits.item()
            elif logits.shape[-1] == 2:
                score = logits[:, 1].item()
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

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
            return token_log_probs.mean().item()

    def _score_with_perplexity(self, prompt: str, response: str) -> float:
        """Score a response using perplexity (negative log likelihood)."""
        log_likelihood = self._score_with_log_likelihood(prompt, response)
        return -log_likelihood

    def _best_of_n_generate(self, prompt_text: str, input_ids: torch.Tensor, **generation_kwargs) -> str:
        """Generate multiple candidates and return the best one."""
        # Generate responses for each feature
        candidates = []
        for feature_idx in self.steering_config.feature_indices:
            try:
                response = self._generate_with_feature(
                    input_ids, feature_idx, self.steering_config.strength, self.generation_config
                )

                if self.clean_responses:
                    response = self._clean_response(response)

                candidates.append((feature_idx, response))

            except Exception as e:
                logger.error(f"Failed to generate with feature {feature_idx}: {e}")
                continue

        if not candidates:
            return None

        # Score all candidates
        scored_candidates = []
        for feature_idx, response in candidates:
            try:
                score = self._score_response(prompt_text, response)
                scored_candidates.append({
                    'feature_idx': feature_idx,
                    'response': response,
                    'score': score
                })

            except Exception as e:
                logger.error(f"Failed to score response for feature {feature_idx}: {e}")
                continue

        if not scored_candidates:
            return None

        # Sort by score (descending) and return the best response
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        print("================ Best of N Candidates ===============")
        print(f"Best of N candidates: {scored_candidates}")
        return scored_candidates[0]['response']

    def forward(self, *args, **kwargs):
        """Standard forward pass - preserved interface."""
        return self.model.forward(*args, **kwargs)

    def _model_call(self, *args, **kwargs):
        """
        Standard model call interface - PRESERVED.
        This is used for loglikelihood calculations and other inference tasks.
        """
        return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        """
        Standard model generate interface - PRESERVED with Best of N enhancement.
        This is used for text generation tasks.
        """
        # Check if we should use Best of N
        if len(self.steering_config.feature_indices) > 1:
            # Extract input_ids from args
            input_ids = args[0] if args else kwargs.get('input_ids')
            if input_ids is not None:
                prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
                # Try Best of N generation
                best_response = self._best_of_n_generate(prompt_text, input_ids, **kwargs)
                
                if best_response is not None:
                    # Tokenize and return the best response
                    best_output = self.tokenizer.encode(best_response, return_tensors="pt").to(self.device)
                    return best_output

        # Fallback to standard generation
        return super()._model_generate(*args, **kwargs)
