from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union, List, Dict
import logging

import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
from sae_lens import SAE

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


@contextmanager
def steer_with_sae_features(
    model: Union[PreTrainedModel, PeftModel], 
    sae: SAE,
    feature_indices: List[int],
    strength: float,
    hook_layer: int
) -> Generator[None, Any, None]:
    """
    Context manager that applies SAE feature steering to a specific layer.
    
    Args:
        model: The transformer model to hook
        sae: SAE model for feature steering
        feature_indices: List of feature indices to activate
        strength: Steering strength
        hook_layer: Layer to apply steering to
    """
    
    def create_sae_hook(feature_idx: int):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            if isinstance(output, tuple):
                resid = output[0].clone()
                other = output[1:]
            else:
                resid = output.clone()
                other = ()

            # Apply SAE steering for this specific feature
            features = sae.encode(resid)
            recon = sae.decode(features)
            error = resid - recon

            # Steer the specific feature
            features[..., feature_idx] = strength
            steered = sae.decode(features) + error
            steered = steered.to(resid.dtype)

            return (steered,) + other if isinstance(output, tuple) else steered

        return hook_fn

    # Apply hook to the specified layer
    handles = []
    target_layer = None
    
    # Find the target layer
    for name, module in model.named_modules():
        if name == f"model.layers.{hook_layer}":
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Could not find layer at index {hook_layer}")
    
    # For multiple features, we'll test each one individually
    # This hook will be set up for a single feature at a time
    current_feature_idx = feature_indices[0] if feature_indices else 0
    hook_fn = create_sae_hook(current_feature_idx)
    handle = target_layer.register_forward_hook(hook_fn)
    handles.append(handle)
    
    try:
        yield current_feature_idx
    finally:
        for handle in handles:
            handle.remove()


@register_model("best_of_n_sae_steered")
class BestOfNSAESteeredModel(HFLM):
    """
    HFLM with Best-of-N sampling using SAE feature steering.
    Tests different SAE features and selects the best response using a reward model.
    """

    def __init__(
        self,
        pretrained: str,
        sae_path: str,
        sae_component: str,
        reward_model_path: str,
        feature_indices: List[int],
        steering_strength: float = 2.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        HFLM with Best-of-N sampling using SAE feature steering.

        Args:
            pretrained: Path to the base model
            sae_path: Path/name of the SAE model
            sae_component: SAE component (e.g., "blocks.19.hook_resid_post")
            reward_model_path: Path to reward model for scoring
            feature_indices: List of SAE feature indices to test
            steering_strength: Strength of feature steering
            device: Device to use
            **kwargs: Additional arguments for HFLM
        """
        # Remover argumentos específicos dos kwargs
        kwargs.pop('feature_indices', None)
        kwargs.pop('steering_strength', None)
        
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        # Load SAE
        logger.info(f"Loading SAE: {sae_path}/{sae_component}")
        self.sae, self.sae_cfg_dict, _ = SAE.from_pretrained(sae_path, sae_component)
        self.sae = self.sae.to(self.device)
        
        # Extract hook layer from SAE config
        self.hook_layer = self.sae.cfg.hook_layer if hasattr(self.sae.cfg, 'hook_layer') else 19

        # Load reward model
        logger.info(f"Loading reward model: {reward_model_path}")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, device_map="auto"
        ).eval()
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

        # SAE steering configuration
        self.feature_indices = feature_indices
        self.steering_strength = steering_strength
        
        logger.info(f"Initialized Best-of-N SAE Steered Model")
        logger.info(f"Feature indices: {feature_indices}")
        logger.info(f"Steering strength: {steering_strength}")
        logger.info(f"Hook layer: {self.hook_layer}")

    @classmethod
    def from_pretrained_models(
        cls,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        sae: SAE,
        reward_model: AutoModelForSequenceClassification,
        reward_tokenizer: AutoTokenizer,
        feature_indices: List[int],
        steering_strength: float = 2.0,
        device: Optional[str] = None,
    ):
        """
        Create instance from pre-loaded models (for direct usage).
        
        Args:
            model: Pre-loaded base model
            tokenizer: Pre-loaded tokenizer
            sae: Pre-loaded SAE model
            reward_model: Pre-loaded reward model
            reward_tokenizer: Pre-loaded reward tokenizer
            feature_indices: List of SAE feature indices to test
            steering_strength: Strength of feature steering
            device: Device to use
        """
        # Create a dummy instance first
        instance = cls.__new__(cls)
        
        # Manually set the required attributes
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = device or str(model.device)
        instance._device = instance.device
        
        # SAE setup
        instance.sae = sae.to(instance.device)
        instance.sae_cfg_dict = sae.cfg.__dict__ if hasattr(sae, 'cfg') else {}
        instance.hook_layer = sae.cfg.hook_layer if hasattr(sae.cfg, 'hook_layer') else 19
        
        # Reward model setup
        instance.reward_model = reward_model
        instance.reward_tokenizer = reward_tokenizer
        
        # Steering configuration
        instance.feature_indices = feature_indices
        instance.steering_strength = steering_strength
        
        logger.info(f"Created Best-of-N SAE Steered Model from pre-loaded models")
        logger.info(f"Feature indices: {feature_indices}")
        logger.info(f"Steering strength: {steering_strength}")
        
        return instance

    def _score_response(self, prompt: str, response: str) -> float:
        """Score a response using the reward model."""
        # Clean response if needed (remove content before </think>)
        end_tag_index = response.find("</think>")
        if end_tag_index != -1:
            response = response[end_tag_index + len("</think>"):].lstrip()
        
        full_input = f"Prompt: {prompt}\nResponse: {response}"

        with torch.inference_mode():
            reward_inputs = self.reward_tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            logits = self.reward_model(**reward_inputs).logits

            if logits.shape[-1] == 1:
                score = logits.item()
            elif logits.shape[-1] == 2:
                score = logits[:, 1].item()
            else:
                score = torch.max(torch.softmax(logits, dim=-1)).item()

            return score

    def _generate_with_feature(
        self, 
        input_ids: torch.Tensor, 
        feature_idx: int, 
        **generation_kwargs
    ) -> str:
        """Generate text with a specific SAE feature activated."""
        with steer_with_sae_features(
            self.model, 
            self.sae, 
            [feature_idx], 
            self.steering_strength, 
            self.hook_layer
        ):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    **generation_kwargs,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                output[0][len(input_ids[0]):], 
                skip_special_tokens=True
            )
            
            return response

    def _generate_candidates(self, input_ids: torch.Tensor, **generation_kwargs) -> List[Dict[str, Any]]:
        """Generate multiple candidates using different SAE features."""
        candidates = []
        
        for feature_idx in self.feature_indices:
            try:
                response = self._generate_with_feature(input_ids, feature_idx, **generation_kwargs)
                
                candidates.append({
                    'response': response,
                    'feature_idx': feature_idx,
                })
                
                logger.debug(f"Generated candidate with feature {feature_idx}")
                
            except Exception as e:
                logger.warning(f"Failed to generate candidate with feature {feature_idx}: {e}")
                continue
        
        return candidates

    def _model_generate(self, input_ids: torch.Tensor, **generation_kwargs):
        """Override generation to use Best-of-N with SAE features."""
        if len(self.feature_indices) > 1:
            # Generate multiple candidates with different features
            candidates = self._generate_candidates(input_ids, **generation_kwargs)
            
            if not candidates:
                logger.warning("No candidates generated, falling back to regular generation")
                return super()._model_generate(input_ids, **generation_kwargs)
            
            # Score candidates
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            scored_candidates = []
            
            for candidate in candidates:
                try:
                    score = self._score_response(input_text, candidate['response'])
                    scored_candidates.append({
                        **candidate,
                        'score': score
                    })
                    logger.debug(f"Feature {candidate['feature_idx']} scored: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to score candidate with feature {candidate['feature_idx']}: {e}")
                    continue
            
            if not scored_candidates:
                logger.warning("No candidates scored, using first candidate")
                best_response = candidates[0]['response']
                best_feature = candidates[0]['feature_idx']
            else:
                # Select best candidate
                best_candidate = max(scored_candidates, key=lambda x: x['score'])
                best_response = best_candidate['response']
                best_feature = best_candidate['feature_idx']
                logger.info(f"Selected feature {best_feature} with score {best_candidate['score']:.4f}")
            
            # Return in the expected format
            response_ids = self.tokenizer.encode(best_response, return_tensors="pt").to(self.device)
            full_output = torch.cat([input_ids, response_ids], dim=-1)
            return full_output
        
        else:
            # Single feature - use regular generation with that feature
            feature_idx = self.feature_indices[0] if self.feature_indices else 0
            response = self._generate_with_feature(input_ids, feature_idx, **generation_kwargs)
            response_ids = self.tokenizer.encode(response, return_tensors="pt").to(self.device)
            full_output = torch.cat([input_ids, response_ids], dim=-1)
            return full_output

    def _model_call(self, input_ids: torch.Tensor, **kwargs):
        """Override model call for likelihood computation (use first feature)."""
        if self.feature_indices:
            feature_idx = self.feature_indices[0]
            with steer_with_sae_features(
                self.model, 
                self.sae, 
                [feature_idx], 
                self.steering_strength, 
                self.hook_layer
            ):
                return super()._model_call(input_ids, **kwargs)
        else:
            return super()._model_call(input_ids, **kwargs)

    def generate_best_of_n(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        do_sample: bool = True,
        return_all_candidates: bool = False,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Public method for generating with Best-of-N (for direct usage).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            return_all_candidates: Whether to return all candidates
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Dictionary with best response and optionally all candidates
        """
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            **generation_kwargs
        }
        
        # Generate candidates
        candidates = self._generate_candidates(input_ids, **gen_kwargs)
        
        if not candidates:
            raise RuntimeError("No candidates were generated successfully")
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            try:
                score = self._score_response(prompt, candidate['response'])
                scored_candidates.append({
                    **candidate,
                    'score': score
                })
            except Exception as e:
                logger.error(f"Failed to score candidate with feature {candidate['feature_idx']}: {e}")
                continue
        
        if not scored_candidates:
            raise RuntimeError("No candidates were scored successfully")
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare result
        result = {
            'best_response': scored_candidates[0]['response'],
            'best_score': scored_candidates[0]['score'],
            'best_feature_idx': scored_candidates[0]['feature_idx']
        }
        
        if return_all_candidates:
            result['all_candidates'] = scored_candidates
        
        logger.info(f"Best candidate: Feature {result['best_feature_idx']} "
                   f"(Score: {result['best_score']:.4f})")
        
        return result