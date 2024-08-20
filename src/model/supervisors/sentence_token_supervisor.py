
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any, List, Optional
import einops
from omegaconf import MISSING
from model.components.contrastive_losses import global_contrastive_loss, sentence_contrastive_loss, sentence_mse_loss
from torch import FloatTensor, nn

from torch import BoolTensor
import torch
from model.components.pooling import GlobalAvgPool
from model.detector.token_decoder_detector import TokenDetectorOutput
from model.img_encoder import ImageEncoderOutput
from model.txt_encoder import TextEncoderOutput

from util.model_utils import prepare_config

log = logging.getLogger(__name__)


@dataclass
class SentenceTokenConfig:
    # --- Loss: Global contrastive CLIP-style (global-contr) ---
    use_global_contr: bool = False
    coeff_global_contr: float = 0.0
    global_contr_temp: float = 0.2
 
    # --- Loss: Sentence-level contrastive (sent-contr) ---
    use_sent_contr: bool = False
    coeff_sent_contr: float = 0.0
    sent_contr_temp: float = 0.25
    cross_sample_sentence_negatives: bool = True
    lambda_sent2reg: float = 0.0
    lambda_reg2sent: float = 1.0

    # --- Loss: Sentence MSE (sent-mse) ---
    use_sent_mse: bool = False
    coeff_sent_mse: float = 0.0

    # --- Loss: Sentence Generation (sent-gen) ---
    use_sent_gen: bool = False
    coeff_sent_gen: float = 0.0
    
    no_skip_roi_pool: bool = False


class SentenceTokenSupervisor(nn.Module):
    def __init__(self, config: SentenceTokenConfig, main_config):
        super().__init__()
        self.config: SentenceTokenConfig = prepare_config(config, SentenceTokenConfig, log)

        self.requires_sentence_tokens = True
        self.requires_anatomy_tokens = False
        self.requires_pathology_tokens = False
        self.requires_region_pathology_tokens = False

        self.aggregator = GlobalAvgPool()
    
    def forward(self, model: 'ChEX', 
                encoded_image: ImageEncoderOutput, 
                encoded_sentences: TextEncoderOutput, 
                sentences: List[Optional[List[str]]], has_sentences: BoolTensor = None, # (N x S_i)
                epoch=None, generate=False, **kwargs):
        assert encoded_sentences is not None, 'SentenceTokenSupervisor requires encoded_sentences, but the dataset does not provide sentences'
        
        if has_sentences is None:
            has_sentences = encoded_sentences.sentence_mask.any(dim=-1)

        # (N x S x d)
        sentence_features = encoded_sentences.sentence_features
        # (N x S)
        sentence_mask = encoded_sentences.sentence_mask
        sentence_mask_in_all = sentence_mask

        if has_sentences is not None:
            encoded_image = encoded_image[has_sentences]
            sentence_features = sentence_features[has_sentences]
            sentence_mask = sentence_mask[has_sentences]
        # (N x H x W x d)
        patch_features = encoded_image.patch_features

        # ---------- Prompt detection ----------
        sentence_grounding_output: TokenDetectorOutput = \
                model.detect_prompts(
                    encoded_image,
                    box_prompts_emb=sentence_features, # (N x S x d)
                    box_prompt_mask=sentence_mask,
                    skip_roi_pool=False if self.config.no_skip_roi_pool else None) # (N x S)

        sentence_region_features = sentence_grounding_output.box_features # (N x S x d)
        
        # ========== Aggregation (for global-contr) ==========
        # (N x d)
        aggregated_sentences = self.aggregator(
            sentence_features, # (N x S x d)
            mask=sentence_mask) # (N x S)
        # (N x d)
        aggregated_patches = self.aggregator(patch_features)
        
        # ========== Loss Functions ==========
        sub_losses = {}
        loss = 0.

        # --- Loss: Global contrastive CLIP-style (global-contr) ---
        if self.config.use_global_contr:
            sub_losses['l_sent/global_contr'] = global_contrastive_loss(aggregated_patches, aggregated_sentences, temp=self.config.global_contr_temp)
            loss += self.config.coeff_global_contr * sub_losses['l_sent/global_contr']

        # --- Loss: Sentence-level contrastive (sent-contr) ---
        if self.config.use_sent_contr:
            sub_losses['l_sent/sent_contr'] = sentence_contrastive_loss(
                sentence_features, # (N x S x d)
                sentence_region_features, # (N x S x d)
                sentence_mask, # (N x S)
                temp=self.config.sent_contr_temp,
                cross_sample_negatives=self.config.cross_sample_sentence_negatives,
                lambda_sent2reg=self.config.lambda_sent2reg,
                lambda_reg2sent=self.config.lambda_reg2sent)
            loss += self.config.coeff_sent_contr * sub_losses['l_sent/sent_contr']

        # --- Loss: Sentence MSE (sent-mse) ---
        if self.config.use_sent_mse:
            sub_losses['l_sent/sent_mse'] = sentence_mse_loss(
                sentence_features, # (N x S x d)
                sentence_region_features, # (N x S x d)
                sentence_mask) # (N x S) 
            
            loss += self.config.coeff_sent_mse * sub_losses['l_sent/sent_mse']

        # --- Loss: Sentence Generation (sent-gen) ---
        if self.config.use_sent_gen:
            sub_losses['l_sent/sent_gen'] = model.train_sentence_decoder(
                flattened_features=sentence_region_features[sentence_mask],
                sentence_mask=sentence_mask_in_all,
                sentences=encoded_sentences.sentences,
                epoch=epoch
            )
            loss += self.config.coeff_sent_gen * sub_losses['l_sent/sent_gen']

        sub_losses['l_sent/total_sent'] = loss

        sent_outputs = {
            'sentences': sentences,
            'encoded_sentences': encoded_sentences,
            'sentence_grounding_output': sentence_grounding_output,
            'sentence_mask': sentence_mask,
            'sentence_region_features': sentence_region_features,
            'grounding': sentence_grounding_output,
        }

        if self.config.use_sent_gen and generate:
            sent_outputs['generated_sentences'] = model.generate_sentences(sentence_region_features, sentence_mask)

        return loss, sub_losses, sent_outputs
    