


from itertools import chain
from typing import List

import torch


def flatten_prompts(prompts: List[List[str]], device):
    flattened_prompts: List[str] = [prompt for sub_prompts in prompts if sub_prompts is not None for prompt in sub_prompts]
    N = len(prompts)
    M_is: List[int] = [len(sub_prompts) if sub_prompts is not None else 0 for sub_prompts in prompts]
    # (N x max(M_i))
    prompt_mask = torch.zeros(N, max(M_is) if len(M_is) > 0 else 0, dtype=torch.bool, device=device)
    for i, M_i in enumerate(M_is):
        prompt_mask[i, :M_i] = True
    return flattened_prompts, prompt_mask

def flatten_prompts_2(prompts: List[List[List[str]]], device):
    flattened_prompts: List[str] = [prompt for sub_prompts in prompts for sub_sub_prompts in sub_prompts for prompt in sub_sub_prompts]
    N = len(prompts)
    M_is: List[int] = [len(sub_prompts) for sub_prompts in prompts]
    L_is_ms: List[List[int]] = [[len(sub_sub_prompts) for sub_sub_prompts in sub_prompts]
                                for sub_prompts in prompts]
    max_L = max(max(L_i_m) for L_i_m in L_is_ms)
    prompt_mask = torch.zeros(N, max(M_is), max_L, dtype=torch.bool, device=device)
    for i, L_i_ms in enumerate(L_is_ms):
        for m, L_i_m in enumerate(L_i_ms):
            prompt_mask[i, m, :L_i_m] = True

    return flattened_prompts, prompt_mask

        
def apply_placeholder(prompt: str, placeholder: str, replacements: List[str]) -> List[str]:
    placeholder = '{' + placeholder + '}'
    if placeholder not in prompt:
        return [prompt]
    else:
        return [prompt.replace(placeholder, replacement) for replacement in replacements]


def fill_prompt_templates(prompts: List[str]) -> List[str]:
    for placeholder, replacements in template_placeholders.items():
        prompts = list(chain(*[apply_placeholder(prompt, placeholder, replacements) for prompt in prompts]))
    assert not any('{' in prompt for prompt in prompts), [prompt for prompt in prompts if '{' in prompt]
    return prompts

def localized_prompt_templates(prompts: List[str], region_templates: List[str]) -> List[List[str]]:
    prompts = [
        [reg_template.format(prompt) for prompt in prompts]
        for reg_template in region_templates
    ]
    assert not any('{' in prompt for templ in prompts for prompt in templ)
    return prompts


template_placeholders = {
    'normal_adj': ['normal', 'unremarkable', 'clear'],
    'shape_noun': ['size', 'silhouette', 'area', 'contours'], 
    'shape_adj': ['round'], 
    'state_verb': ['appears', 'is', 'are', 'remains', 'remain', 'appear', 'exists', 'exist'], 
    'indication_noun': ['signs', 'evidence', 'case of', 'presence', 'findings', 'suspicious findings'], 
    'indication_adj': ['noticeable', 'visible', 'seen', 'appearent', 'observable'], 
    'indication_verb': ['indicates', 'suggests', 'suggesting', 'indicating', 'consistent with'], 
    'passive_indication_verb': ['can be identified', 'can be seen', 'is present', 'is noted'], 
    'unchanged': ['has not improved', 'unchanged', 'remains'], 
    'limits_noun': ['limits'], 
    'moderate_adj': ['mild', 'moderate', 'extensive', 'small', 'slight', 'stable', 'intact', 'mild-moderate',], 
    'strong_adj': [
        'large', 'significant', 'acute', 'widespread', 'relevant', 'difficult', 'apparent', 'prominent',
        'convincing', 'extensive', 'severe', 'critical', 'altered', 'patchy', 'degenerative', 'substantial',
        'predominant', 'massive', 'noticeable'], 
    'increased_adj': ['elevated', 'enlarged', 'increased', 'larger', 'large', 'widened'], 
    'size_noun': ['enlargement'], 
    'visible_adj': ['visible', 'seen', 'appearent'], 
    'relation_adj': ['regarding',' relating to', 'concerning', 'involving'], 
    'support_dev_noun': ['catheter', 'tubes', 'support device', 'monitoring device', 'wires', 'pacemaker'], 
    'change': ['little change', 'unchanged',], 
    'lung_adj': ['pulmonary','pul', 'lung', 'airspace']
}