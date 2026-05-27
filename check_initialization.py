from peft_pretraining.modeling_llama import LlamaForCausalLM as LlamaForCausalLM_Prenorm
from peft_pretraining.modeling_llama_prepostnorm import LlamaForCausalLM as LlamaForCausalLM_Prepostnorm
from peft_pretraining.modeling_llama_hybridnorm import LlamaForCausalLM as LlamaForCausalLM_hybridnorm

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def check_initialization(model_config="configs/llama_1b.json"):
    model_config = AutoConfig.from_pretrained(model_config)

    prenorm_model = LlamaForCausalLM_Prenorm(config=model_config)
    prenorm_model_2 = LlamaForCausalLM_Prenorm(config=model_config)
    prepostnorm_model = LlamaForCausalLM_Prepostnorm(config=model_config)
    hybridnorm_model = LlamaForCausalLM_hybridnorm(config=model_config)

    prenorm_param_dict = {}
    for pn, p in prenorm_model.named_parameters():
        prenorm_param_dict[pn] = p

    for pn, p in prepostnorm_model.named_parameters():
        if pn in prenorm_param_dict.keys():
            p.data = prenorm_param_dict[pn].data

    for pn, p in hybridnorm_model.named_parameters():
        if pn in prenorm_param_dict.keys():
            p.data = prenorm_param_dict[pn].data

    for pn, p in prenorm_model_2.named_parameters():
        if pn in prenorm_param_dict.keys():
            p.data = prenorm_param_dict[pn].data

    input = torch.randint(0, 100, (1, 3))
    position_id = torch.arange(0, 3).unsqueeze(0)
    attention_mask = torch.ones((1, 3))

    prenorm_output = prenorm_model(input, position_ids=position_id, attention_mask=attention_mask)
    prenorm_output2 = prenorm_model_2(input, position_ids=position_id, attention_mask=attention_mask)
    prepostnorm_output = prepostnorm_model(input, position_ids=position_id, attention_mask=attention_mask)
    hybridnorm_output = hybridnorm_model(input, position_ids=position_id)

    print(f"prenorm-prepostnorm: {torch.cosine_similarity(prenorm_output[0], prepostnorm_output[0], dim=-1)}")
    print(f"prenorm-hybridnorm: {torch.cosine_similarity(prenorm_output[0], hybridnorm_output[0], dim=-1)}")
    print(f"prepostnorm-hybridnorm: {torch.cosine_similarity(hybridnorm_output[0], prepostnorm_output[0], dim=-1)}")
    print(f"prenorm-prenorm: {torch.cosine_similarity(prenorm_output[0], prenorm_output2[0], dim=-1)}")


if __name__ == "__main__":
    check_initialization(model_config="configs/llama_1b.json")