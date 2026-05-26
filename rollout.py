import torch, transformers
from PIL import Image

def init_new_tokens(new_tokens, like_tokens, weights, model, tokenizer):
    """
    Args:
        new_tokens: list of new token strings to add to the tokenizer
        like_tokens: list of existing token strings to base the new token embeddings on
        weights: tensor of shape (len(new_tokens), len(like_tokens)) specifying the weights for combining the like_token embeddings to create the new token embeddings
    Returns:
        list of new token ids corresponding to the new_tokens
    """
    # Find embeddings of related tokens
    token_ids = tokenizer.encode(like_tokens)
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(torch.tensor(token_ids, device=embed_layer.weight.device)).squeeze(1)
    # Merge them into new similar tokens
    assert weights.shape == (len(new_tokens), len(like_tokens))
    assert torch.allclose(weights.sum(axis=1), torch.ones(len(new_tokens)))
    embeds = (embeds * weights.to(embeds.device, embeds.dtype).unsqueeze(-1)).sum(axis=1)
    # Insert into tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        embed_layer.weight.data[-len(new_tokens):] = embeds
    return tokenizer.encode(new_tokens)

def parse_actions(text:str) -> list:
    actions = []
    while "<|action>" in text and "<action|>" in text:
        start = text.index("<|action>") + len("<|action>")
        end = text.index("<action|>")
        action_text = text[start:end].strip()
        actions.append(action_text)
        text = text[end + len("<action|>"):]
    return actions

def state_encoder(state) -> torch.Tensor:
    pass

async def execute(model:transformers.modeling_utils.PreTrainedModel, tokenizer, env, text:str, image:str|Image.Image=None, tools=[], max_steps=10) -> list:
    # Initialize trajectory and get initial state
    trajectory = []
    # Create initial input message
    messages = [{"role": "system", "content": 
                """
                You are an agent, will be given a current state and a task description, and must interact with the environment by generating actions. \
                The state will be provided between a start state token (i.e., <|state>) and an end state token (i.e., <|state|>), and will be represented as a fixed-length vector embedding. \
                Please format your actions between a start action token (i.e., <|action>) and an end action token (i.e., <action|>) tags. \
                The environment will respond with observations and rewards based on your actions. \
                """},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}, {"type": "text", "text": "<|state>"+"<state|>"}]
    # Tokenize input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=True,
        tools=tools
    ).to(model.device)
    # Get embeddings
    inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
    if ple_submodel:=next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False):
        per_layer_inputs = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)

    for _ in range(max_steps):
        # Determine location in embedding for the state representation
        state_token_id = tokenizer.encode("<|state>")[0]
        state_mask = (inputs["input_ids"] == torch.tensor(state_token_id).to(model.device)).reshape(-1)
        # Format current state into prompt
        state_features = state_encoder(env.state)
        inputs_embeds[state_mask:state_mask+len(state_features)] = state_features

        # Most models are decoder-only, so we can directly pass (modified) embeddings
        kwargs = {
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "use_cache": True,
            "logits_to_keep": 1,
            "output_logits": True,
            "return_dict_in_generate": True
        }
        kwargs["inputs_embeds"] = inputs_embeds
        if ple_submodel: 
            kwargs["per_layer_inputs"] = per_layer_inputs
        
        # Generate next action
        outputs = model.generate(
            **kwargs
        )
        text = tokenizer.decode(outputs.sequences, skip_special_tokens=False)

        # Parse and execute action
        actions = parse_actions(text)
        env = await env.step(actions)

        trajectory.append({"inputs_embeds": inputs_embeds, "logits": torch.stack(outputs.logits).squeeze(), "reward": env.reward})
        messages = [{"role": "user", "content": [{"type": "text", "text": "Determine the next action based on the new state. <|state>"+"<state|>"}]}]

        if "done" in actions:
            break

    return trajectory