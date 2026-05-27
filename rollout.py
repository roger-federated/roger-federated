import torch, transformers
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

TOOL_CALL_TOKENS = [
    ("<|tool_call>", "<tool_call|>"),
    ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>")
]

def _make_tool_call_prefix_fn(tokenizer):
    """Restrict generation to valid JSON tool call format once <tool_call> is emitted."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}
    inner = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
    def prefix_fn(batch_id, sent_tokens):
        decoded:str = tokenizer.decode(sent_tokens)
        tool_call = [start in decoded for start, end in TOOL_CALL_TOKENS]
        if not any(tool_call):
            return list(range(tokenizer.vocab_size))
        tool_call_tokens = TOOL_CALL_TOKENS[tool_call.index(True)]
        after = decoded.rsplit(tool_call_tokens[0], 1)[0]
        before = after.rsplit(tool_call_tokens[1], 1)[-1]
        return inner(batch_id, torch.tensor(tokenizer.encode(before, add_special_tokens=False)))
    return prefix_fn

async def execute_tools(text, tool_handlers, env):
    pass

def state_encoder(state) -> torch.Tensor:
    pass

async def execute(model:transformers.modeling_utils.PreTrainedModel, tokenizer, env, text:str, image:str|Image.Image=None, tools=[], tool_handlers:dict={}, max_steps=10) -> list:
    # Initialize trajectory and get initial state
    trajectory = []
    # Create initial input message
    messages = [{"role": "system", "content": 
                """
                You are an agent, will be given a current state and a task description, and must interact with the provided (MCP) tools in order to accomplish the task. \
                The current state will be provided between a start state token (i.e., <|state>) and an end state token (i.e., <state|>). \
                After thinking about your long and short-term intent, immediately provide your tool call. \
                After having performed this action, a new state will be provided for you to act upon. \
                """},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}, {"type": "text", "text": "<|state>"+"<state|>"}]

    prefix_fn = _make_tool_call_prefix_fn(tokenizer)
    for _ in range(max_steps):
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

        # Most models are decoder-only, so we can directly pass (modified) embeddings
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        # Some models have per-layer embeddings
        if ple_submodel:=next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False):
            per_layer_inputs = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)

        # Determine location in embedding for the state representation
        state_token_id = tokenizer.encode("<|state>")[0]
        state_idx = (inputs["input_ids"] == torch.tensor(state_token_id).to(model.device)).reshape(-1)
        # Format current state into prompt
        state_features = state_encoder(env.state)
        inputs_embeds[state_idx:state_idx+len(state_features)] = state_features
        
        # Prepare inputs for generation
        kwargs = {
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "use_cache": True,
            "logits_to_keep": 1,
            "output_logits": True,
            "return_dict_in_generate": True,
            "inputs_embeds": inputs_embeds,
            "prefix_allowed_tokens_fn": prefix_fn
        }
        if ple_submodel:
            kwargs["per_layer_inputs"] = per_layer_inputs
        # Generate next action
        outputs = model.generate(
            **kwargs
        )
        text = tokenizer.decode(outputs.sequences, skip_special_tokens=False)

        # Parse and execute action
        env = await execute_tools(text, tool_handlers, env)
        
        # Save trajectory
        trajectory.append({"inputs_embeds": inputs_embeds, "logits": torch.stack(outputs.logits).squeeze(), "reward": env.reward})
        messages = [{"role": "user", "content": [{"type": "text", "text": "Determine your next action based on the new state. <|state>"+"<state|>"}]}]

        if env.state == "terminal":
            break

    return trajectory