import torch

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