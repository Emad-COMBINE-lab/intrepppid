import torch.nn.functional as F


def embedding_dropout(training, embed, words, p=0.2):
    if not training:
        masked_embed_weight = embed.weight
    elif not p:
        masked_embed_weight = embed.weight
    else:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - p
        ).expand_as(embed.weight) / (1 - p)
        masked_embed_weight = mask * embed.weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )
    return X
