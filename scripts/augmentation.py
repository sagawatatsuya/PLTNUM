import random

def random_change_augmentation(aas, cfg):
    residue_tokens = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
    stracture_aware_tokens = ("a", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "y")
    length = len(aas)
    swap_indices = random.sample(
        range(length), int(length * cfg.random_change_ratio)
    )
    new_aas = ""
    for i, aa in enumerate(aas):
        if i in swap_indices:
            if aas[i] in residue_tokens:
                new_aas += random.choice(residue_tokens)
            elif aas[i] in stracture_aware_tokens:
                new_aas += random.choice(stracture_aware_tokens)
        else:
            new_aas += aa
    return new_aas


def mask_augmentation(aas, cfg):
    length = len(aas)
    swap_indices = random.sample(
        range(0, length // cfg.token_length),
        int(length // cfg.token_length * cfg.mask_ratio),
    )
    for ith in swap_indices:
        aas = (
            aas[: ith * cfg.token_length]
            + "@" * cfg.token_length
            + aas[(ith + 1) * cfg.token_length :]
        )
    aas = aas.replace("@@", "<mask>").replace("@", "<mask>")
    return aas


def random_delete_augmentation(aas, cfg):
    length = len(aas)
    swap_indices = random.sample(
        range(0, length // cfg.token_length),
        int(length // cfg.token_length * cfg.random_delete_ratio),
    )
    for ith in swap_indices:
        aas = (
            aas[: ith * cfg.token_length]
            + "@" * cfg.token_length
            + aas[(ith + 1) * cfg.token_length :]
        )
    aas = aas.replace("@@", "").replace("@", "")
    return aas


def truncate_augmentation(aas, cfg):
    length = len(aas)
    if length > cfg.max_length:
        diff = length - cfg.max_length
        start = random.randint(0, diff)
        return aas[start : start + cfg.max_length]
    else:
        return aas