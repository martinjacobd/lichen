"""Diagnostic reads for the bench harnesses: the unmasked top-N, keyed by token string.

The backend reads through an allowed_token_ids mask, which makes the label mass 1 by
construction -- so a read at the wrong position looks as healthy as a right one. The harnesses
exist to catch exactly that (`captured`, and whether the argmax is a label at all), so they read
unmasked, and they want the argmax's STRING to tell a label from 'yes' for 'Yes' or from prose.
"""
import functools


def unmasked(ep):
    """`ep`, reading the unmasked top --top-logprobs of the whole vocabulary."""
    ep.mask = False
    return ep


def by_string(ep, top: dict[int, float]) -> dict[str, float]:
    """A top-N keyed by token id, re-keyed by the token's text (the larger value on a clash)."""
    text = _detokenizer(ep)
    out: dict[str, float] = {}
    for i, lp in top.items():
        s = text(i)
        out[s] = max(lp, out.get(s, lp))
    return out


@functools.cache
def _detokenizer(ep):
    @functools.cache
    def text(i: int) -> str:
        return ep._post("/detokenize", {"model": ep.served, "tokens": [i]})["prompt"]
    return text
