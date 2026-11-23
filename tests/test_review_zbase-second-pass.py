"""z_base background: one decoder pass per head, same numbers as the old second full pass."""

from __future__ import annotations

import torch
from scvi.module._totalvae import TOTALVAE
from torch.nn.modules.batchnorm import _BatchNorm

from tests.test_ambient import _model, sim  # noqa: F401  (sim is a fixture)


def _gen_inputs(model, adata, module):
    tensors = next(iter(model._make_data_loader(adata=adata, batch_size=16)))
    with torch.no_grad():
        inf = module.inference(**module._get_inference_input(tensors))
    return module._get_generative_input(tensors, inf)


def _bn_counts(module):
    return {n: int(m.num_batches_tracked) for n, m in module.decoder.named_modules() if isinstance(m, _BatchNorm)}


def test_zbase_train_forward_advances_each_batchnorm_once(sim):  # noqa: F811
    model, adata = _model(sim, protein_background_source="z_base", use_batch_norm="decoder")
    module = model.module
    module.train()
    gen_in = _gen_inputs(model, adata, module)
    before = _bn_counts(module)
    assert before
    module.generative(**gen_in)
    after = _bn_counts(module)
    assert {k: after[k] - before[k] for k in before} == dict.fromkeys(before, 1)


def test_zbase_background_matches_old_second_pass_in_eval(sim):  # noqa: F811
    model, adata = _model(sim, protein_background_source="z_base", use_batch_norm="decoder")
    module = model.module
    module.eval()
    gen_in = _gen_inputs(model, adata, module)
    z_base = gen_in["z_base"].expand_as(gen_in["z"])
    lib, bidx, lab = gen_in["library_gene"], gen_in["batch_index"], gen_in["label"]
    with torch.no_grad():
        torch.manual_seed(7)
        new = module.generative(**gen_in)["py_"]
        torch.manual_seed(7)
        TOTALVAE.generative(module, gen_in["z"], lib, bidx, lab)  # first pass at z, as before
        old = TOTALVAE.generative(module, z_base, lib, bidx, lab)["py_"]  # old second full pass
    for key in ("back_alpha", "back_beta", "rate_back"):
        assert torch.equal(new[key], old[key]), key


def test_default_source_is_bit_identical_to_parent(sim):  # noqa: F811
    model, adata = _model(sim, use_batch_norm="decoder")
    module = model.module
    module.train()
    gen_in = _gen_inputs(model, adata, module)
    args = (gen_in["z"], gen_in["library_gene"], gen_in["batch_index"], gen_in["label"])
    torch.manual_seed(3)
    new = module.generative(**gen_in)
    torch.manual_seed(3)
    old = TOTALVAE.generative(module, *args)
    for grp in ("px_", "py_", "py_norm_"):
        for key, val in old[grp].items():
            assert torch.equal(new[grp][key], val), (grp, key)
