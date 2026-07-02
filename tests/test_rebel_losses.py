import torch
from omegaconf import OmegaConf

from losses.computer import LossComputer
from tests.test_rebel import _cfg, build_model


def _loss_cfg():
    cfg = _cfg()
    cfg.loss = {
        "name": "rebel",
        "rebel": {
            "final": 1.0,
            "base_aux": 0.35,
            "belief_prior": 0.15,
            "obs_aux": 0.20,
            "rebel_aux": 0.10,
            "corrected_aux": 0.05,
            "candidate_oracle": 0.15,
            "arbitration": 0.05,
            "correction": 0.05,
            "temporal": 0.03,
            "offset_smooth": 0.005,
            "write_reg": 0.01,
        },
    }
    return cfg


def test_rebel_loss_keys_and_gradients_reach_decoder():
    cfg = _loss_cfg()
    stage = OmegaConf.create({"point_supervision": False, "train_num_points": 16, "oversample_ratio": 3, "importance_sample_ratio": 0.75})
    model = build_model(cfg, device="cpu")
    data = {"rgb": torch.randn(1, 2, 1, 32, 32), "cls_gt": torch.randint(0, 2, (1, 2, 32, 32)), "current_iter": 5}
    out = model(data)
    data.update(out)
    data["supervised_indices"] = torch.ones(1, 2, dtype=torch.bool)
    losses = LossComputer(cfg, stage).compute(data, [1])
    for key in (
        "rebel_final",
        "rebel_base_aux",
        "rebel_belief_prior",
        "rebel_obs_aux",
        "rebel_decoder_aux",
        "rebel_corrected_aux",
        "rebel_candidate_oracle",
        "rebel_arbitration",
        "rebel_correction",
        "rebel_temporal",
        "rebel_offset_smooth",
        "rebel_write_reg",
    ):
        assert key in losses
    losses["total_loss"].backward()
    assert model.decoder.mask_head.weight.grad is not None
    assert model.memory.mask_prior_head.weight.grad is not None
    assert model.fusion.context[-1].weight.grad is not None


def test_rebel_correction_loss_weights_disagreement_region():
    from rebel.losses import weighted_ce
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 0, 0, 0] = 4.0
    soft = torch.zeros(1, 2, 2, 2)
    soft[:, 1] = 1.0
    low = weighted_ce(logits, soft, torch.ones(1, 1, 2, 2))
    high = weighted_ce(logits, soft, torch.tensor([[[[5.0, 1.0], [1.0, 1.0]]]]))
    assert high > low
