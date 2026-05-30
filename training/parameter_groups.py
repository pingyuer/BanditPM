import logging

log = logging.getLogger()


def get_parameter_groups(model, stage_cfg, print_log=False):
    """
    Assign different weight decays and learning rates to different parameters.
    Returns a parameter group which can be passed to the optimizer.
    """
    weight_decay = stage_cfg.weight_decay
    embed_weight_decay = stage_cfg.embed_weight_decay
    backbone_lr_ratio = stage_cfg.backbone_lr_ratio
    base_lr = stage_cfg.learning_rate
    anchor_ode_lr_ratio = stage_cfg.get("anchor_ode_lr_ratio", None)
    functional_anchor_lr_ratio = stage_cfg.get("functional_anchor_lr_ratio", None)
    unext_lr_ratio = float(stage_cfg.get("unext_lr_ratio", backbone_lr_ratio))
    residual_head_lr_mult = float(stage_cfg.get("residual_head_lr_mult", 1.0))

    if anchor_ode_lr_ratio is not None or functional_anchor_lr_ratio is not None:
        method_lr_ratio = float(functional_anchor_lr_ratio if functional_anchor_lr_ratio is not None else anchor_ode_lr_ratio)
        method_group_name = "functional_anchor" if functional_anchor_lr_ratio is not None else "anchor_ode"
        unext_params = []
        temporal_params = []
        residual_params = []
        embed_params = []
        other_params = []
        embedding_names = ['summary_pos', 'query_init', 'query_emb', 'obj_pe']
        embedding_names = [e + '.weight' for e in embedding_names]

        memo = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or param in memo:
                continue
            memo.add(param)
            if name.startswith('module.'):
                name = name[7:]

            if name.startswith('backbone.'):
                unext_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a UNeXt/base segmenter parameter.')
            elif method_group_name == "functional_anchor" and name.startswith(
                ('residual_heads.', 'faf.residual_head.', 'faf.residual_refiner.', 'faf.trust_gate_net.', 'faf.fusion.')
            ):
                residual_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a functional_anchor residual head parameter.')
            elif name.startswith((
                'state_encoder.',
                'ode_bank.',
                'affine_regressor.',
                'geometry_regressor.',
                'guidance_projs.',
                'gate_head.',
                'confidence.',
                'phase_encoder.',
                'state_ode.',
                'anchor_bank.',
                'anchor_decoder.',
                'residual_heads.',
                'injector.',
                'fusion.',
                'faf.',
            )):
                temporal_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a {method_group_name} parameter.')
            elif any(name.endswith(e) for e in embedding_names):
                embed_params.append(param)
                if print_log:
                    log.info(f'{name} counted as an embedding parameter.')
            else:
                other_params.append(param)

        return [
            {
                'params': unext_params,
                'lr': base_lr * unext_lr_ratio,
                'weight_decay': weight_decay,
                'name': 'unext_base',
            },
            {
                'params': temporal_params,
                'lr': base_lr * method_lr_ratio,
                'weight_decay': weight_decay,
                'name': method_group_name,
            },
            {
                'params': residual_params,
                'lr': base_lr * method_lr_ratio * residual_head_lr_mult,
                'weight_decay': weight_decay,
                'name': 'functional_anchor_residual_heads',
            },
            {
                'params': embed_params,
                'lr': base_lr,
                'weight_decay': embed_weight_decay,
                'name': 'embedding',
            },
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
                'name': 'other',
            },
        ]

    backbone_params = []
    embed_params = []
    other_params = []

    embedding_names = ['summary_pos', 'query_init', 'query_emb', 'obj_pe']
    embedding_names = [e + '.weight' for e in embedding_names]

    # inspired by detectron2
    memo = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid duplicating parameters
        if param in memo:
            continue
        memo.add(param)

        if name.startswith('module'):
            name = name[7:]

        inserted = False
        if name.startswith('pixel_encoder.'):
            backbone_params.append(param)
            inserted = True
            if print_log:
                log.info(f'{name} counted as a backbone parameter.')
        else:
            for e in embedding_names:
                if name.endswith(e):
                    embed_params.append(param)
                    inserted = True
                    if print_log:
                        log.info(f'{name} counted as an embedding parameter.')
                    break

        if not inserted:
            other_params.append(param)

    parameter_groups = [
        {
            'params': backbone_params,
            'lr': base_lr * backbone_lr_ratio,
            'weight_decay': weight_decay
        },
        {
            'params': embed_params,
            'lr': base_lr,
            'weight_decay': embed_weight_decay
        },
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        },
    ]

    return parameter_groups
