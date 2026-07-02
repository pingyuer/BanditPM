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
    cardia_lr_ratio = stage_cfg.get("cardia_lr_ratio", None)
    rebel_lr_ratio = stage_cfg.get("rebel_lr_ratio", None)
    debel_lr_ratio = stage_cfg.get("debel_lr_ratio", None)
    geomaskformer_lr_ratio = stage_cfg.get("geomaskformer_lr_ratio", None)
    unext_lr_ratio = float(stage_cfg.get("unext_lr_ratio", backbone_lr_ratio))
    residual_head_lr_mult = float(stage_cfg.get("residual_head_lr_mult", 1.0))
    gar_offset_lr_mult = float(stage_cfg.get("cardia_offset_lr_mult", stage_cfg.get("gar_offset_lr_mult", 0.5)))
    gar_selector_lr_mult = float(stage_cfg.get("cardia_selector_lr_mult", stage_cfg.get("gar_selector_lr_mult", 2.0)))
    gar_boundary_lr_mult = float(stage_cfg.get("cardia_boundary_lr_mult", stage_cfg.get("gar_boundary_lr_mult", 2.0)))
    gar_proposal_lr_mult = float(stage_cfg.get("cardia_proposal_lr_mult", stage_cfg.get("gar_proposal_lr_mult", 2.0)))
    cardia_ode_control_lr_mult = float(stage_cfg.get("cardia_ode_control_lr_mult", 2.0))

    if anchor_ode_lr_ratio is not None or functional_anchor_lr_ratio is not None or cardia_lr_ratio is not None or rebel_lr_ratio is not None or debel_lr_ratio is not None or geomaskformer_lr_ratio is not None:
        if geomaskformer_lr_ratio is not None:
            method_lr_ratio = float(geomaskformer_lr_ratio)
            method_group_name = "geomaskformer"
        elif debel_lr_ratio is not None:
            method_lr_ratio = float(debel_lr_ratio)
            method_group_name = "debel"
        elif rebel_lr_ratio is not None:
            method_lr_ratio = float(rebel_lr_ratio)
            method_group_name = "rebel"
        elif cardia_lr_ratio is not None:
            method_lr_ratio = float(cardia_lr_ratio)
            method_group_name = "cardia"
        elif functional_anchor_lr_ratio is not None:
            method_lr_ratio = float(functional_anchor_lr_ratio)
            method_group_name = "functional_anchor"
        else:
            method_lr_ratio = float(anchor_ode_lr_ratio)
            method_group_name = "anchor_ode"
        unext_params = []
        unext_no_decay_params = []
        temporal_params = []
        temporal_no_decay_params = []
        residual_params = []
        gar_offset_params = []
        gar_selector_params = []
        gar_boundary_params = []
        gar_proposal_params = []
        cardia_ode_control_params = []
        cardia_ode_control_no_decay_params = []
        embed_params = []
        other_params = []
        embedding_names = ['summary_pos', 'query_init', 'query_emb', 'obj_pe']
        embedding_names = [e + '.weight' for e in embedding_names]

        def no_decay_name(param_name: str) -> bool:
            leaf = param_name.rsplit('.', 1)[-1]
            return (
                leaf == 'bias'
                or 'norm.' in param_name.lower()
                or 'raw_gamma' in param_name
                or 'raw_selector_logit_scale' in param_name
                or 'raw_context_gate' in param_name
                or 'raw_stage3_injection_scale' in param_name
                or 'raw_temperature' in param_name
                or 'static_logits' in param_name
            )

        memo = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or param in memo:
                continue
            memo.add(param)
            if name.startswith('module.'):
                name = name[7:]

            if name.startswith(('backbone.', 'frame_net.', 'encoder.backbone.')):
                (unext_no_decay_params if no_decay_name(name) else unext_params).append(param)
                if print_log:
                    log.info(f'{name} counted as a UNeXt/base segmenter parameter.')
            elif name.startswith(('gar_stage2.offset_head.', 'gar_stage3.offset_head.', 'ode_gen2.offset_head.', 'ode_gen3.offset_head.', 'ode.delta_obs_head.', 'ode.delta_mem_head.', 'grid_solver.head.')):
                gar_offset_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a deformation offset parameter.')
            elif name.startswith((
                'ode_gen2.write_head.',
                'ode_gen3.write_head.',
                'ode_gen2.decay_head.',
                'ode_gen3.decay_head.',
                'ode_gen2.token_proj.',
                'ode_gen3.token_proj.',
                'ode_gen2.runtime_token_proj.',
                'ode_gen3.runtime_token_proj.',
                'ode_gen2.context_token_proj.',
                'ode_gen3.context_token_proj.',
                'ode_gen2.raw_context_gate',
                'ode_gen3.raw_context_gate',
                'cardiac_context.',
                'sldm2.gate_mlp.',
                'sldm3.gate_mlp.',
                'sldm2.runtime_token_update.',
                'sldm3.runtime_token_update.',
                'kv_memory2.',
                'kv_memory3.',
                'ode.gate_head.',
                'ode.write_decay_head.',
                'memory.arbiter.',
                'query_decoder.',
                'grid_solver.film.',
            )):
                (cardia_ode_control_no_decay_params if no_decay_name(name) else cardia_ode_control_params).append(param)
                if print_log:
                    log.info(f'{name} counted as an ODE-control parameter.')
            elif name.startswith((
                'gar_stage2.spatial_selector.',
                'gar_stage2.global_selector.',
                'gar_stage2.raw_selector_logit_scale',
                'gar_stage3.spatial_selector.',
                'gar_stage3.global_selector.',
                'gar_stage3.raw_selector_logit_scale',
                'ode_gen2.spatial_selector.',
                'ode_gen2.global_selector.',
                'ode_gen2.raw_selector_logit_scale',
                'ode_gen3.spatial_selector.',
                'ode_gen3.global_selector.',
                'ode_gen3.raw_selector_logit_scale',
            )):
                gar_selector_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a CARDIA selector parameter.')
            elif name.startswith((
                'boundary_fusion.edge_gate.',
                'boundary_fusion.edge_gate_head.',
                'boundary_fusion.channel_gate.',
                'boundary_fusion.boundary_aux_head.',
            )):
                gar_boundary_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a CARDIA boundary gate parameter.')
            elif name.startswith(('proposal_head.', 'logit_fusion.', 'fusion.', 'decoder.mask_head.', 'obs_head.', 'correction.', 'boundary_residual.')):
                gar_proposal_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a proposal/logit-fusion parameter.')
            elif method_group_name == "functional_anchor" and name.startswith(
                ('residual_heads.', 'faf.residual_head.', 'faf.residual_refiner.', 'faf.trust_gate_net.', 'faf.fusion.')
            ):
                residual_params.append(param)
                if print_log:
                    log.info(f'{name} counted as a functional_anchor residual head parameter.')
            elif name.startswith((
                'raw_stage3_injection_scale',
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
                'gar_stage3.',
                'gar_stage2.',
                'runtime_memory3.',
                'runtime_memory2.',
                'sldm3.',
                'sldm2.',
                'kv_memory3.',
                'kv_memory2.',
                'cardiac_context.',
                'ode_gen3.',
                'ode_gen2.',
                'grid_solver.',
                'fuse3.',
                'fuse2.',
                'boundary_fusion.',
                'proposal_head.',
                'logit_fusion.',
                'encoder.obs_project.',
                'ode.',
                'memory.',
                'decoder.',
                'obs_head.',
                'correction.',
                'fusion.',
                'tokenizer.',
                'video_encoder.',
                'query_decoder.',
                'grid_solver.',
                'boundary_residual.',
                'image_tokenizer.',
                'mask_tokenizer.',
                'transformer.',
                'pixel_decoder.',
                'proposal_decoder.',
            )):
                (temporal_no_decay_params if no_decay_name(name) else temporal_params).append(param)
                if print_log:
                    log.info(f'{name} counted as a {method_group_name} method parameter.')
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
                'params': unext_no_decay_params,
                'lr': base_lr * unext_lr_ratio,
                'weight_decay': 0.0,
                'name': 'unext_base_no_decay',
            },
            {
                'params': temporal_params,
                'lr': base_lr * method_lr_ratio,
                'weight_decay': weight_decay,
                'name': method_group_name,
            },
            {
                'params': temporal_no_decay_params,
                'lr': base_lr * method_lr_ratio,
                'weight_decay': 0.0,
                'name': f'{method_group_name}_no_decay',
            },
            {
                'params': gar_offset_params,
                'lr': base_lr * method_lr_ratio * gar_offset_lr_mult,
                'weight_decay': weight_decay,
                'name': f'{method_group_name}_offset',
            },
            {
                'params': gar_selector_params,
                'lr': base_lr * method_lr_ratio * gar_selector_lr_mult,
                'weight_decay': 0.0,
                'name': f'{method_group_name}_selector',
            },
            {
                'params': gar_boundary_params,
                'lr': base_lr * method_lr_ratio * gar_boundary_lr_mult,
                'weight_decay': 0.0,
                'name': f'{method_group_name}_boundary_gate',
            },
            {
                'params': gar_proposal_params,
                'lr': base_lr * method_lr_ratio * gar_proposal_lr_mult,
                'weight_decay': weight_decay,
                'name': f'{method_group_name}_proposal_head',
            },
            {
                'params': cardia_ode_control_params,
                'lr': base_lr * method_lr_ratio * cardia_ode_control_lr_mult,
                'weight_decay': weight_decay,
                'name': f'{method_group_name}_ode_control',
            },
            {
                'params': cardia_ode_control_no_decay_params,
                'lr': base_lr * method_lr_ratio * cardia_ode_control_lr_mult,
                'weight_decay': 0.0,
                'name': f'{method_group_name}_ode_control_no_decay',
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
