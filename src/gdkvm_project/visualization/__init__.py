from gdkvm_project.visualization.panels import render_dpfr_diagnostic_panel, render_sequence_panel
from gdkvm_project.visualization.registry import VISUALIZER_REGISTRY

VISUALIZER_REGISTRY.register("sequence_panel")(render_sequence_panel)
VISUALIZER_REGISTRY.register("dpfr")(render_dpfr_diagnostic_panel)
VISUALIZER_REGISTRY.register("dual_prompt_flow_refinement")(render_dpfr_diagnostic_panel)

__all__ = ["VISUALIZER_REGISTRY", "render_sequence_panel", "render_dpfr_diagnostic_panel"]
