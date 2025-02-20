from .sdf_head import SDF_head
from .render_head import Render_head



def build_head(args):
    sdf_head = None
    render_args = args.render_head
    render_head = Render_head(render_args.in_channels, render_args.num_classes, render_args.up_ratio, render_args.over_sampling_rate, render_args.ratio_importance)

    sdf_args = args.get('sdf_head', None)
    if sdf_args is not None:
        sdf_head = SDF_head(render_args.in_channels, render_args.num_classes, sdf_args.pos_embd_channel, sdf_args.pos_embd_temp, sdf_args.ratio, sdf_args.mlp_channels)
    return render_head, sdf_head