# Bundled third-party sources

The root LICENSE applies to original solarL code and documentation. Third-party
source snapshots keep their own license status and notices; the root MIT grant
does not relicense them. In particular, vendor/selflift_runtime is not covered
by solarL's MIT grant; see its source record below.

Only this custom-node folder is required; the following implementations are private
modules, not imports from separately installed custom-node packages.

| Bundled file | Upstream | License |
|---|---|---|
| vendor/apt_motion.py | https://github.com/cardenluo/ComfyUI-Apt_Preset — NodeBasic/minimaxH3.py | MIT; licenses/apt_motion.py.LICENSE |
| vendor/upscale_3d.py | https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler — nodes/minimax_h3_latent_upscaler_3d.py | MIT; licenses/upscale_3d.py.LICENSE |

Snapshots were taken from the user's installed versions on 2026-10-13.
Source SHA-256 values and paths are recorded in vendor/sources.json.
Copyright and license notices are preserved verbatim in licenses/.
Model weights are not bundled and retain their own terms.

tiling.py is a solarL implementation of overlapping spatial prediction tiles using
ComfyUI's native H3 PackedLayout and model wrappers. It does not copy or import
SelfLift code. It is an approximate spatial model evaluation: full references and
audio are retained per tile, absolute positions are preserved, and video predictions
are blended before the sampler advances. It does not create cross-tile attention or
guarantee equivalence to full-frame sampling.


## SelfLift private runtime
Source: https://github.com/facok/comfyui-SelfLift . This runtime is an unchanged snapshot of the user-installed local source, with a private package initializer. Source SHA-256 hashes are in vendor/selflift_runtime/SOURCE.json. No LICENSE file was found in the installed source; this inclusion does not assert permission to redistribute that source under MIT or any other open-source license.
