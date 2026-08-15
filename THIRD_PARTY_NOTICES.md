# Third-party notices

MINT interfaces with upstream research software and model assets. Those works
are not relicensed under MINT's MIT license.

| Component | Upstream | License notes | Distributed here |
| --- | --- | --- | --- |
| LingBot-Map model code | `robbyant/lingbot-map` | Apache-2.0; model terms may differ | A minimal source subset is included for model construction |
| GeoCalib | `cvg/GeoCalib` | Apache-2.0; weights are CC BY 4.0 | No |
| MoGe | `microsoft/MoGe` | MIT; model terms may differ | No |
| Mega-SAM | `mega-sam/mega-sam` | Apache-2.0 for software; CC BY 4.0 for other materials | No |
| HaWoR | `ThunderVVV/HaWoR` | Restrictive non-commercial/no-derivatives terms | No |
| MANO | Official MANO website | Separate registration and license agreement | No |
| PyTorch3D | `facebookresearch/pytorch3d` | BSD-3-Clause | No |

The data preparation workflow may be used with Ego4D, EPIC-KITCHENS, or EgoDex
only under the terms accepted by the person who obtained the data. Dataset
availability does not automatically grant redistribution rights.

Before publishing a release:

1. Pin every optional repository to a reviewed commit.
2. Retain upstream copyright and notice files.
3. Confirm that model checkpoint terms permit the intended use.
4. Keep MANO and other registration-gated files outside source distributions.
5. Obtain written approval for every redistributed sample clip.
