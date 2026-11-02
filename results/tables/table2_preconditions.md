# 表 2  判据前置条件清单 A1 到 A10

转录自 `docs/literature/06_适用条件审计_方向定义与查新.md` 第 2.2 节。表中 P2 到 P11 为该方向的已有结论编号。

| 编号 | 前置条件 | 违反时的后果 | 已有占位 |
|---|---|---|---|
| A1 | Observation modality complete: at least one end has an array, so arrival or departure directions can be measured | Direction constraints unavailable; the arrival-direction vectors the criterion is built on cannot be observed and the FIM construction fails | Wymeersch RIS Table I; Emenonye RIS limitations; Villaverde (theorem level) |
| A2 | Paths resolvable in delay: delay difference exceeds $1/B$ | Sub-tap paths form an ambiguity subspace whose Fisher information is zero | P3 |
| A3 | Paths resolvable in angle: angular separation exceeds the array angular resolution | Angular merging, which is the angular-domain counterpart of A2 | joint with A2, in the P3 family |
| A4 | Correspondence between paths and geometric features is known | Data association ambiguity; joint association and state estimation becomes necessary | P11 |
| A5 | Nuisance parameters eliminated correctly (complex gain, clock offset, source spectrum) | Position information is inflated or truncated by a wrong elimination convention | P4 |
| A6 | Anchor positions, array orientations and floor geometry known | Geometry prior error enters the position dimensions; the bound is inflated or saturates | P8, P6 |
| A7 | Noise statistics known and credible, usually assumed Gaussian | The derivation flexibility of the FIM and CRB no longer holds | standard statistical premise |
| A8 | Asymptotics hold: enough snapshots and signal-to-noise ratio for the CRB to be attainable | Threshold effect; the CRB is not attainable at low signal-to-noise ratio | third question of Vallisneri 2008 |
| A9 | Geometry non-degenerate: anchors not coplanar or collinear | Rank structure degenerates, yet rank and condition-number checks cannot see it | P5 |
| A10 | Only local identifiability is required, or global ambiguity is excluded separately | Full column rank does not exclude discrete ambiguity; mirror and collinear solutions | P2 |
