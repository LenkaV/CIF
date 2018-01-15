# Composite Indicators Framework (CIF) for Business Cycle Analysis

Composite indicators are recognized to be an eligible tool of the business cycle analysis, especially because they can be easily interpreted although they summarize multidimensional relationships between individual economic indicators.

The methodology of composite indicators construction was described in detail by several organizations (OECD, Conference Board etc.). It therefore came as a surprise, that no publicly available software program haven't supported the whole computational process or its automation till now. This new python library was proposed to fill this gap!

It nowadays contains more than 30 functions designed to construct composite leading indicators and covers several areas:
- loading data directly from OECD API,
- basic conversion from quaterly to monthly data,
- data transformations (seasonal adjustment, stabilising forecasts, detrending, normalization),
- ex-post turning points detection (Bry-Boschan algorithm),
- real-time turning points detection from archive values,
- evaluation,
- aggregation into composite indicator,
- visualisations,
- and more.

It will soon be available via pip.
