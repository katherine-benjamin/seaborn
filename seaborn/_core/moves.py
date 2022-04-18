from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from pandas import DataFrame
    from seaborn._core.groupby import GroupBy


@dataclass
class Move:

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str,
    ) -> DataFrame:
        raise NotImplementedError


@dataclass
class Jitter(Move):

    width: float = 0
    x: float = 0
    y: float = 0

    seed: Optional[int] = None

    # TODO what is the best way to have a reasonable default?
    # The problem is that "reasonable" seems dependent on the mark

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str,
    ) -> DataFrame:

        # TODO is it a problem that GroupBy is not used for anything here?
        # Should we type it as optional?

        data = data.copy()

        rng = np.random.default_rng(self.seed)

        def jitter(data, col, scale):
            noise = rng.uniform(-.5, +.5, len(data))
            offsets = noise * scale
            return data[col] + offsets

        if self.width:
            data[orient] = jitter(data, orient, self.width * data["space"])
        if self.x:
            data["x"] = jitter(data, "x", self.x)
        if self.y:
            data["y"] = jitter(data, "y", self.y)

        return data


@dataclass
class Dodge(Move):

    empty: str = "keep"  # keep, drop, fill
    gap: float = 0

    # TODO accept just a str here?
    by: Optional[list[str]] = None

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str,
    ) -> DataFrame:

        grouping_vars = [v for v in groupby.order if v in data]

        groups = groupby.agg(data, {"space": "max"})
        if self.empty == "fill":
            groups = groups.dropna()

        def groupby_pos(s):
            grouper = [groups[v] for v in [orient, "col", "row"] if v in data]
            return s.groupby(grouper, sort=False, observed=True)

        def scale_space(w):
            # TODO what value to fill missing widths??? Hard problem...
            # TODO short circuit this if outer widths has no variance?
            empty = 0 if self.empty == "fill" else w.mean()
            filled = w.fillna(empty)
            scale = filled.max()
            norm = filled.sum()
            if self.empty == "keep":
                w = filled
            return w / norm * scale

        def space_to_offsets(w):
            return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2

        new_space = groupby_pos(groups["space"]).transform(scale_space)
        offsets = groupby_pos(new_space).transform(space_to_offsets)

        if self.gap:
            new_space *= 1 - self.gap

        groups["_dodged"] = groups[orient] + offsets
        groups["space"] = new_space

        out = (
            data
            .drop("space", axis=1)
            .merge(groups, on=grouping_vars, how="left")
            .drop(orient, axis=1)
            .rename(columns={"_dodged": orient})
        )

        return out
