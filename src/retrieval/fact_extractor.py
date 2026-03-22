from __future__ import annotations

import re

from src.models import QueryFact


class FactExtractor:
    def extract(self, query: str, seed: QueryFact | None = None) -> QueryFact:
        fact = seed or QueryFact()
        text = query.lower()

        height_match = re.search(r"(\d+(?:\.\d+)?)\s*(m|metre|metres|meter|meters)\b", text)
        if height_match:
            fact.height_m = float(height_match.group(1))

        floor_area_match = re.search(r"(\d+(?:\.\d+)?)\s*(sq\.?\s*m|square metres|sqm)\b", text)
        if floor_area_match:
            fact.floor_area_sqm = float(floor_area_match.group(1))

        floors_match = re.search(r"(\d+)\s*(storey|storeys|story|stories|floor|floors)\b", text)
        if floors_match:
            fact.floors = int(floors_match.group(1))

        plot_area_match = re.search(r"plot\s*(area)?\s*(of)?\s*(\d+(?:\.\d+)?)\s*(sq\.?\s*m|sqm)", text)
        if plot_area_match:
            fact.plot_area_sqm = float(plot_area_match.group(3))

        return fact

