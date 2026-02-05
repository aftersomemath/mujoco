from typing import Any, Dict, List, Set
from mujoco_sysid.report.sections.base import ReportSection

class RowSection(ReportSection):
    """A report section that displays multiple other sections side-by-side."""

    def __init__(
        self,
        title: str,
        sections: List[ReportSection],
        anchor: str = "",
        collapsible: bool = True,
        description: str = "",
    ):
        super().__init__(collapsible=collapsible)
        self._title = title
        self.sections = sections
        self._anchor = anchor
        self._description = description

    @property
    def title(self) -> str:
        return self._title

    @property
    def anchor(self) -> str:
        return self._anchor

    @property
    def template_filename(self) -> str:
        return "row.html"

    def header_includes(self) -> Set[str]:
        includes = set()
        for section in self.sections:
            includes.update(section.header_includes())
        return includes

    def get_context(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "anchor": self.anchor,
            "description": self._description,
        }
