from enum import IntEnum
from typing import Dict, Iterator, Optional


class TaxonomicRanks(IntEnum):
    """Enumeration of supported taxonomic levels."""

    DOMAIN = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6

    _prefix_to_rank: Dict[str, "TaxonomicRanks"]

    @property
    def name(self) -> str:
        return super().name.lower()

    @property
    def prefix(self) -> str:
        return f"{self.name[0]}__"

    @property
    def child(self) -> Optional["TaxonomicRanks"]:
        """Get the child (more specific) taxonomic rank."""
        try:
            return TaxonomicRanks(self.value + 1)
        except ValueError:
            return None  # Already at lowest rank

    @property
    def parent(self) -> Optional["TaxonomicRanks"]:
        """Get the parent (broader) taxonomic rank."""
        try:
            return TaxonomicRanks(self.value - 1)
        except ValueError:
            return None  # Already at highest rank

    @classmethod
    def from_name(cls, rank: str) -> "TaxonomicRanks":
        """Get enum member from rank name."""
        rank = rank.upper()
        try:
            return cls[rank]  # type: ignore
        except KeyError:
            raise ValueError(f"Invalid taxonomic rank: {rank}")

    @classmethod
    def from_prefix(cls, prefix: str) -> "TaxonomicRanks":
        """Get enum member from taxonomic prefix."""
        if not hasattr(cls, "_prefix_to_rank"):
            cls._prefix_to_rank = {rank.prefix: rank for rank in cls}  # type: ignore

        mapping: Dict[str, TaxonomicRanks] = getattr(cls, "_prefix_to_rank")  # type: ignore

        if prefix not in mapping:
            raise ValueError(f"Invalid taxonomic prefix: {prefix}")

        return mapping[prefix]

    @classmethod
    def iter_from_domain(cls) -> Iterator["TaxonomicRanks"]:
        """Yield ranks from DOMAIN (broadest) to SPECIES (most specific)."""
        rank: Optional[TaxonomicRanks] = cls.DOMAIN
        while rank is not None:
            yield rank
            rank = rank.child

    @classmethod
    def iter_from_species(cls) -> Iterator["TaxonomicRanks"]:
        """Yield ranks from SPECIES (most specific) to DOMAIN (broadest)."""
        rank: Optional[TaxonomicRanks] = cls.SPECIES
        while rank is not None:
            yield rank
            rank = rank.parent

    def iter_up(self) -> Iterator["TaxonomicRanks"]:
        """Yield ranks from the current rank up to DOMAIN (inclusive)."""
        rank: Optional[TaxonomicRanks] = self
        while rank is not None:
            yield rank
            rank = rank.parent

    def iter_down(self) -> Iterator["TaxonomicRanks"]:
        """Yield ranks from the current rank down to SPECIES (inclusive)."""
        rank: Optional[TaxonomicRanks] = self
        while rank is not None:
            yield rank
            rank = rank.child

    def get_regex(self) -> str:
        """
        Get regex pattern for matching taxonomy strings at exactly this rank.
        
        Returns:
            Regex pattern that requires this rank and excludes child ranks.
        
        Examples:
            >>> TaxonomicRanks.PHYLUM.get_regex()
            'p__[^;]+(?:;c__)?$'  # Requires p__, optionally ends with child prefix
            
            >>> TaxonomicRanks.GENUS.get_regex()
            'g__[^;]+(?:;s__)?$'  # Requires g__, optionally ends with child prefix
        """
        # Match this rank's prefix
        pattern = f"{self.prefix}[^;]+"
        
        # Optionally allow immediate child prefix at the end
        child = self.child
        if child:
            pattern += f"(?:;{child.prefix})?$"
        else:
            pattern += "$"
        
        return pattern