"""
Column Mapper - Smart Column Mapping for Schema-Agnostic Engines

Maps dataset columns to engine requirements based on semantic types and business entities,
enabling engines to work with any dataset structure.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher

from .applicability_scorer import EngineRequirements
from .schema_intelligence import BusinessEntity, ColumnProfile, SemanticType

logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """Mapping of semantic role to actual column name."""

    semantic_role: str
    column_name: str
    confidence: float
    semantic_type: SemanticType
    business_entity: BusinessEntity
    is_required: bool


@dataclass
class MappingResult:
    """Result of column mapping operation."""

    success: bool
    mappings: dict[str, ColumnMapping]
    missing_required: list[str]
    ambiguous: list[tuple[str, list[str]]]  # role -> candidate columns
    confidence: float
    message: str


class ColumnMapper:
    """
    Intelligently map dataset columns to engine requirements.

    Uses semantic types, business entities, and fuzzy name matching to find
    the best column matches for each engine's requirements.
    """

    def __init__(self, fuzzy_threshold: float = 0.6):
        """
        Initialize the column mapper.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy name matching (0-1)
        """
        self.fuzzy_threshold = fuzzy_threshold

    def map_columns(self, profiles: dict[str, ColumnProfile], requirements: EngineRequirements) -> MappingResult:
        """
        Map dataset columns to engine requirements.

        Args:
            profiles: Column profiles from ColumnProfiler
            requirements: Engine requirements definition

        Returns:
            MappingResult with column mappings and diagnostics
        """
        logger.info("Mapping columns to engine requirements")

        mappings = {}
        missing_required = []
        ambiguous = []

        # Map required semantics
        for req_semantic in requirements.required_semantics:
            role = req_semantic.value
            mapping = self._find_best_match(role, req_semantic, None, profiles, is_required=True)

            if mapping:
                mappings[role] = mapping
            else:
                missing_required.append(role)

        # Map required entities
        for req_entity in requirements.required_entities:
            role = req_entity.value
            mapping = self._find_best_match(role, None, req_entity, profiles, is_required=True)

            if mapping:
                mappings[role] = mapping
            else:
                missing_required.append(role)

        # Map optional semantics
        for role, semantic_types in requirements.optional_semantics.items():
            candidates = []
            for semantic_type in semantic_types:
                candidates.extend(
                    [
                        col
                        for col, prof in profiles.items()
                        if prof.semantic_type == semantic_type and col not in [m.column_name for m in mappings.values()]
                    ]
                )

            if len(candidates) == 1:
                prof = profiles[candidates[0]]
                mappings[role] = ColumnMapping(
                    semantic_role=role,
                    column_name=candidates[0],
                    confidence=prof.confidence,
                    semantic_type=prof.semantic_type,
                    business_entity=prof.detected_entity,
                    is_required=False,
                )
            elif len(candidates) > 1:
                ambiguous.append((role, candidates))
                # Pick the one with highest confidence
                best = max(candidates, key=lambda c: profiles[c].confidence)
                prof = profiles[best]
                mappings[role] = ColumnMapping(
                    semantic_role=role,
                    column_name=best,
                    confidence=prof.confidence * 0.8,  # Penalty for ambiguity
                    semantic_type=prof.semantic_type,
                    business_entity=prof.detected_entity,
                    is_required=False,
                )

        # Calculate overall confidence
        if mappings:
            confidence = sum(m.confidence for m in mappings.values()) / len(mappings)
        else:
            confidence = 0.0

        # Determine success
        success = len(missing_required) == 0

        # Generate message
        if success:
            if ambiguous:
                message = f"Mapping successful with {len(ambiguous)} ambiguous fields"
            else:
                message = "Mapping successful - all requirements met"
        else:
            message = f"Missing required fields: {', '.join(missing_required)}"

        return MappingResult(
            success=success,
            mappings=mappings,
            missing_required=missing_required,
            ambiguous=ambiguous,
            confidence=confidence,
            message=message,
        )

    def _find_best_match(
        self,
        role: str,
        semantic_type: SemanticType | None,
        business_entity: BusinessEntity | None,
        profiles: dict[str, ColumnProfile],
        is_required: bool,
    ) -> ColumnMapping | None:
        """
        Find the best matching column for a requirement.

        Args:
            role: Semantic role name
            semantic_type: Required semantic type (if any)
            business_entity: Required business entity (if any)
            profiles: All column profiles
            is_required: Whether this is a required field

        Returns:
            ColumnMapping if found, None otherwise
        """
        candidates = []

        # Filter by semantic type
        if semantic_type:
            candidates = [(col, prof) for col, prof in profiles.items() if prof.semantic_type == semantic_type]
        # Filter by business entity
        elif business_entity:
            candidates = [(col, prof) for col, prof in profiles.items() if prof.detected_entity == business_entity]

        if not candidates:
            # Fallback: fuzzy name matching
            candidates = [
                (col, prof) for col, prof in profiles.items() if self._fuzzy_match(role, col) >= self.fuzzy_threshold
            ]

        if not candidates:
            return None

        # If multiple candidates, pick the best one
        if len(candidates) == 1:
            col, prof = candidates[0]
            confidence = prof.confidence
        else:
            # Score candidates by:
            # 1. Name similarity
            # 2. Column confidence
            # 3. Business entity match (bonus)
            scored = []
            for col, prof in candidates:
                score = prof.confidence * 0.6
                score += self._fuzzy_match(role, col) * 0.3
                if business_entity and prof.detected_entity == business_entity:
                    score += 0.1
                scored.append((score, col, prof))

            _, col, prof = max(scored, key=lambda x: x[0])
            confidence = prof.confidence * 0.9  # Slight penalty for ambiguity

        return ColumnMapping(
            semantic_role=role,
            column_name=col,
            confidence=confidence,
            semantic_type=prof.semantic_type,
            business_entity=prof.detected_entity,
            is_required=is_required,
        )

    def _fuzzy_match(self, target: str, candidate: str) -> float:
        """
        Calculate fuzzy similarity between two strings.

        Args:
            target: Target string (e.g., 'cost')
            candidate: Candidate string (e.g., 'total_cost')

        Returns:
            Similarity score between 0 and 1
        """
        target_lower = target.lower()
        candidate_lower = candidate.lower()

        # Exact match
        if target_lower == candidate_lower:
            return 1.0

        # Substring match
        if target_lower in candidate_lower or candidate_lower in target_lower:
            return 0.9

        # Sequence matcher
        return SequenceMatcher(None, target_lower, candidate_lower).ratio()

    def resolve_ambiguity(self, mapping_result: MappingResult, resolutions: dict[str, str]) -> MappingResult:
        """
        Resolve ambiguous mappings with user input.

        Args:
            mapping_result: Original mapping result with ambiguities
            resolutions: Dict mapping role -> chosen column name

        Returns:
            Updated MappingResult with resolved ambiguities
        """
        updated_mappings = mapping_result.mappings.copy()

        for role, chosen_column in resolutions.items():
            if role in updated_mappings:
                # Update existing mapping
                old_mapping = updated_mappings[role]
                updated_mappings[role] = ColumnMapping(
                    semantic_role=role,
                    column_name=chosen_column,
                    confidence=1.0,  # User confirmed
                    semantic_type=old_mapping.semantic_type,
                    business_entity=old_mapping.business_entity,
                    is_required=old_mapping.is_required,
                )

        # Remove resolved ambiguities
        remaining_ambiguous = [
            (role, candidates) for role, candidates in mapping_result.ambiguous if role not in resolutions
        ]

        # Recalculate confidence
        if updated_mappings:
            confidence = sum(m.confidence for m in updated_mappings.values()) / len(updated_mappings)
        else:
            confidence = 0.0

        return MappingResult(
            success=len(mapping_result.missing_required) == 0,
            mappings=updated_mappings,
            missing_required=mapping_result.missing_required,
            ambiguous=remaining_ambiguous,
            confidence=confidence,
            message="Ambiguities resolved"
            if not remaining_ambiguous
            else f"{len(remaining_ambiguous)} ambiguities remaining",
        )


def auto_map_columns(profiles: dict[str, ColumnProfile], requirements: EngineRequirements) -> dict[str, str]:
    """
    Convenience function for automatic column mapping.

    Args:
        profiles: Column profiles
        requirements: Engine requirements

    Returns:
        Dict mapping semantic roles to column names
    """
    mapper = ColumnMapper()
    result = mapper.map_columns(profiles, requirements)

    if not result.success:
        logger.warning(f"Auto-mapping incomplete: {result.message}")

    return {role: mapping.column_name for role, mapping in result.mappings.items()}
