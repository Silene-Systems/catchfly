"""Property-based tests for normalization invariants using Hypothesis."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from catchfly.normalization.dictionary import DictionaryNormalization

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Reasonable text values (non-empty, printable, bounded length)
text_values = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z"), max_codepoint=0x7E),
    min_size=1,
    max_size=30,
)

# Non-empty lists of text values
value_lists = st.lists(text_values, min_size=1, max_size=20)

# Mapping dicts: non-empty, mapping strings to strings
mapping_dicts = st.dictionaries(
    keys=text_values,
    values=text_values,
    min_size=1,
    max_size=15,
)


# ---------------------------------------------------------------------------
# Property 1: Every input value appears in the output mapping
# ---------------------------------------------------------------------------


class TestEveryInputMapped:
    @given(mapping=mapping_dicts, extra_values=value_lists)
    @settings(max_examples=50)
    def test_all_values_in_mapping(
        self, mapping: dict[str, str], extra_values: list[str]
    ) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, passthrough_unmapped=True
        )
        # Combine mapped keys with extra values
        values = list(mapping.keys()) + extra_values
        result = normalizer.normalize(values, context_field="test")

        unique_values = set(values)
        for v in unique_values:
            assert v in result.mapping, f"Value '{v}' missing from mapping"


# ---------------------------------------------------------------------------
# Property 2: All mapping values are strings
# ---------------------------------------------------------------------------


class TestMappingValuesAreStrings:
    @given(mapping=mapping_dicts, values=value_lists)
    @settings(max_examples=50)
    def test_values_are_strings(
        self, mapping: dict[str, str], values: list[str]
    ) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, passthrough_unmapped=True
        )
        result = normalizer.normalize(values, context_field="test")

        for _key, val in result.mapping.items():
            assert isinstance(val, str)


# ---------------------------------------------------------------------------
# Property 3: Mapping is idempotent — normalizing canonical values returns unchanged
# ---------------------------------------------------------------------------


class TestIdempotent:
    @given(mapping=mapping_dicts)
    @settings(max_examples=50)
    def test_canonical_values_stable(self, mapping: dict[str, str]) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, passthrough_unmapped=True
        )
        # First pass: normalize the keys
        keys = list(mapping.keys())
        result1 = normalizer.normalize(keys, context_field="test")

        # Collect canonical values from first pass
        canonical_values = list(set(result1.mapping.values()))
        if not canonical_values:
            return

        # Second pass: normalize the canonical values
        result2 = normalizer.normalize(canonical_values, context_field="test")

        # Each canonical that is itself a key in the mapping should map
        # to its own target; those that are not keys should pass through
        for cv in canonical_values:
            if cv in mapping:
                # The canonical IS a key, so it maps to mapping[cv]
                assert result2.mapping[cv] == mapping[cv]
            else:
                # The canonical is NOT a key, so it passes through to itself
                assert result2.mapping[cv] == cv


# ---------------------------------------------------------------------------
# Property 4: case_insensitive=True — "NYC" and "nyc" map to same canonical
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    @given(mapping=mapping_dicts)
    @settings(max_examples=50)
    def test_case_variants_same_canonical(self, mapping: dict[str, str]) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, case_insensitive=True, passthrough_unmapped=True
        )

        for key in mapping:
            variants = [key, key.lower(), key.upper()]
            result = normalizer.normalize(variants, context_field="test")

            # All case variants should map to the same canonical
            canonicals = {result.mapping[v] for v in variants if v in result.mapping}
            assert len(canonicals) == 1, (
                f"Case variants of '{key}' mapped to different canonicals: {canonicals}"
            )

    def test_explicit_case_example(self) -> None:
        normalizer = DictionaryNormalization(
            mapping={"NYC": "New York"},
            case_insensitive=True,
        )
        result = normalizer.normalize(
            ["NYC", "nyc", "Nyc", "nyC"], context_field="city"
        )
        canonicals = set(result.mapping.values())
        assert canonicals == {"New York"}


# ---------------------------------------------------------------------------
# Property 5: passthrough_unmapped=True — unmapped values map to themselves
# ---------------------------------------------------------------------------


class TestPassthroughUnmapped:
    @given(mapping=mapping_dicts, unmapped=value_lists)
    @settings(max_examples=50)
    def test_unmapped_values_passthrough(
        self, mapping: dict[str, str], unmapped: list[str]
    ) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, passthrough_unmapped=True
        )
        # Only use values NOT in the mapping
        truly_unmapped = [v for v in unmapped if v not in mapping]
        if not truly_unmapped:
            return

        result = normalizer.normalize(truly_unmapped, context_field="test")

        for v in set(truly_unmapped):
            assert result.mapping[v] == v, (
                f"Unmapped value '{v}' should map to itself, got '{result.mapping[v]}'"
            )

    def test_explicit_passthrough_example(self) -> None:
        normalizer = DictionaryNormalization(
            mapping={"NYC": "New York"},
            passthrough_unmapped=True,
        )
        result = normalizer.normalize(
            ["NYC", "Chicago", "Denver"], context_field="city"
        )
        assert result.mapping["NYC"] == "New York"
        assert result.mapping["Chicago"] == "Chicago"
        assert result.mapping["Denver"] == "Denver"


# ---------------------------------------------------------------------------
# Property 6 (bonus): empty input always gives empty mapping
# ---------------------------------------------------------------------------


class TestEmptyInput:
    @given(mapping=mapping_dicts)
    @settings(max_examples=20)
    def test_empty_values_empty_mapping(self, mapping: dict[str, str]) -> None:
        normalizer = DictionaryNormalization(
            mapping=mapping, passthrough_unmapped=True
        )
        result = normalizer.normalize([], context_field="test")
        assert result.mapping == {}
