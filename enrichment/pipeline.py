from enrichment.types import ScrapeResult, EnrichedResult
from enrichment.normalizers.identity import IdentityNormalizer
from enrichment.validators.basic import BasicValidator
from enrichment.entity_resolution.resolver import resolve_entities

def enrich(scrape_result: ScrapeResult) -> EnrichedResult:
    normalizer = IdentityNormalizer()
    validator = BasicValidator()

    data = normalizer.normalize(scrape_result.raw_data)
    data = resolve_entities(data)

    valid = validator.validate(data)

    return EnrichedResult(
        scraper_name=scrape_result.scraper_name,
        enriched_data=data if valid else {},
        confidence=1.0 if valid else 0.0,
        notes=None if valid else "Validation failed"
    )
