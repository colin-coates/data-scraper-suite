from enrichment.normalizers.base import BaseNormalizer

class IdentityNormalizer(BaseNormalizer):
    def normalize(self, data):
        normalized = dict(data)
        if "name" in normalized:
            normalized["name"] = normalized["name"].strip().title()
        return normalized
