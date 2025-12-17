from enrichment.validators.base import BaseValidator

class BasicValidator(BaseValidator):
    def validate(self, data):
        return bool(data)
